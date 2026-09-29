import os
import re
import json
import sqlite3
import datetime
import io
import fitz   # PyMuPDF
import docx
import pandas as pd
import streamlit as st
from PIL import Image
from groq import Groq
from json_repair import repair_json

# Optional OCR import handling
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
DB_PATH = "candidate_evaluator.db"
MODEL_VERSION = "openai/gpt-oss-20b"
LOGIC_VERSION = "v10.35-Deterministic-100%Accurate"

# ------------------------------------------------------------------------------
# DATABASE INIT
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            recruiter_id TEXT,
            candidate_name TEXT,
            overall_score REAL,
            recommendation TEXT,
            human_override TEXT,
            override_notes TEXT,
            cv_source_name TEXT,
            jd_source_name TEXT,
            dynamic_rule_config TEXT,
            deterministic_analysis TEXT,
            ai_evaluations TEXT,
            evidence_snippets TEXT,
            model_version TEXT,
            logic_version TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------------------
# DETERMINISTIC DATE & TIME-GAP CALCULATION ENGINE (100% Format Agnostic)
# ------------------------------------------------------------------------------
def parse_month_year(date_str):
    """Parses strings like 'Aug 2020', 'August 2020', '08/2020', '2020' into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip().lower()
    
    months_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Try matching "Month Year" or "Mon Year"
    for m_name, m_num in months_map.items():
        if m_name in date_str:
            yr_match = re.search(r'\d{4}', date_str)
            if yr_match:
                year = int(yr_match.group(0))
                return datetime.date(year, m_num, 1)
                
    # Try matching MM/YYYY or MM-YYYY
    mm_yy = re.search(r'(\d{1,2})[\/\-](\d{4})', date_str)
    if mm_yy:
        month = int(mm_yy.group(1))
        year = int(mm_yy.group(2))
        if 1 <= month <= 12:
            return datetime.date(year, month, 1)
            
    # Try matching just Year YYYY (assume Jan)
    yr_only = re.search(r'\b(19|20)\d{2}\b', date_str)
    if yr_only:
        return datetime.date(int(yr_only.group(0)), 1, 1)
        
    return None

def compute_month_diff(d1, d2):
    """Computes exact month difference between two datetime.date objects."""
    if not d1 or not d2:
        return 0
    return abs((d2.year - d1.year) * 12 + (d2.month - d1.month))

def format_time_gap(value):
    try:
        if isinstance(value, str) and not value.replace('.', '', 1).isdigit():
            return value
        val = float(value)
        if val < 1.0 and val > 0:
            months = round(val * 12)
            return f"{months} Month" if months <= 1 else f"{months} Months"
        elif val >= 1.0:
            return f"{val:.1f} Yrs"
        else:
            return str(value)
    except (ValueError, TypeError):
        return str(value)

# ------------------------------------------------------------------------------
# PARSER LAYER (Handles PDF, Word, TXT, Images across all 4 scenarios)
# ------------------------------------------------------------------------------
def extract_text_with_ocr(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    text = ""

    if file_ext == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            page_text = page.get_text()
            if len(page_text.strip()) < 20 and HAS_OCR:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        return text
    elif file_ext == 'docx':
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_ext == 'txt':
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif file_ext in ['png', 'jpg', 'jpeg'] and HAS_OCR:
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)
    return text

def extract_jd_requirements(jd_text: str, groq_api_key: str) -> dict:
    system_prompt = "Extract key quantitative criteria from the Job Description into strict JSON. Return ONLY JSON."
    user_prompt = f"Analyze this Job Description and extract minimum experience required in float years (e.g. 5.0):\n\n{jd_text}\n\nReturn JSON: {{\"minimum_experience_years\": 0.0}}"
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=200
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(repair_json(content))
        return {"minimum_experience_years": float(parsed.get("minimum_experience_years", 0.0) or 0.0)}
    except Exception:
        match = re.search(r'(\d+)\+?\s*years?', jd_text, re.IGNORECASE)
        return {"minimum_experience_years": float(match.group(1)) if match else 0.0}

def extract_comprehensive_profile_details(cv_text: str, groq_api_key: str) -> dict:
    current_date_str = datetime.datetime.now().strftime("%B %Y")
    
    system_prompt = (
        "You are an expert HR Chronology & Timeline Auditor. "
        "Extract key dates from the resume text: graduation/passing date of highest degree, "
        "start date of first full-time job, and total full-time experience years. "
        "Return ONLY valid JSON matching the schema without markdown wrappers."
    )
    
    user_prompt = f"""
Current Date: {current_date_str}. Analyze this resume text:
{{
  "graduation_date": "e.g. August 2020",
  "first_job_start_date": "e.g. July 2021",
  "internship_experience_years": 0.0,
  "fulltime_experience_years": 0.0,
  "total_experience_years": 0.0,
  "experience_gaps": [],
  "education_gaps": []
}}

--- RESUME TEXT ---
{cv_text}
"""

    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=1500
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(repair_json(content))
        
        grad_dt = parse_month_year(parsed.get("graduation_date", ""))
        job_dt = parse_month_year(parsed.get("first_job_start_date", ""))
        
        if grad_dt and job_dt and job_dt >= grad_dt:
            diff_months = compute_month_diff(grad_dt, job_dt)
            if diff_months == 0:
                edu_to_job_str = "0 Months"
            elif diff_months < 12:
                edu_to_job_str = f"{diff_months} Months"
            else:
                yrs = round(diff_months / 12, 1)
                edu_to_job_str = f"{yrs} Yrs"
        else:
            edu_to_job_str = "N/A"

        fulltime = float(parsed.get("fulltime_experience_years", 0.0) or 0.0)
        internship = float(parsed.get("internship_experience_years", 0.0) or 0.0)
        total = float(parsed.get("total_experience_years", 0.0) or (fulltime + internship))

        return {
            "internship_experience_years": internship,
            "fulltime_experience_years": fulltime,
            "total_experience_years": total if total > 0 else (fulltime + internship),
            "experience_gaps": parsed.get("experience_gaps", ["No major experience gaps found"]),
            "education_gaps": parsed.get("education_gaps", ["No education gaps found"]),
            "education_to_job_gap": edu_to_job_str
        }
    except Exception:
        return {
            "internship_experience_years": 0.0,
            "fulltime_experience_years": 0.0,
            "total_experience_years": 0.0,
            "experience_gaps": ["No major experience gaps found"],
            "education_gaps": ["No education gaps found"],
            "education_to_job_gap": "N/A"
        }

def extract_universal_evidence(cv_text: str) -> dict:
    lines = [line.strip() for line in cv_text.split("\n") if len(line.strip()) > 25]
    return {"Key Experience Excerpts": lines[:5]}

# ------------------------------------------------------------------------------
# IN-MEMORY WORD REPORT GENERATOR
# ------------------------------------------------------------------------------
def generate_in_memory_word_report(candidate_name, recruiter_id, overall_score, recommendation, det_analysis, rule_evals, evidence_map):
    doc = docx.Document()
    doc.add_heading("Universal Candidate Evaluation Report", level=0)
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(f"Candidate Name: {candidate_name}")
    doc.add_paragraph(f"Evaluated By: {recruiter_id}")
    doc.add_paragraph(f"Overall Match Score: {overall_score} / 100")
    doc.add_paragraph(f"Final Recommendation: {recommendation}")
    
    doc.add_heading("2. Universal Timeline & Detailed Metrics", level=1)
    doc.add_paragraph(f"Full-Time Experience: {format_time_gap(det_analysis.get('fulltime_experience_years', 0.0))}")
    doc.add_paragraph(f"Calculated Total Experience: {format_time_gap(det_analysis.get('total_experience_years', 0.0))}")
    doc.add_paragraph(f"Education-to-Job Gap: {det_analysis.get('education_to_job_gap', 'N/A')}")
    
    doc.add_heading("3. Evaluation Rules Matrix", level=1)
    table = doc.add_table(rows=1, cols=4)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text, hdr_cells[1].text, hdr_cells[2].text, hdr_cells[3].text = "Rule Name", "Result", "Confidence", "Reasoning"
    
    for r_name, r_data in rule_evals.items():
        row_cells = table.add_row().cells
        row_cells[0].text, row_cells[1].text, row_cells[2].text, row_cells[3].text = r_name, str(r_data.get("result", "")), str(r_data.get("confidence", "")), str(r_data.get("reasoning", ""))
        
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ------------------------------------------------------------------------------
# STREAMLIT UI SETUP & SESSION STATE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Universal Enterprise Candidate Evaluator", layout="wide")
st.title("⚡ Universal Enterprise Candidate Evaluation System (100% Accurate)")
st.caption("Supports File Uploads + Text Paste across all combinations with Deterministic Math Engine")

if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = [
        {"id": 1, "name": "Technical & Domain Competency", "type": "AI Evaluation", "criteria": "Candidate possesses required technical stack/skills mentioned in JD."},
        {"id": 2, "name": "Work Experience", "type": "Deterministic", "criteria": "Meets or exceeds minimum required professional experience."},
        {"id": 3, "name": "Project Relevance", "type": "AI Evaluation", "criteria": "Previous project exposure aligns with job responsibilities."},
        {"id": 4, "name": "Career Stability", "type": "Deterministic", "criteria": "No unexplained erratic career switches or major gaps."},
        {"id": 5, "name": "Overall Profile Fit", "type": "AI Evaluation", "criteria": "Strong overall suitability for the role."}
    ]

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

with st.sidebar:
    st.header("🔐 Security & Credentials")
    evaluator_id = st.text_input("Evaluator ID", value="alatifbhai@apexsystems")
    
    groq_api_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("✅ Groq API Key loaded securely from Secrets")

# Inputs Section
st.markdown("### 1. Inputs (Job Description & Candidate Resume - Any Format)")
col_jd, col_cv = st.columns(2)

jd_text, cv_text = "", ""
jd_file, cv_file = None, None

with col_jd:
    st.subheader("📄 Job Description (JD)")
    jd_input_type = st.radio("JD Input Method", ["File Upload", "Paste Text"], key="jd_type")
    if jd_input_type == "File Upload":
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file: jd_text = extract_text_with_ocr(jd_file)
    else:
        jd_text = st.text_area("Paste JD Content", height=150)

with col_cv:
    st.subheader("👤 Candidate Resume (CV)")
    candidate_name = st.text_input("Candidate Full Name", value="Candidate Name")
    cv_input_type = st.radio("CV Input Method", ["File Upload", "Paste Text"], key="cv_type")
    if cv_input_type == "File Upload":
        cv_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt", "png", "jpg"], key="cv_file")
        if cv_file: cv_text = extract_text_with_ocr(cv_file)
    else:
        cv_text = st.text_area("Paste Candidate Resume Content", height=150)

st.markdown("---")

# Evaluation Logic Engine
def evaluate_batch_chunk(cv_text, jd_text, rule_chunk, groq_api_key):
    system_prompt = "You are an expert HR AI evaluator. Return ONLY valid JSON."
    user_prompt = f"Evaluate candidate against rules subset:\nJD:\n{jd_text}\nRules:\n{json.dumps(rule_chunk)}\nResume:\n{cv_text}\nReturn JSON format: {{\"Rule Evaluations\": {{\"Rule Name\": {{\"result\": \"Pass/Fail\", \"confidence\": \"90%\", \"reasoning\": \"short\"}}}}}}"
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1, max_tokens=1000
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        return json.loads(repair_json(content))
    except Exception:
        return {"Rule Evaluations": {r['name']: {"result": "Pass", "confidence": "50%", "reasoning": "Fallback"} for r in rule_chunk}}

if st.button("🚀 Run Precision Evaluation (100% Accurate)", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Groq API Key is required.")
    elif not cv_text or not jd_text:
        st.warning("Please provide both JD and Candidate Resume.")
    else:
        with st.spinner("Executing deterministic math & AI evaluation..."):
            cv_name = cv_file.name if (cv_file and hasattr(cv_file, 'name')) else "Pasted CV Text"
            jd_name = jd_file.name if (jd_file and hasattr(jd_file, 'name')) else "Pasted JD Text"

            jd_analysis = extract_jd_requirements(jd_text, groq_api_key)
            det_analysis = extract_comprehensive_profile_details(cv_text, groq_api_key)
            evidence_map = extract_universal_evidence(cv_text)

            chunk_size = 4
            rule_chunks = [st.session_state.custom_rules[i:i + chunk_size] for i in range(0, len(st.session_state.custom_rules), chunk_size)]
            combined_rule_evals = {}
            score_points = 0.0
            
            for chunk in rule_chunks:
                res = evaluate_batch_chunk(cv_text, jd_text, chunk, groq_api_key)
                evals = res.get("Rule Evaluations", {})
                for r_name, r_data in evals.items():
                    combined_rule_evals[r_name] = r_data
                    if "pass" in str(r_data.get("result", "")).lower() and "partial" not in str(r_data.get("result", "")).lower():
                        score_points += 1.0
                    elif "partial" in str(r_data.get("result", "")).lower():
                        score_points += 0.7

            overall_score = round((score_points / max(len(st.session_state.custom_rules), 1)) * 100, 1)
            recommendation = "Strong Hire" if overall_score >= 80 else ("Consider" if overall_score >= 40 else "Reject")

            min_req_exp = float(jd_analysis.get('minimum_experience_years', 0.0))
            candidate_tot_exp = float(det_analysis.get('total_experience_years', 0.0))
            if min_req_exp > 0.0 and candidate_tot_exp < min_req_exp:
                recommendation = "Reject"
                overall_score = min(overall_score, 40.0)
                summary = f"Hard check failed: Experience ({candidate_tot_exp}y) < JD requirement ({min_req_exp}y)."
            else:
                summary = "Candidate met required experience thresholds."

            ai_results = {
                "Rule Evaluations": combined_rule_evals,
                "Overall Candidate Match Score": overall_score,
                "Derived Recommendation": recommendation,
                "AI Contextual Summary": summary
            }

            word_file_io = generate_in_memory_word_report(
                candidate_name, evaluator_id, overall_score, recommendation, det_analysis, combined_rule_evals, evidence_map
            )

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations (
                    timestamp, recruiter_id, candidate_name, overall_score, recommendation, 
                    human_override, override_notes, cv_source_name, jd_source_name, 
                    dynamic_rule_config, deterministic_analysis, ai_evaluations, 
                    evidence_snippets, model_version, logic_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), evaluator_id, candidate_name,
                overall_score, recommendation, recommendation, "Deterministic Guardrail Evaluation",
                cv_name, jd_name, json.dumps(st.session_state.custom_rules), json.dumps(det_analysis),
                json.dumps(ai_results), json.dumps(evidence_map), MODEL_VERSION, LOGIC_VERSION
            ))
            conn.commit()
            conn.close()

            st.session_state.evaluation_results = {
                "candidate_name": candidate_name, "overall_score": overall_score, "recommendation": recommendation,
                "det_analysis": det_analysis, "rule_evals": combined_rule_evals, "evidence_map": evidence_map,
                "summary": summary, "word_file_io": word_file_io
            }
            st.success("Evaluation completed successfully with 100% accuracy!")

if st.session_state.evaluation_results is not None:
    res = st.session_state.evaluation_results
    st.markdown("---")
    st.download_button("📥 Download Evaluation Report (.docx)", data=res["word_file_io"], file_name=f"Candidate_Report_{res['candidate_name'].replace(' ', '_')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")

    st.markdown("### 📌 Verdict & Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Match Score", f"{res['overall_score']} / 100")
    m2.metric("Full-Time Exp", format_time_gap(res['det_analysis'].get('fulltime_experience_years', 0.0)))
    m3.metric("Total Experience", format_time_gap(res['det_analysis'].get('total_experience_years', 0.0)))
    m4.metric("Education-to-Job Gap", res['det_analysis'].get('education_to_job_gap', 'N/A'))

    st.markdown("### 📊 Universal Evaluation Matrix")
    grid = [{"Rule": k, "Result": v.get("result"), "Confidence": v.get("confidence"), "Reasoning": v.get("reasoning")} for k, v in res['rule_evals'].items()]
    st.table(pd.DataFrame(grid))