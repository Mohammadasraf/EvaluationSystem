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
LOGIC_VERSION = "v10.42-Accurate-Education-Gap-Fix"

# ------------------------------------------------------------------------------
# DATABASE INIT (With Automatic Schema Alignment)
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(evaluations)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if columns and "cv_source_name" not in columns:
        cursor.execute("DROP TABLE IF EXISTS evaluations")
        
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
# HELPER: FORMAT TIME GAPS (Years / Months)
# ------------------------------------------------------------------------------
def format_time_gap(value):
    try:
        if isinstance(value, str) and not value.replace('.', '', 1).isdigit():
            return value
            
        val = float(value)
        if val < 1.0 and val > 0:
            months = round(val * 12)
            if months <= 1:
                return f"{months} Month"
            else:
                return f"{months} Months"
        elif val >= 1.0:
            return f"{val:.1f} Yrs"
        else:
            return str(value)
    except (ValueError, TypeError):
        return str(value)

# ------------------------------------------------------------------------------
# DETERMINISTIC DATE PARSING & CALCULATION ENGINE
# ------------------------------------------------------------------------------
def parse_date_str(date_str):
    if not date_str:
        return None
    
    date_str = date_str.replace("'", "").replace("'", "").replace("`", "").strip().lower()
    
    if "present" in date_str or "current" in date_str or "till date" in date_str:
        return datetime.datetime.now()
    
    match = re.search(r'([a-z]+)\s*(\d{2,4})', date_str)
    if match:
        month_str, year_str = match.groups()
        if len(year_str) == 2:
            year_str = "20" + year_str
        try:
            month = 0
            month_lower = month_str[:3]
            months_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            if month_lower in months_list:
                month = months_list.index(month_lower) + 1
            if month == 0:
                return None
            return datetime.datetime(int(year_str), month, 1)
        except ValueError:
            pass
    return None

def extract_experience_via_regex(text: str) -> float:
    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\+?\s*years?',
        r'(\d+(?:\.\d+)?)\+?\s*yrs?'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0

def calculate_deterministic_timeline(cv_text: str) -> dict:
    cleaned_cv = cv_text.replace("'", "").replace("'", "")
    date_range_pattern = r'([A-Za-z]+\s*\d{2,4})\s*[\–\-\to]\s*([A-Za-z]+\s*\d{2,4}|Present|Current|Till Date)'
    matches = re.findall(date_range_pattern, cleaned_cv, re.IGNORECASE)
    
    internship_months = 0
    fulltime_months = 0
    
    work_periods = []
    for start_str, end_str in matches:
        pos = cleaned_cv.find(start_str)
        snippet_context = cleaned_cv[max(0, pos - 120): pos + 50].lower() if pos != -1 else ""
        
        edu_keywords = ['b.tech', 'b.e', 'm.tech', 'b.sc', 'm.sc', 'graduation', 'degree', 'cgpa', 'college', 'university', 'school', 'education']
        if any(kw in snippet_context for kw in edu_keywords):
            continue

        start_dt = parse_date_str(start_str)
        end_dt = parse_date_str(end_str)
        if start_dt and end_dt and start_dt <= end_dt:
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
            
            if "intern" in snippet_context or "trainee" in snippet_context or "internship" in snippet_context:
                internship_months += months
                work_periods.append((start_dt, end_dt, "internship"))
            else:
                fulltime_months += months
                work_periods.append((start_dt, end_dt, "fulltime"))

    work_periods = sorted(work_periods, key=lambda x: x[0])

    exp_gaps = []
    for i in range(len(work_periods) - 1):
        current_end = work_periods[i][1]
        next_start = work_periods[i+1][0]
        if next_start > current_end:
            gap_months = (next_start.year - current_end.year) * 12 + (next_start.month - current_end.month) - 1
            if gap_months >= 2:
                exp_gaps.append(f"Gap of {gap_months} months between {current_end.strftime('%b %Y')} and {next_start.strftime('%b %Y')}")

    if not exp_gaps:
        exp_gaps = ["No major experience gaps found"]

    calculated_total_years = round((fulltime_months + internship_months) / 12.0, 1)
    if calculated_total_years == 0.0:
        calculated_total_years = extract_experience_via_regex(cv_text)
        fulltime_months = int(calculated_total_years * 12)

    # --- ACCURATE EDUCATION END DATE PARSING ---
    # Look specifically for education section range end date (e.g., Jun 2016 – Aug 2020 -> picks Aug 2020)
    edu_end_date = None
    edu_section_match = re.search(r'(education|b\.?tech|b\.?e\.?|graduation).*?(20\d{2})', cleaned_cv, re.IGNORECASE)
    if edu_section_match:
        # Search for all date ranges within 300 chars of education keyword
        edu_pos = cleaned_cv.lower().find('education')
        if edu_pos == -1:
            edu_pos = 0
        edu_subtext = cleaned_cv[edu_pos:edu_pos + 400]
        sub_ranges = re.findall(r'([A-Za-z]+\s*\d{2,4})\s*[\–\-\to]\s*([A-Za-z]+\s*\d{2,4})', edu_subtext)
        if sub_ranges:
            # Take the end date of the last matched range in education section
            edu_end_date = parse_date_str(sub_ranges[-1][1])

    education_to_job_gap_str = "Seamless Transition"
    if edu_end_date and work_periods:
        first_work_start = work_periods[0][0]
        if first_work_start > edu_end_date:
            gap_months = (first_work_start.year - edu_end_date.year) * 12 + (first_work_start.month - edu_end_date.month) - 1
            if gap_months >= 2:
                education_to_job_gap_str = f"{gap_months} Months Gap"
            else:
                education_to_job_gap_str = "Seamless Transition"

    return {
        "internship_experience_years": round(internship_months / 12.0, 1),
        "fulltime_experience_years": round(fulltime_months / 12.0, 1),
        "total_experience_years": max(calculated_total_years, round((fulltime_months + internship_months) / 12.0, 1)),
        "experience_gaps": exp_gaps,
        "education_gaps": ["No education gaps found"],
        "education_to_job_gap": education_to_job_gap_str
    }

# ------------------------------------------------------------------------------
# PARSER LAYER (In-Memory Processing)
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
    user_prompt = f"""
Analyze this Job Description and extract:
1. "minimum_experience_years": float number representing minimum years required (e.g., 6.0 for 6 years). If not specified, return 0.0.

JD TEXT:
{jd_text}

Return JSON:
{{
  "minimum_experience_years": 0.0
}}
"""
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(repair_json(content))
        return {"minimum_experience_years": float(parsed.get("minimum_experience_years", 0.0) or 0.0)}
    except Exception:
        match = re.search(r'(\d+)\+?\s*years?', jd_text, re.IGNORECASE)
        val = float(match.group(1)) if match else 0.0
        return {"minimum_experience_years": val}

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
    doc.add_paragraph(f"Evaluation Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Overall Match Score: {overall_score} / 100")
    doc.add_paragraph(f"Final Recommendation: {recommendation}")
    
    doc.add_heading("2. Universal Timeline & Detailed Metrics", level=1)
    doc.add_paragraph(f"Full-Time Experience: {format_time_gap(det_analysis.get('fulltime_experience_years', 0.0))}")
    doc.add_paragraph(f"Internship Experience: {format_time_gap(det_analysis.get('internship_experience_years', 0.0))}")
    doc.add_paragraph(f"Calculated Total Experience: {format_time_gap(det_analysis.get('total_experience_years', 0.0))}")
    doc.add_paragraph(f"Education-to-Job Gap: {format_time_gap(det_analysis.get('education_to_job_gap', 'Seamless Transition'))}")
    
    doc.add_heading("3. Evaluation Rules Matrix", level=1)
    table = doc.add_table(rows=1, cols=4)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Rule Name"
    hdr_cells[1].text = "Result"
    hdr_cells[2].text = "Confidence"
    hdr_cells[3].text = "Reasoning"
    
    for r_name, r_data in rule_evals.items():
        row_cells = table.add_row().cells
        row_cells[0].text = r_name
        row_cells[1].text = str(r_data.get("result", ""))
        row_cells[2].text = str(r_data.get("confidence", ""))
        row_cells[3].text = str(r_data.get("reasoning", ""))
        
    doc.add_heading("4. Profile Excerpts", level=1)
    for sk, snippets in evidence_map.items():
        doc.add_paragraph(f"Category: {sk}", style='List Bullet')
        for snip in snippets:
            doc.add_paragraph(f'"{snip}"', style='Intense Quote')
            
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ------------------------------------------------------------------------------
# STREAMLIT UI SETUP & SESSION STATE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Universal Enterprise Candidate Evaluator", layout="wide")

st.title("⚡ Universal Enterprise Candidate Evaluation System (In-Memory)")
st.caption("Deterministic Math Engine + LLM Precision + 100% Browser-Based Processing")

if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = [
        {"id": 1, "name": "Technical & Domain Competency", "type": "AI Evaluation", "criteria": "Candidate possesses required technical stack/skills mentioned in JD. (Allow partial match if core stack is strong and secondary tools are missing)."},
        {"id": 2, "name": "Work Experience", "type": "Deterministic", "criteria": "Meets or exceeds minimum required professional experience."},
        {"id": 3, "name": "Project Relevance", "type": "AI Evaluation", "criteria": "Previous project exposure aligns with job responsibilities."},
        {"id": 4, "name": "Career Stability", "type": "Deterministic", "criteria": "No unexplained erratic career switches or major gaps."},
        {"id": 5, "name": "Overall Profile Fit", "type": "AI Evaluation", "criteria": "Strong overall suitability for the role."}
    ]

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

with st.sidebar:
    st.header("🔐 Security & Credentials")
    evaluator_id = st.text_input("Evaluator / User ID", value="alatifbhai@apexsystems")

    groq_api_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    if not groq_api_key:
        if "groq_api_key_input" not in st.session_state:
            st.session_state.groq_api_key_input = ""
            
        groq_api_key = st.text_input(
            "Groq API Key", 
            value=st.session_state.groq_api_key_input, 
            type="password"
        )
        st.session_state.groq_api_key_input = groq_api_key
    else:
        st.success("✅ Groq API Key loaded securely from Secrets")
        
    st.info(f"Model: {MODEL_VERSION}\nLogic: {LOGIC_VERSION}\nStorage: In-Memory / Browser Only")

# Section 1: Inputs
st.markdown("### 1. Inputs (Job Description & Candidate Resume)")

jd_file = None
cv_file = None
jd_text = ""
cv_text = ""
candidate_name = "Candidate Name"

col_jd, col_cv = st.columns(2)

with col_jd:
    st.subheader("📄 Job Description (JD)")
    jd_input_type = st.radio("JD Input Method", ["File Upload", "Paste Text"], key="jd_type")
    if jd_input_type == "File Upload":
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file:
            jd_text = extract_text_with_ocr(jd_file)
    else:
        jd_text = st.text_area("Paste JD Content", height=150)

with col_cv:
    st.subheader("👤 Candidate Resume (CV)")
    candidate_name = st.text_input("Candidate Full Name", value="Candidate Name")
    cv_input_type = st.radio("CV Input Method", ["File Upload", "Paste Text"], key="cv_type")
    if cv_input_type == "File Upload":
        cv_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt", "png", "jpg"], key="cv_file")
        if cv_file:
            cv_text = extract_text_with_ocr(cv_file)
    else:
        cv_text = st.text_area("Paste Candidate Resume Content", height=150)

st.markdown("---")

# Section 2: Rules Builder
st.markdown("### 2. 🎛️ N-Rules Builder")
with st.expander("➕ Manage Custom Evaluation Rules", expanded=False):
    new_name = st.text_input("Rule Name")
    new_type = st.selectbox("Rule Type", ["Deterministic", "Skill Check", "Compliance", "Custom"])
    new_criteria = st.text_area("Rule Description / Criteria")
    
    if st.button("Add Rule"):
        if new_name and new_criteria:
            st.session_state.custom_rules.append({
                "id": len(st.session_state.custom_rules)+1,
                "name": new_name,
                "type": new_type,
                "criteria": new_criteria
            })
            st.success("Rule added successfully!")
            st.rerun()
        else:
            st.warning("Please fill both Rule Name and Criteria.")

rules_to_keep = []
for idx, rule in enumerate(st.session_state.custom_rules):
    c1, c2, c3 = st.columns([1, 4, 1])
    c1.markdown(f"**Rule {idx+1}**")
    c2.markdown(f"**{rule['name']}**: {rule['criteria']}")
    if c3.button("❌", key=f"del_{rule['id']}"):
        continue
    rules_to_keep.append(rule)
st.session_state.custom_rules = rules_to_keep

st.markdown("---")

# ------------------------------------------------------------------------------
# EVALUATION & GUARDRAIL ENGINE
# ------------------------------------------------------------------------------
def evaluate_batch_chunk(cv_text, jd_text, rule_chunk, groq_api_key):
    system_prompt = (
        "You are an expert HR AI evaluator. "
        "Evaluate rules using three statuses: 'Pass', 'Partial Match', or 'Fail'. "
        "Return ONLY valid JSON format matching the requested structure without markdown code blocks."
    )
    
    user_prompt = f"""
Evaluate the candidate against the rules subset.

--- JOB DESCRIPTION ---
{jd_text}

--- RULES SUBSET ---
{json.dumps(rule_chunk, indent=2)}

--- RESUME ---
{cv_text}

Return ONLY valid JSON:
{{
  "Rule Evaluations": {{
    "Rule Name": {{"result": "Pass / Partial Match / Fail", "confidence": "90%", "reasoning": "Short objective explanation under 15 words."}}
  }}
}}
"""
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("