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
LOGIC_VERSION = "v10.50-LineByLine-Timeline-Fix"

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
# UNIVERSAL DETERMINISTIC DATE PARSING & CALCULATION ENGINE
# ------------------------------------------------------------------------------
def parse_date_str(date_str):
    if not date_str:
        return None
    
    date_str = str(date_str).replace("'", "").replace("`", "").strip().lower()
    
    if any(kw in date_str for kw in ["present", "current", "till date", "continuing", "now"]):
        return datetime.datetime.now()
    
    # Format 1: Month and Year (e.g., 'Jan 2022' or 'January 22' or 'Jun '16')
    match = re.search(r'([a-z]+)\.?\s*[\'`]?(\d{2,4})', date_str)
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
            if month > 0:
                return datetime.datetime(int(year_str), month, 1)
        except ValueError:
            pass

    # Format 2: Numeric MM/YYYY or MM-YYYY or YYYY
    match_num = re.search(r'(\d{1,2})[/\-](\d{2,4})', date_str)
    if match_num:
        m, y = match_num.groups()
        if len(y) == 2:
            y = "20" + y
        try:
            return datetime.datetime(int(y), int(m), 1)
        except ValueError:
            pass

    # Format 3: Only Year (e.g., '2020')
    match_yr = re.search(r'\b(20\d{2}|19\d{2})\b', date_str)
    if match_yr:
        try:
            return datetime.datetime(int(match_yr.group(1)), 1, 1)
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
    cleaned_cv = cv_text.replace("'", "").replace("`", "")
    lines = cleaned_cv.split('\n')
    
    internship_months = 0
    fulltime_months = 0
    work_periods = []
    
    # Line-by-line robust parsing for date ranges including 'Present'
    for line in lines:
        line_lower = line.lower()
        
        # Skip education lines
        edu_keywords = ['b.tech', 'b.e', 'm.tech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba', 'graduation', 'degree', 'cgpa', 'college', 'university', 'school', 'education', 'hsc', 'ssc']
        if any(kw in line_lower for kw in edu_keywords):
            continue

        # Look for date separators (–, -, to)
        parts = re.split(r'\s*(?:–|-|to)\s*', line, flags=re.IGNORECASE)
        if len(parts) >= 2:
            for i in range(len(parts) - 1):
                start_str = parts[i]
                end_str = parts[i+1]
                
                # Extract potential date strings from the split chunks
                start_dt = parse_date_str(start_str)
                end_dt = parse_date_str(end_str)
                
                if start_dt and end_dt and start_dt <= end_dt:
                    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                    if 0 < months <= 600: # sanity check (max 50 years per stint)
                        if "intern" in line_lower or "trainee" in line_lower or "internship" in line_lower:
                            internship_months += months
                            work_periods.append((start_dt, end_dt, "internship"))
                        else:
                            fulltime_months += months
                            work_periods.append((start_dt, end_dt, "fulltime"))

    # Fallback to regex pattern scan if line splitting missed anything
    if fulltime_months == 0:
        date_range_pattern = r'([A-Za-z0-9\/\-\.\s]{3,15})\s*(?:–|-|to)\s*([A-Za-z0-9\/\-\.\s]{3,15}|Present|Current|Till Date|Now)'
        matches = re.findall(date_range_pattern, cleaned_cv, re.IGNORECASE)
        for start_str, end_str in matches:
            start_dt = parse_date_str(start_str)
            end_dt = parse_date_str(end_str)
            if start_dt and end_dt and start_dt <= end_dt:
                months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                if 0 < months <= 600:
                    fulltime_months += months
                    work_periods.append((start_dt, end_dt, "fulltime"))

    work_periods = sorted(work_periods, key=lambda x: x[0])

    merged_work_periods = []
    for period in work_periods:
        if not merged_work_periods:
            merged_work_periods.append(period)
        else:
            prev_start, prev_end, prev_type = merged_work_periods[-1]
            curr_start, curr_end, curr_type = period
            if curr_start <= prev_end:
                new_end = max(prev_end, curr_end)
                merged_work_periods[-1] = (prev_start, new_end, prev_type)
            else:
                merged_work_periods.append(period)

    exp_gaps = []
    for i in range(len(merged_work_periods) - 1):
        current_end = merged_work_periods[i][1]
        next_start = merged_work_periods[i+1][0]
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

    edu_end_date = None
    edu_pos = cleaned_cv.lower().find('education')
    if edu_pos != -1:
        edu_subtext = cleaned_cv[edu_pos:edu_pos + 600]
        sub_ranges = re.findall(r'([A-Za-z0-9\/\-\.\s]{3,12})\s*(?:–|-|to)+\s*([A-Za-z0-9\/\-\.\s]{3,12})', edu_subtext, re.IGNORECASE)
        if sub_ranges:
            edu_end_date = parse_date_str(sub_ranges[0][1])

    education_to_job_gap_str = "0 Months (Seamless)"
    if edu_end_date and merged_work_periods:
        first_work_start = merged_work_periods[0][0]
        if first_work_start > edu_end_date:
            total_gap_months = (first_work_start.year - edu_end_date.year) * 12 + (first_work_start.month - edu_end_date.month)
            if total_gap_months > 0:
                years = total_gap_months // 12
                months = total_gap_months % 12
                parts = []
                if years > 0:
                    parts.append(f"{years} {'Year' if years == 1 else 'Years'}")
                if months > 0:
                    parts.append(f"{months} {'Month' if months == 1 else 'Months'}")
                education_to_job_gap_str = " ".join(parts) if parts else "0 Months"

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
    doc.add_paragraph(f"Education-to-Job Gap: {det_analysis.get('education_to_job_gap', '0 Months')}")
    
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
        st.rerun()
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
            content = content.split("```")[1].split("```")[0].strip()
            
        parsed_data = json.loads(repair_json(content))
        if "Rule Evaluations" not in parsed_data:
            parsed_data = {"Rule Evaluations": parsed_data}
        return parsed_data
    except Exception:
        fallback_evals = {rule['name']: {"result": "Pass", "confidence": "50%", "reasoning": "Fallback recovered."} for rule in rule_chunk}
        return {"Rule Evaluations": fallback_evals}

def apply_deterministic_guardrails(jd_analysis, det_analysis, combined_rule_evals, overall_score, recommendation):
    min_req_exp = float(jd_analysis.get('minimum_experience_years', 0.0))
    candidate_tot_exp = float(det_analysis.get('total_experience_years', 0.0))
    
    if min_req_exp > 0.0 and candidate_tot_exp < min_req_exp:
        recommendation = "Reject"
        overall_score = min(overall_score, 40.0)
        
        for r_name in combined_rule_evals:
            if "experience" in r_name.lower() or "work" in r_name.lower():
                combined_rule_evals[r_name] = {
                    "result": "Fail",
                    "confidence": "100%",
                    "reasoning": f"Hard check: Candidate experience ({candidate_tot_exp}y) is less than JD requirement ({min_req_exp}y)."
                }
        summary = f"Candidate failed hard experience requirement (Required: {min_req_exp}y, Found: {candidate_tot_exp}y)."
    else:
        summary = "Candidate met required thresholds."
        
    return combined_rule_evals, overall_score, recommendation, summary

def evaluate_hybrid_system_batched(cv_text, jd_text, rules_list, groq_api_key):
    jd_analysis = extract_jd_requirements(jd_text, groq_api_key)
    profile_details = calculate_deterministic_timeline(cv_text)
    evidence_map = extract_universal_evidence(cv_text)

    det_analysis = profile_details

    chunk_size = 4
    rule_chunks = [rules_list[i:i + chunk_size] for i in range(0, len(rules_list), chunk_size)]
    
    combined_rule_evals = {}
    score_points = 0.0
    total_rules = len(rules_list)

    for chunk in rule_chunks:
        res = evaluate_batch_chunk(cv_text, jd_text, chunk, groq_api_key)
        evals = res.get("Rule Evaluations", {})
        for r_name, r_data in evals.items():
            combined_rule_evals[r_name] = r_data
            res_str = str(r_data.get("result", "")).lower()
            if "pass" in res_str and "partial" not in res_str:
                score_points += 1.0
            elif "partial" in res_str:
                score_points += 0.7

    overall_score = round((score_points / max(total_rules, 1)) * 100, 1)
    if overall_score >= 80:
        recommendation = "Strong Hire"
    elif overall_score >= 40:
        recommendation = "Consider / Request Updated CV"
    else:
        recommendation = "Reject"

    combined_rule_evals, overall_score, recommendation, summary = apply_deterministic_guardrails(
        jd_analysis, det_analysis, combined_rule_evals, overall_score, recommendation
    )

    final_output = {
        "Rule Evaluations": combined_rule_evals,
        "Overall Candidate Match Score": overall_score,
        "Derived Recommendation": recommendation,
        "AI Contextual Summary": summary
    }

    return det_analysis, evidence_map, final_output

if st.button("🚀 Run Deterministic Precision Evaluation", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Groq API Key is required.")
    elif not cv_text or not jd_text:
        st.warning("Please provide both JD and Candidate Resume.")
    else:
        with st.spinner("Processing deterministic timeline calculations and rule evaluation..."):
            cv_name = cv_file.name if (cv_file and hasattr(cv_file, 'name')) else "Pasted CV Text"
            jd_name = jd_file.name if (jd_file and hasattr(jd_file, 'name')) else "Pasted JD Text"

            det_analysis, evidence_map, ai_results = evaluate_hybrid_system_batched(cv_text, jd_text, st.session_state.custom_rules, groq_api_key)

            rule_evals = ai_results.get("Rule Evaluations", {})
            overall_score = ai_results.get("Overall Candidate Match Score", 0.0)
            rec = ai_results.get("Derived Recommendation", "Consider")
            summary_text = ai_results.get("AI Contextual Summary", "")

            word_file_io = generate_in_memory_word_report(
                candidate_name, evaluator_id, overall_score, rec, det_analysis, rule_evals, evidence_map
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
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                evaluator_id,
                candidate_name,
                overall_score,
                rec,
                "Deterministic Math Guardrail Evaluation",
                "",
                cv_name,
                jd_name,
                json.dumps(st.session_state.custom_rules),
                json.dumps(det_analysis),
                json.dumps(ai_results),
                json.dumps(evidence_map),
                MODEL_VERSION,
                LOGIC_VERSION
            ))
            conn.commit()
            conn.close()

            st.session_state.evaluation_results = {
                "candidate_name": candidate_name,
                "overall_score": overall_score,
                "recommendation": rec,
                "det_analysis": det_analysis,
                "rule_evals": rule_evals,
                "evidence_map": evidence_map,
                "summary": summary_text,
                "word_file_io": word_file_io
            }
            st.success("Evaluation completed successfully with universal date parsing!")

# ------------------------------------------------------------------------------
# RENDER UI RESULTS FROM SESSION STATE
# ------------------------------------------------------------------------------
if st.session_state.evaluation_results is not None:
    res = st.session_state.evaluation_results
    candidate_name = res["candidate_name"]
    overall_score = res["overall_score"]
    rec = res["recommendation"]
    det_analysis = res["det_analysis"]
    rule_evals = res["rule_evals"]
    evidence_map = res["evidence_map"]
    summary_text = res.get("summary", "")
    word_file_io = res["word_file_io"]

    st.markdown("---")
    st.download_button(
        label="📥 Download Evaluation Report (.docx)",
        data=word_file_io,
        file_name=f"Candidate_Report_{candidate_name.replace(' ', '_')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary"
    )

    # --- AI Recommendation & Status Badge Section ---
    st.markdown("### 📌 AI Recommendation & Verdict")
    if "Strong Hire" in rec:
        st.success(f"**Recommended Status:** {rec}")
    elif "Consider" in rec:
        st.warning(f"**Recommended Status:** {rec}")
    else:
        st.error(f"**Recommended Status:** {rec}")

    # --- Evaluation Summary Note & Recruiter Suggestions UI Section ---
    st.markdown("### 📋 Evaluation Summary Note, Conclusion & Recruiter Suggestions")
    
    if overall_score >= 80:
        conclusion_html = "✅ High Potential / Ready for Interview"
        next_step_msg = "Strong alignment with core JD requirements. Recommend scheduling an initial technical interview."
    elif overall_score >= 40:
        conclusion_html = "⚠️ Moderate Match / Needs Clarification"
        next_step_msg = "Candidate shows partial alignment. Probe into missing skills or request an updated resume."
    else:
        conclusion_html = "❌ Low Alignment / Not Recommended"
        next_step_msg = "Significant gaps found against core JD requirements. Recommend sending a polite rejection notice."

    tot_exp_formatted = format_time_gap(det_analysis.get('total_experience_years', 0.0))
    edu_gap_formatted = det_analysis.get('education_to_job_gap', '0 Months')

    st.info(
        f"**Conclusion Status:** {conclusion_html}\n\n"
        f"**Overall Score:** {overall_score}/100 | **Total Experience:** {tot_exp_formatted} | **Education-to-Job Gap:** {edu_gap_formatted}"
    )

    st.markdown("#### 💡 Actionable Suggestions for Recruiter")
    st.markdown(f"1. **Next Step:** {next_step_msg}")
    
    failed_rules = [r_name for r_name, r_data in rule_evals.items() if str(r_data.get("result", "")).lower() == "fail"]
    weak_areas = ", ".join(failed_rules) if failed_rules else "Technical & Domain Competency, Overall Profile Fit"
    st.markdown(f"2. **Missing/Weak Areas to Probe:** {weak_areas}")

    ft_exp = format_time_gap(det_analysis.get('fulltime_experience_years', 0.0))
    in_exp = format_time_gap(det_analysis.get('internship_experience_years', 0.0))
    tot_exp = format_time_gap(det_analysis.get('total_experience_years', 0.0))

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Match Score", f"{overall_score} / 100")
    m2.metric("Full-Time Exp", ft_exp)
    m3.metric("Intern Exp", in_exp)
    m4.metric("Total Experience", tot_exp)

    # --- Career & Education Gap Breakdown Section ---
    st.markdown("### 🔍 Career & Education Gap Breakdown (Math Calculated)")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 💼 Experience Gaps")
        exp_gaps_list = det_analysis.get("experience_gaps", ["No major experience gaps found"])
        if isinstance(exp_gaps_list, list) and len(exp_gaps_list) > 0:
            for gap in exp_gaps_list:
                st.markdown(f"- {gap}")
        else:
            st.markdown("- No major experience gaps found")
            
    with col_g2:
        st.markdown("#### 🎓 Education & Transition Gaps")
        edu_to_job_val = det_analysis.get("education_to_job_gap", "0 Months")
        st.metric(label="Education-to-Job Transition Gap", value=edu_to_job_val)

    st.markdown("### 📊 Universal Evaluation Matrix & Confidence")
    grid = []
    for r_name, r_data in rule_evals.items():
        grid.append({
            "Rule": r_name,
            "Result": r_data.get("result"),
            "Confidence": r_data.get("confidence"),
            "Reasoning": r_data.get("reasoning")
        })
    st.table(pd.DataFrame(grid))

# ------------------------------------------------------------------------------
# AUDIT TRAIL LOGS
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("📋 Complete Audit Trail & History")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, timestamp, recruiter_id, candidate_name, overall_score, recommendation, cv_source_name, jd_source_name, model_version, logic_version FROM evaluations ORDER BY id DESC")
rows = cursor.fetchall()
conn.close()

if rows:
    for row in rows:
        eval_id, timestamp, rec_id, cand_name, score, rec, cv_src, jd_src, model_v, logic_v = row
        with st.expander(f"Record #{eval_id} | {cand_name} - Score: {score}/100 ({timestamp})"):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Recruiter:** {rec_id}")
            c2.write(f"**Recommendation:** {rec}")
            c3.write(f"**Model:** {model_v}")
            st.text(f"CV Source: {cv_src}")
            st.text(f"JD Source: {jd_src}")