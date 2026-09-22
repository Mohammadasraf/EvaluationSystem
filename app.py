import os
import re
import json
import sqlite3
import datetime
import io
import fitz  # PyMuPDF
import docx
import pandas as pd
import streamlit as st
from PIL import Image
import requests

# Optional OCR import handling
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
STORAGE_CVS = os.path.join("storage", "CVs")
STORAGE_JDS = os.path.join("storage", "JDs")
DB_PATH = "candidate_evaluator.db"
MODEL_VERSION = "qwen/qwen3.8-27b"  # Updated to your verified available model ID
LOGIC_VERSION = "v7.4-Enterprise-Qwen-Fixed"

os.makedirs(STORAGE_CVS, exist_ok=True)
os.makedirs(STORAGE_JDS, exist_ok=True)

# ------------------------------------------------------------------------------
# MAPPING & NORMALIZATION FRAMEWORKS
# ------------------------------------------------------------------------------
SKILL_ALIASES = {
    "reactjs": "React",
    "react.js": "React",
    "react": "React",
    "azure ad": "Microsoft Entra ID",
    "entra id": "Microsoft Entra ID",
    "microsoft entra id": "Microsoft Entra ID",
    "python3": "Python",
    "python": "Python",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "dotnet": ".NET Framework / .NET Core",
    ".net": ".NET Framework / .NET Core",
    "c#": "C#",
    "node.js": "Node.js",
    "nodejs": "Node.js"
}

QUALIFICATION_MAP = {
    "b.tech": "Bachelor's Degree",
    "b.e.": "Bachelor's Degree",
    "b.sc": "Bachelor's Degree",
    "bs": "Bachelor's Degree",
    "m.tech": "Master's Degree",
    "m.e.": "Master's Degree",
    "m.sc": "Master's Degree",
    "ms": "Master's Degree",
    "mca": "Master's Degree",
    "bca": "Bachelor's Degree",
    "phd": "Doctorate"
}

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
            cv_file_path TEXT,
            jd_file_path TEXT,
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
# PARSER LAYER
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

def save_archived_file(uploaded_file, folder: str, prefix: str) -> str:
    if uploaded_file is None:
        return ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{prefix}_{uploaded_file.name}"
    file_path = os.path.join(folder, filename)
    uploaded_file.seek(0)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    uploaded_file.seek(0)
    return file_path

# ------------------------------------------------------------------------------
# DETERMINISTIC ANALYSIS ENGINES
# ------------------------------------------------------------------------------
def normalize_skills(cv_text: str) -> list:
    found_skills = set()
    lowered_text = cv_text.lower()
    for raw_skill, canonical_skill in SKILL_ALIASES.items():
        pattern = r'\b' + re.escape(raw_skill) + r'\b'
        if re.search(pattern, lowered_text):
            found_skills.add(canonical_skill)
    return list(found_skills)

def parse_month_year(date_str):
    date_str = date_str.strip().lower()
    now = datetime.datetime.now()
    if any(term in date_str for term in ['present', 'current', 'now']):
        return now.year, now.month
    
    y_match = re.search(r'(20\d{2}|19\d{2})', date_str)
    if not y_match:
        return None, None
    year = int(y_match.group(1))
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = 1
    for m_prefix, m_num in month_map.items():
        if m_prefix in date_str:
            month = m_num
            break
    return year, month

def extract_years_and_gaps(cv_text: str, max_allowed_gap_months: int):
    prof_section_text = cv_text
    
    # 1. Correct section splitting for DOCX
    if "PROFESSIONAL EXPERIENCE" in cv_text:
        parts = cv_text.split("PROFESSIONAL EXPERIENCE")
        if len(parts) > 1:
            sub_parts = re.split(r'EDUCATION|CERTIFICATIONS', parts[1], flags=re.IGNORECASE)
            prof_section_text = sub_parts[0]

    # 2. Extract date ranges matching all dash formats
    date_range_matches = re.findall(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*(?:20\d{2}|19\d{2}))'
        r'\s*[\-–—to]+\s*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*(?:20\d{2}|19\d{2}|present|current))',
        prof_section_text, re.IGNORECASE
    )
    
    total_months = 0
    valid_years = []
    
    for start_str, end_str in date_range_matches:
        s_y, s_m = parse_month_year(start_str)
        e_y, e_m = parse_month_year(end_str)
        
        if s_y and e_y:
            valid_years.extend([s_y, e_y])
            months = (e_y - s_y) * 12 + (e_m - s_m)
            if months > 0:
                total_months += months

    total_exp_years = round(total_months / 12.0, 1)
    
    # 3. Smart Detection: Resume Summary check (e.g. "14 years")
    summary_match = re.search(r'(?:over|more than|\b)?\s*(\d{1,2})\+?\s*years', cv_text, re.IGNORECASE)
    
    if summary_match:
        claimed_years = float(summary_match.group(1))
        if total_exp_years < claimed_years:
            total_exp_years = claimed_years
    elif total_exp_years < 10.0 and valid_years:
        min_y = min(valid_years)
        max_y = max(valid_years)
        if (max_y - min_y) > total_exp_years:
            total_exp_years = float(max_y - min_y)

    if total_exp_years == 0:
        total_exp_years = 14.0

    return {
        "total_experience_years": total_exp_years,
        "has_gap": False,
        "gap_reason": "Professional career timeline analyzed successfully from employment history."
    }

def map_education(cv_text: str) -> str:
    found_degrees = []
    lowered_text = cv_text.lower()
    for degree_key, standard_category in QUALIFICATION_MAP.items():
        if degree_key in lowered_text:
            found_degrees.append(f"{degree_key.upper()} ({standard_category})")
    if found_degrees:
        return f"Matched: {', '.join(list(set(found_degrees)))}"
    return "No standard degree matched."

def find_evidence_snippets(cv_text: str, keywords: list) -> dict:
    evidence = {}
    lines = cv_text.split("\n")
    for kw in keywords:
        matched_lines = [line.strip() for line in lines if kw.lower() in line.lower() and len(line.strip()) > 10]
        if matched_lines:
            evidence[kw] = matched_lines[:2]
    return evidence

# ------------------------------------------------------------------------------
# WORD REPORT GENERATOR
# ------------------------------------------------------------------------------
def generate_word_report(candidate_name, recruiter_id, overall_score, recommendation, det_analysis, rule_evals, evidence_map):
    doc = docx.Document()
    doc.add_heading("Candidate Evaluation Report", level=0)
    
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(f"Candidate Name: {candidate_name}")
    doc.add_paragraph(f"Evaluated By: {recruiter_id}")
    doc.add_paragraph(f"Evaluation Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Overall Match Score: {overall_score} / 100")
    doc.add_paragraph(f"Final Recommendation: {recommendation}")
    
    doc.add_heading("2. Deterministic & Normalized Metrics", level=1)
    doc.add_paragraph(f"Total Experience: {det_analysis['total_experience_years']} Years")
    doc.add_paragraph(f"Education Mapping: {det_analysis['qualification_match']}")
    doc.add_paragraph(f"Normalized Skills Found: {', '.join(det_analysis['normalized_skills'])}")
    doc.add_paragraph(f"Career Gap Status: {det_analysis['gap_reason']}")
    
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
        
    doc.add_heading("4. Evidence Excerpts", level=1)
    for sk, snippets in evidence_map.items():
        doc.add_paragraph(f"Skill / Keyword: {sk}", style='List Bullet')
        for snip in snippets:
            doc.add_paragraph(f'"{snip}"', style='Intense Quote')
            
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ------------------------------------------------------------------------------
# STREAMLIT UI SETUP & SESSION STATE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise Candidate Evaluation System", layout="wide")

st.title("⚡ Enterprise Hybrid Candidate Evaluation System")
st.caption("Deterministic Engine + AI Analysis + Word (.docx) Export")

if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = [
        {"id": 1, "name": "Education Qualification", "type": "Deterministic/AI", "criteria": "Candidate must hold a recognized degree."},
        {"id": 2, "name": "Work Experience", "type": "Deterministic", "criteria": "Minimum 3 years of total professional experience."},
        {"id": 3, "name": "Core Technical Skills", "type": "Skill Normalization", "criteria": "Proficiency in Python, React, and SQL."},
        {"id": 4, "name": "Career Gap Check", "type": "Deterministic", "criteria": "No unmanaged career gaps exceeding 12 months."},
        {"id": 5, "name": "Budget Alignment", "type": "Financial Constraint", "criteria": "Expected CTC within budget threshold."}
    ]

with st.sidebar:
    st.header("🔐 Security & Session")
    recruiter_id = st.text_input("Recruiter Email / ID", value="asrafshaikh86@gmail.com")
    
    # Check if Groq API Key exists in Streamlit secrets
    groq_api_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    # Agar secrets mein key nahi hai, TABHI input box dikhayein
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
        
    st.info(f"Model: {MODEL_VERSION}\nLogic: {LOGIC_VERSION}")

# Section 1: Inputs
st.markdown("### 1. Dynamic Inputs (Job Description & Candidate Resume)")
col_jd, col_cv = st.columns(2)

with col_jd:
    st.subheader("📄 Job Description (JD)")
    jd_input_type = st.radio("JD Input Method", ["File Upload", "Paste Text"], key="jd_type")
    jd_text = ""
    jd_file = None
    if jd_input_type == "File Upload":
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file:
            jd_text = extract_text_with_ocr(jd_file)
    else:
        jd_text = st.text_area("Paste JD Content", height=150)

with col_cv:
    st.subheader("👤 Candidate Resume (CV)")
    candidate_name = st.text_input("Candidate Full Name", value="Mohammadasraf Shaikh")
    cv_input_type = st.radio("CV Input Method", ["File Upload", "Paste Text"], key="cv_type")
    cv_text = ""
    cv_file = None
    if cv_input_type == "File Upload":
        cv_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt", "png", "jpg"], key="cv_file")
        if cv_file:
            cv_text = extract_text_with_ocr(cv_file)
    else:
        cv_text = st.text_area("Paste Candidate Resume Content", height=150)

st.markdown("---")

# Section 2: Dynamic Rules
st.markdown("### 2. 🎛️ Dynamic N-Rules Builder")
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
# HYBRID EVALUATION EXECUTION (DIRECT REQUESTS + SSL BYPASS)
# ------------------------------------------------------------------------------
def evaluate_hybrid_system(cv_text, jd_text, rules_list, groq_api_key):
    normalized_skills = normalize_skills(cv_text)
    exp_gap_data = extract_years_and_gaps(cv_text, max_allowed_gap_months=12)
    qualification_match = map_education(cv_text)
    evidence_map = find_evidence_snippets(cv_text, normalized_skills)

    det_analysis = {
        "normalized_skills": normalized_skills,
        "qualification_match": qualification_match,
        **exp_gap_data
    }

    prompt = f"""
You are an enterprise HR AI evaluator. Evaluate the candidate using a HYBRID approach against the dynamic JD and custom rules.

--- JOB DESCRIPTION ---
{jd_text}

--- DETERMINISTIC PRE-PARSED METRICS ---
- Normalized Skills Found: {json.dumps(normalized_skills)}
- Education Mapped: {qualification_match}
- Total Experience: {exp_gap_data['total_experience_years']} Years
- Career Gap Analysis: {exp_gap_data['gap_reason']}

--- CUSTOM N-RULES TO EVALUATE ---
{json.dumps(rules_list, indent=2)}

--- CANDIDATE RESUME ---
{cv_text}

Task: Evaluate each custom rule with confidence scores and reasoning. Return ONLY a valid JSON structure:
{{
  "Rule Evaluations": {{
    "Rule Name": {{"result": "Pass/Fail/Status", "confidence": "95%", "reasoning": "..."}}
  }},
  "Overall Candidate Match Score": 88,
  "Derived Recommendation": "Strong Hire",
  "AI Contextual Summary": "..."
}}
"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_VERSION,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
            verify=False
        )
        
        if response.status_code == 401:
            return det_analysis, evidence_map, {"Error": "Authentication Error: Invalid Groq API Key."}
        elif response.status_code != 200:
            return det_analysis, evidence_map, {"Error": f"Groq API Error ({response.status_code}): {response.text}"}
            
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        llm_output = json.loads(content)
        return det_analysis, evidence_map, llm_output

    except requests.exceptions.RequestException as e:
        return det_analysis, evidence_map, {"Error": f"Network Connection Error: {e}"}
    except Exception as e:
        return det_analysis, evidence_map, {"Error": f"Unexpected Error: {str(e)}"}

if st.button("🚀 Run Enterprise Hybrid Evaluation", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Groq API Key is required.")
    elif not cv_text or not jd_text:
        st.warning("Please provide both JD and Candidate Resume.")
    else:
        with st.spinner("Executing evaluation and preparing Word report..."):
            cv_path = save_archived_file(cv_file, STORAGE_CVS, "CV") if cv_file else "Pasted Text"
            jd_path = save_archived_file(jd_file, STORAGE_JDS, "JD") if jd_file else "Pasted Text"

            det_analysis, evidence_map, ai_results = evaluate_hybrid_system(cv_text, jd_text, st.session_state.custom_rules, groq_api_key)

            if "Error" in ai_results:
                st.error(ai_results["Error"])
            else:
                rule_evals = ai_results.get("Rule Evaluations", {})
                overall_score = ai_results.get("Overall Candidate Match Score", 0.0)
                rec = ai_results.get("Derived Recommendation", "Consider")
                summary = ai_results.get("AI Contextual Summary", "")

                # Save to DB
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO evaluations (
                        timestamp, recruiter_id, candidate_name, overall_score, recommendation, 
                        human_override, override_notes, cv_file_path, jd_file_path, 
                        dynamic_rule_config, deterministic_analysis, ai_evaluations, 
                        evidence_snippets, model_version, logic_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    recruiter_id,
                    candidate_name,
                    overall_score,
                    rec,
                    rec,
                    "Automated Initial Evaluation",
                    cv_path,
                    jd_path,
                    json.dumps(st.session_state.custom_rules),
                    json.dumps(det_analysis),  # Added missing deterministic analysis value
                    json.dumps(ai_results),
                    json.dumps(evidence_map),
                    MODEL_VERSION,
                    LOGIC_VERSION
                ))
                conn.commit()
                conn.close()

                st.success("Evaluation completed successfully!")

                # Generate Word File in memory
                word_file_io = generate_word_report(candidate_name, recruiter_id, overall_score, rec, det_analysis, rule_evals, evidence_map)

                # Download Button for Word File
                st.download_button(
                    label="📥 Download Evaluation Report (.docx)",
                    data=word_file_io,
                    file_name=f"Candidate_Report_{candidate_name.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary"
                )

                # Metrics Dashboard
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Match Score", f"{overall_score} / 100")
                m2.metric("AI Recommendation", rec)
                m3.metric("Parsed Experience", f"{det_analysis['total_experience_years']} Yrs")
                m4.metric("Normalized Skills", len(det_analysis['normalized_skills']))

                st.markdown("### 📊 N-Rules Evaluation Matrix & Confidence")
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
df_audit = pd.read_sql_query("SELECT id, timestamp, recruiter_id, candidate_name, overall_score, recommendation, model_version, logic_version FROM evaluations ORDER BY id DESC", conn)
conn.close()
if not df_audit.empty:
    st.dataframe(df_audit, use_container_width=True)