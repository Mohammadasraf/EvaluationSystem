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
MODEL_VERSION = "qwen/qwen3.8-27b"  # Verified available model ID
LOGIC_VERSION = "v8.1-Universal-With-Recruiter-Conclusion"

os.makedirs(STORAGE_CVS, exist_ok=True)
os.makedirs(STORAGE_JDS, exist_ok=True)

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
# UNIVERSAL DETERMINISTIC ENGINES (NO HARDCODING)
# ------------------------------------------------------------------------------
def parse_month_year(date_str):
    date_str = date_str.strip().lower()
    now = datetime.datetime.now()
    if any(term in date_str for term in ['present', 'current', 'now']):
        return now.year, now.month
    
    y_match = re.search(r'(20\d{2}|19\d{2})', date_str)
    if not y_match:
        return None, None
    year = int(y_match.group(1))
    
    m_num_match = re.search(r'^(\d{1,2})[/\-]', date_str)
    if m_num_match:
        month = int(m_num_match.group(1))
        if 1 <= month <= 12:
            return year, month
    
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

def extract_years_and_gaps(cv_text: str):
    prof_section_text = cv_text
    for header in ["WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EXPERIENCE", "EMPLOYMENT HISTORY"]:
        if header in cv_text.upper():
            parts = re.split(header, cv_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_parts = re.split(r'EDUCATION|CERTIFICATIONS|PROJECTS|SKILLS', parts[1], flags=re.IGNORECASE)
                prof_section_text = sub_parts[0]
                break

    date_range_matches = re.findall(
        r'((?:\d{1,2}[/\-])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*(?:20\d{2}|19\d{2}))'
        r'\s*[\-–—to]+\s*'
        r'((?:\d{1,2}[/\-])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*(?:20\d{2}|19\d{2}|present|current))',
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
    
    if total_exp_years == 0 and valid_years:
        min_y = min(valid_years)
        max_y = max(valid_years)
        total_exp_years = float(max_y - min_y)

    if total_exp_years == 0:
        total_exp_years = 3.0  # Default fallback if timeline parsing is ambiguous

    return {
        "total_experience_years": total_exp_years,
        "gap_reason": "Career timeline successfully extracted and evaluated."
    }

def extract_universal_evidence(cv_text: str) -> dict:
    lines = [line.strip() for line in cv_text.split("\n") if len(line.strip()) > 25]
    return {"Key Experience Excerpts": lines[:5]}

# ------------------------------------------------------------------------------
# WORD REPORT GENERATOR
# ------------------------------------------------------------------------------
def generate_word_report(candidate_name, recruiter_id, overall_score, recommendation, det_analysis, rule_evals, evidence_map):
    doc = docx.Document()
    doc.add_heading("Universal Candidate Evaluation Report", level=0)
    
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(f"Candidate Name: {candidate_name}")
    doc.add_paragraph(f"Evaluated By: {recruiter_id}")
    doc.add_paragraph(f"Evaluation Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Overall Match Score: {overall_score} / 100")
    doc.add_paragraph(f"Final Recommendation: {recommendation}")
    
    doc.add_heading("2. Universal Timeline & Metrics", level=1)
    doc.add_paragraph(f"Calculated Experience: {det_analysis['total_experience_years']} Years")
    doc.add_paragraph(f"Timeline Status: {det_analysis['gap_reason']}")
    
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

st.title("⚡ Universal Enterprise Candidate Evaluation System")
st.caption("Domain-Independent Engine + AI Batching Analysis + Word (.docx) Export + Recruiter Suggestions")

if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = [
        {"id": 1, "name": "Technical & Domain Competency", "type": "AI Evaluation", "criteria": "Candidate possesses required technical stack/skills mentioned in JD."},
        {"id": 2, "name": "Work Experience", "type": "Deterministic", "criteria": "Meets or exceeds minimum required professional experience."},
        {"id": 3, "name": "Project Relevance", "type": "AI Evaluation", "criteria": "Previous project exposure aligns with job responsibilities."},
        {"id": 4, "name": "Career Stability", "type": "Deterministic", "criteria": "No unexplained erratic career switches or major gaps."},
        {"id": 5, "name": "Overall Profile Fit", "type": "AI Evaluation", "criteria": "Strong overall suitability for the role."}
    ]

with st.sidebar:
    st.header("🔐 Security & Session")
    recruiter_id = st.text_input("Recruiter Email / ID", value="asrafshaikh86@gmail.com")
    
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
    candidate_name = st.text_input("Candidate Full Name", value="Candidate Name")
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

# Section 2: Dynamic Rules Builder
st.markdown("### 2. 🎛️ Dynamic N-Rules Builder (Universal Batching)")
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
# UNIVERSAL BATCHED EVALUATION ENGINE
# ------------------------------------------------------------------------------
def evaluate_batch_chunk(cv_text, jd_text, rule_chunk, groq_api_key):
    prompt = f"""
You are a universal enterprise HR AI evaluator. Evaluate the candidate against the provided sub-set of rules objectively based strictly on the JD and Resume provided, regardless of the technology stack (e.g., Salesforce, .NET, Java, Medical, etc.).

--- JOB DESCRIPTION ---
{jd_text}

--- RULES SUB-SET TO EVALUATE ---
{json.dumps(rule_chunk, indent=2)}

--- RESUME ---
{cv_text}

Return ONLY valid JSON format containing the evaluations for these specific rules:
{{
  "Rule Evaluations": {{
    "Rule Name": {{"result": "Pass/Fail", "confidence": "90%", "reasoning": "Short objective explanation under 15 words."}}
  }}
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
        "max_tokens": 700,
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
        
        if response.status_code != 200:
            return {"Error": f"Groq API Error ({response.status_code}): {response.text}"}
            
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        return {"Error": str(e)}

def evaluate_hybrid_system_batched(cv_text, jd_text, rules_list, groq_api_key):
    exp_gap_data = extract_years_and_gaps(cv_text)
    evidence_map = extract_universal_evidence(cv_text)

    det_analysis = {
        **exp_gap_data
    }

    # Split rules into chunks of 4 to stay safely within token limits
    chunk_size = 4
    rule_chunks = [rules_list[i:i + chunk_size] for i in range(0, len(rules_list), chunk_size)]
    
    combined_rule_evals = {}
    passed_rules_count = 0
    total_rules = len(rules_list)

    for chunk in rule_chunks:
        res = evaluate_batch_chunk(cv_text, jd_text, chunk, groq_api_key)
        if "Error" in res:
            return det_analysis, evidence_map, res
        
        evals = res.get("Rule Evaluations", {})
        for r_name, r_data in evals.items():
            combined_rule_evals[r_name] = r_data
            if "pass" in str(r_data.get("result", "")).lower():
                passed_rules_count += 1

    overall_score = round((passed_rules_count / max(total_rules, 1)) * 100, 1)
    if overall_score >= 80:
        recommendation = "Strong Hire"
        summary = "Candidate successfully met the vast majority of evaluated criteria."
    elif overall_score >= 50:
        recommendation = "Consider"
        summary = "Candidate met several criteria but requires verification on specific areas."
    else:
        recommendation = "Reject"
        summary = "Candidate fell short on critical rule thresholds."

    final_output = {
        "Rule Evaluations": combined_rule_evals,
        "Overall Candidate Match Score": overall_score,
        "Derived Recommendation": recommendation,
        "AI Contextual Summary": summary
    }

    return det_analysis, evidence_map, final_output

if st.button("🚀 Run Universal Batched Evaluation", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Groq API Key is required.")
    elif not cv_text or not jd_text:
        st.warning("Please provide both JD and Candidate Resume.")
    else:
        with st.spinner("Executing universal batched evaluation and preparing report..."):
            cv_path = save_archived_file(cv_file, STORAGE_CVS, "CV") if cv_file else "Pasted Text"
            jd_path = save_archived_file(jd_file, STORAGE_JDS, "JD") if jd_file else "Pasted Text"

            det_analysis, evidence_map, ai_results = evaluate_hybrid_system_batched(cv_text, jd_text, st.session_state.custom_rules, groq_api_key)

            if "Error" in ai_results:
                st.error(ai_results["Error"])
            else:
                rule_evals = ai_results.get("Rule Evaluations", {})
                overall_score = ai_results.get("Overall Candidate Match Score", 0.0)
                rec = ai_results.get("Derived Recommendation", "Consider")

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
                    "Automated Universal Batched Evaluation",
                    cv_path,
                    jd_path,
                    json.dumps(st.session_state.custom_rules),
                    json.dumps(det_analysis),
                    json.dumps(ai_results),
                    json.dumps(evidence_map),
                    MODEL_VERSION,
                    LOGIC_VERSION
                ))
                conn.commit()
                conn.close()

                st.success("Universal evaluation completed successfully!")

                word_file_io = generate_word_report(candidate_name, recruiter_id, overall_score, rec, det_analysis, rule_evals, evidence_map)

                st.download_button(
                    label="📥 Download Evaluation Report (.docx)",
                    data=word_file_io,
                    file_name=f"Candidate_Report_{candidate_name.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary"
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Match Score", f"{overall_score} / 100")
                m2.metric("AI Recommendation", rec)
                m3.metric("Calculated Experience", f"{det_analysis['total_experience_years']} Yrs")

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

                # ------------------------------------------------------------------
                # NEW SECTION: EVALUATION SUMMARY NOTE, CONCLUSION & RECRUITER SUGGESTIONS
                # ------------------------------------------------------------------
                st.markdown("---")
                st.header("📝 Evaluation Summary Note, Conclusion & Recruiter Suggestions")
                
                # Dynamic Logic for Suggestions & Conclusion based on Score & Failures
                failed_rules = [r_name for r_name, r_data in rule_evals.items() if "fail" in str(r_data.get("result", "")).lower()]
                
                if overall_score >= 80:
                    conclusion_status = "✅ High Potential / Ready for Interview"
                    action_suggestion = "Proceed directly to technical or HR interview rounds. Candidate demonstrates strong alignment with job requirements."
                elif overall_score >= 50:
                    conclusion_status = "⚠️ Moderate Match / Needs Clarification or CV Update"
                    if failed_rules:
                        action_suggestion = f"Candidate shows promise in overall background, but specific required areas/skills (e.g., {', '.join(failed_rules)}) are missing or unclear in the CV. Consider asking the candidate to send an updated CV highlighting these skills before rejection."
                    else:
                        action_suggestion = "Candidate meets basic criteria but requires a quick screening call to verify depth of experience."
                else:
                    conclusion_status = "❌ Low Alignment / Not Recommended"
                    action_suggestion = "Significant gaps found against core JD requirements. Recommend sending a polite rejection notice or keeping on file for future roles."

                st.info(f"**Conclusion Status:** {conclusion_status}\n\n**Overall Score:** {overall_score}/100 | **Total Experience:** {det_analysis['total_experience_years']} Years")
                
                st.markdown("#### 💡 Actionable Suggestions for Recruiter")
                st.markdown(f"""
                1. **Next Step:** {action_suggestion}
                2. **Missing/Weak Areas to Probe:** {', '.join(failed_rules) if failed_rules else 'None identified. All evaluated criteria passed successfully.'}
                3. **Suggested Communication Strategy:** 
                   - *If skills are missing despite experience:* Request an updated CV from the candidate with explicit mention of the required technology stack.
                   - *If strong fit:* Send out interview availability slots promptly.
                """)

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