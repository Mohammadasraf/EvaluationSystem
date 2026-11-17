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
LOGIC_VERSION = "v18.2-Universal-Semantic-Extraction-Locked-Button"

# ------------------------------------------------------------------------------
# FAST LANGUAGE PRE-CHECK (TO SHOW TRANSLATION SPINNER ONLY WHEN NEEDED)
# ------------------------------------------------------------------------------
def detect_language_fast(text: str, groq_api_key: str) -> str:
    """Quickly detects if text is English or another language before running full translation."""
    if not text or not text.strip():
        return "English"
    
    sample = text[:600].lower()
    english_markers = ["experience", "education", "skills", "summary", "work", "project", "responsibilities", "requirements", "the", "and", "with"]
    matches = sum(1 for word in english_markers if re.search(rf'\b{word}\b', sample))
    non_ascii_ratio = sum(1 for c in sample if ord(c) > 127) / max(len(sample), 1)
    
    if matches >= 3 and non_ascii_ratio < 0.02:
        return "English"
    
    if not groq_api_key:
        return "English"

    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "Identify the primary language of the text. Reply with ONLY the language name (e.g., English, Spanish, German, French)."},
                {"role": "user", "content": text[:500]}
            ],
            temperature=0.0,
            max_tokens=15
        )
        return completion.choices[0].message.content.strip().replace(".", "")
    except Exception:
        return "English"

# ------------------------------------------------------------------------------
# UNIVERSAL MULTILINGUAL TRANSLATION & STRUCTURED TIMELINE EXTRACTION
# ------------------------------------------------------------------------------
def translate_and_extract_cv_structure(text: str, groq_api_key: str) -> tuple[str, str, dict]:
    empty_timeline = {
        "fulltime_roles": [],
        "internship_roles": [],
        "education_entries": [],
        "stated_total_experience_years": 0.0
    }

    if not text or not text.strip():
        return "", "English", empty_timeline

    if not groq_api_key:
        return text, "Unknown (No API Key)", empty_timeline

    system_prompt = (
        "You are an expert multilingual HR data parser and translator. "
        "Your job is to: "
        "1) Detect the document's primary language and translate the entire text accurately into English. "
        "2) Extract structured timeline intervals into strict JSON. "
        "CRITICAL EXTRACTION & NORMALIZATION RULES: "
        "- Translate all foreign month abbreviations (e.g., 'Ago' for August, 'Fév' for February) into standard English abbreviations ('Aug', 'Feb') or numeric format before parsing. "
        "- Normalize all start and end dates strictly to 'MM/YYYY' or 'YYYY' format, and use 'Present' for ongoing roles/studies. "
        "- 'fulltime_roles': Include ONLY paid, formal full-time or contract employment at a company/organization. "
        "- 'internship_roles': Include ONLY formal internships, apprenticeships, or industrial traineeships. "
        "- STRICT EXCLUSIONS: NEVER include university degrees, master's/bachelor's programs, academic projects, personal projects, hackathons, awards, or certifications in 'fulltime_roles' or 'internship_roles'. Put degrees/studies ONLY in 'education_entries'. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Analyze, translate, and extract structured timelines from this CV:

--- CV TEXT ---
{text[:6000]}

Return ONLY valid JSON matching this exact schema:
{{
  "detected_language": "Language Name (e.g., Spanish, English, German)",
  "translated_text": "Complete English translation of the CV...",
  "fulltime_roles": [
    {{"title": "Job Title", "company": "Company Name", "start": "MM/YYYY or YYYY", "end": "MM/YYYY or YYYY or Present"}}
  ],
  "internship_roles": [
    {{"title": "Intern Title", "company": "Company Name", "start": "MM/YYYY or YYYY", "end": "MM/YYYY or YYYY or Present"}}
  ],
  "education_entries": [
    {{"degree": "Degree Name", "institution": "University Name", "start": "MM/YYYY or YYYY", "end": "MM/YYYY or YYYY or Present"}}
  ],
  "stated_total_experience_years": 0.0
}}
"""
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(repair_json(content))
        detected_lang = parsed.get("detected_language", "English")
        translated_text = parsed.get("translated_text", text)

        if len(text) > 6000 and "english" not in detected_lang.lower():
            full_prompt = f"Translate the following {detected_lang} document to English. Output ONLY the translated text:\n\n{text}"
            full_completion = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                max_tokens=4000
            )
            translated_text = full_completion.choices[0].message.content.strip()

        timeline_data = {
            "fulltime_roles": parsed.get("fulltime_roles", []) or [],
            "internship_roles": parsed.get("internship_roles", []) or [],
            "education_entries": parsed.get("education_entries", []) or [],
            "stated_total_experience_years": float(parsed.get("stated_total_experience_years", 0.0) or 0.0)
        }
        return translated_text, detected_lang, timeline_data
    except Exception:
        return text, "English (Fallback)", empty_timeline


def detect_and_translate_jd(text: str, groq_api_key: str) -> tuple[str, str]:
    if not text or not text.strip():
        return "", "English"
    if not groq_api_key:
        return text, "Unknown (No API Key)"

    system_prompt = (
        "Detect the language of the Job Description and translate it to professional English if it is not in English. "
        "Return ONLY valid JSON: {\"detected_language\": \"English\", \"translated_text\": \"...\"}"
    )
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:4000]}
            ],
            temperature=0.0,
            max_tokens=2500
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(repair_json(content))
        return parsed.get("translated_text", text), parsed.get("detected_language", "English")
    except Exception:
        return text, "English (Fallback)"

# ------------------------------------------------------------------------------
# GLOBAL PII ANONYMIZATION & MASKING ENGINE (ENGLISH ONLY)
# ------------------------------------------------------------------------------
def mask_pii_in_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'\[REDACTED_PHONE\](?:\s*[\/\-,]\s*\[REDACTED_PHONE\])+', '[REDACTED_PHONE]', text)
    text = re.sub(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '[REDACTED_URL]', text)
    text = re.sub(r'\b(?:linkedin\.com|github\.com|twitter\.com|facebook\.com|gitlab\.com)/[a-zA-Z0-9_\-/]+', '[REDACTED_URL]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_GOVT_ID]', text)
    text = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b', '[REDACTED_GOVT_ID]', text)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[REDACTED_GOVT_ID]', text)
    text = re.sub(r'\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}\s?\d{6}\s?[A-D]{1}\b', '[REDACTED_GOVT_ID]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{8}[A-Z]{1}\b', '[REDACTED_GOVT_ID]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z]{1,2}\d[A-Z0-9]?\s?\d[A-Z]{2}\b', '[REDACTED_ADDRESS]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[REDACTED_ADDRESS]', text)
    text = re.sub(r'\b\d{6}\b', '[REDACTED_ADDRESS]', text)

    address_keys = r'Address|Location|Residing at|Residential Address|Permanent Address|Current Address'
    text = re.sub(rf'\b(?:{address_keys})\s*:\s*[^\n]+', '[REDACTED_ADDRESS]', text, flags=re.IGNORECASE)

    lines = text.split('\n')
    masked_lines = []
    global_address_keywords = [
        'street', 'st.', 'avenue', 'ave', 'boulevard', 'blvd', 'road', 'rd.', 'lane', 'ln.', 
        'drive', 'dr.', 'way', 'court', 'ct.', 'suite', 'ste', 'apartment', 'apt', 'building',
        'bldg', 'flat', 'floor', 'house no', 'plot no', 'block no', 'postal', 'zip code'
    ]
    for line in lines:
        line_lower = line.lower()
        if any(re.search(rf'\b{re.escape(kw)}\b', line_lower) for kw in global_address_keywords):
            masked_lines.append('[REDACTED_ADDRESS]')
        else:
            masked_lines.append(line)
    text = '\n'.join(masked_lines)
    text = re.sub(r'(\[REDACTED_ADDRESS\]\s*[\/\-,]\s*)+', '[REDACTED_ADDRESS] ', text)
    text = re.sub(r'\b(?:Candidate Name|Full Name|Name)\s*:\s*[A-Za-z\s.]+', '[REDACTED_NAME]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:DOB|Date of Birth|Birth Date)\s*:\s*[^\n]+', '[REDACTED_DEMOGRAPHIC]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Age)\s*:\s*\d{1,2}\s*(?:years|yrs)?', '[REDACTED_DEMOGRAPHIC]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Gender|Sex)\s*:\s*(?:Male|Female|Other|M|F)', '[REDACTED_DEMOGRAPHIC]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Marital Status)\s*:\s*(?:Single|Married|Unmarried|Divorced)', '[REDACTED_DEMOGRAPHIC]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Passport No|Passport Number)\s*:\s*[A-Z0-9]+', '[REDACTED_GOVT_ID]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Nationality)\s*:\s*[A-Za-z\s]+', '[REDACTED_DEMOGRAPHIC]', text, flags=re.IGNORECASE)
    return text

# ------------------------------------------------------------------------------
# DATABASE INIT
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
# HELPER & DETERMINISTIC MATH ENGINE
# ------------------------------------------------------------------------------
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
            return "0.0 Yrs"
    except (ValueError, TypeError):
        return str(value)

def parse_date_str(date_str: str, is_end: bool = False):
    if not date_str:
        return None
    date_str = str(date_str).replace("'", "").replace("`", "").strip().lower()
    if any(kw in date_str for kw in ["present", "current", "till date", "continuing", "now", "today"]):
        return datetime.datetime.now()
    
    match = re.search(r'([a-z]+)\.?\s*[\'`]?(\d{2,4})', date_str)
    if match:
        month_str, year_str = match.groups()
        if len(year_str) == 2:
            year_int = int(year_str)
            year_str = f"19{year_str}" if year_int >= 70 else f"20{year_str}"
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

    match_num = re.search(r'(\d{1,2})[/\-\.](\d{2,4})', date_str)
    if match_num:
        m, y = match_num.groups()
        if len(y) == 2:
            y = f"19{y}" if int(y) >= 70 else f"20{y}"
        try:
            m_int = int(m)
            if 1 <= m_int <= 12:
                return datetime.datetime(int(y), m_int, 1)
        except ValueError:
            pass

    match_yr = re.search(r'\b(20\d{2}|19\d{2})\b', date_str)
    if match_yr:
        try:
            return datetime.datetime(int(match_yr.group(1)), 12 if is_end else 1, 1)
        except ValueError:
            pass
    return None

def calculate_deterministic_timeline(timeline_data: dict) -> dict:
    def build_intervals(role_list):
        intervals = []
        for item in role_list:
            start_dt = parse_date_str(item.get("start", ""), is_end=False)
            end_dt = parse_date_str(item.get("end", ""), is_end=True)
            if start_dt and end_dt and start_dt <= end_dt:
                months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                if 0 < months <= 600:
                    intervals.append((start_dt, end_dt))
        return intervals

    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    ft_periods = build_intervals(timeline_data.get("fulltime_roles", []))
    in_periods = build_intervals(timeline_data.get("internship_roles", []))
    merged_ft = merge_intervals(ft_periods)
    merged_in = merge_intervals(in_periods)

    ft_months = sum((e.year - s.year) * 12 + (e.month - s.month) + 1 for s, e in merged_ft)
    in_months = sum((e.year - s.year) * 12 + (e.month - s.month) + 1 for s, e in merged_in)

    total_years = round(ft_months / 12.0, 1)
    intern_years = round(in_months / 12.0, 1)

    if total_years == 0.0 and not merged_ft:
        stated_exp = float(timeline_data.get("stated_total_experience_years", 0.0) or 0.0)
        if stated_exp > 0.0:
            total_years = round(stated_exp, 1)

    exp_gaps = []
    for i in range(len(merged_ft) - 1):
        curr_end = merged_ft[i][1]
        next_start = merged_ft[i + 1][0]
        if next_start > curr_end:
            gap_m = (next_start.year - curr_end.year) * 12 + (next_start.month - curr_end.month) - 1
            if gap_m >= 2:
                exp_gaps.append(f"Gap of {gap_m} months between {curr_end.strftime('%b %Y')} and {next_start.strftime('%b %Y')}")

    if not exp_gaps:
        exp_gaps = ["No major experience gaps found"]

    edu_end_dates = [parse_date_str(edu.get("end", ""), is_end=True) for edu in timeline_data.get("education_entries", [])]
    edu_end_dates = [d for d in edu_end_dates if d]
    edu_end_date = max(edu_end_dates) if edu_end_dates else None
    education_to_job_gap_str = "0 Months (Seamless)"
    all_work_starts = [s for s, _ in merged_ft] + [s for s, _ in merged_in]

    if edu_end_date and all_work_starts:
        first_work_start = min(all_work_starts)
        if first_work_start > edu_end_date:
            total_gap_months = (first_work_start.year - edu_end_date.year) * 12 + (first_work_start.month - edu_end_date.month) - 1
            if total_gap_months >= 2:
                years = total_gap_months // 12
                months = total_gap_months % 12
                parts = []
                if years > 0:
                    parts.append(f"{years} {'Year' if years == 1 else 'Years'}")
                if months > 0:
                    parts.append(f"{months} {'Month' if months == 1 else 'Months'}")
                education_to_job_gap_str = " ".join(parts) if parts else "0 Months (Seamless)"

    return {
        "internship_experience_years": intern_years,
        "fulltime_experience_years": total_years,
        "total_experience_years": total_years,
        "experience_gaps": exp_gaps,
        "education_gaps": ["No education gaps found"],
        "education_to_job_gap": education_to_job_gap_str
    }

def extract_text_with_ocr(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    text = ""
    if file_ext == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            page_text = page.get_text("text")
            if len(page_text.strip()) < 20 and HAS_OCR:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        return text
    elif file_ext == 'docx':
        doc = docx.Document(uploaded_file)
        paragraphs_text = [para.text for para in doc.paragraphs if para.text.strip()]
        tables_text = [" | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()]) for table in doc.tables for row in table.rows]
        return "\n".join(paragraphs_text + [t for t in tables_text if t])
    elif file_ext == 'txt':
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif file_ext in ['png', 'jpg', 'jpeg'] and HAS_OCR:
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)
    return text

def extract_jd_requirements(jd_text: str, groq_api_key: str) -> dict:
    system_prompt = "Extract key quantitative criteria and core domains from the Job Description into strict JSON. Return ONLY JSON."
    user_prompt = f"""Analyze this Job Description and extract:
1. "minimum_experience_years": float number representing minimum years required. If not specified, return 0.0.
2. "critical_keywords": list of top 3 critical technical skills or tools required.

JD TEXT:
{jd_text}

Return JSON: {{"minimum_experience_years": 0.0, "critical_keywords": ["python", "sql"]}}
"""
    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=250
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(repair_json(content))
        return {
            "minimum_experience_years": float(parsed.get("minimum_experience_years", 0.0) or 0.0),
            "critical_keywords": parsed.get("critical_keywords", [])
        }
    except Exception:
        match = re.search(r'(\d+)\+?\s*years?', jd_text, re.IGNORECASE)
        val = float(match.group(1)) if match else 0.0
        return {"minimum_experience_years": val, "critical_keywords": []}

def extract_universal_evidence(cv_text: str) -> dict:
    lines = [line.strip() for line in cv_text.split("\n") if len(line.strip()) > 25]
    return {"Key Experience Excerpts": lines[:5]}

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
    hdr_cells[0].text, hdr_cells[1].text, hdr_cells[2].text, hdr_cells[3].text = "Rule Name", "Result", "Confidence", "Reasoning"
    for r_name, r_data in rule_evals.items():
        row_cells = table.add_row().cells
        row_cells[0].text = r_name
        row_cells[1].text = str(r_data.get("result", ""))
        row_cells[2].text = str(r_data.get("confidence", ""))
        row_cells[3].text = str(r_data.get("reasoning", ""))
        
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def evaluate_batch_chunk(cv_text, jd_text, rule_chunk, groq_api_key, det_analysis=None):
    system_prompt = (
        "You are an expert HR AI evaluator. "
        "Evaluate rules using three statuses: 'Pass', 'Partial Match', or 'Fail'. "
        "Return ONLY valid JSON format matching the requested structure without markdown code blocks."
    )
    gaps_summary = ""
    if det_analysis and "experience_gaps" in det_analysis:
        gaps_summary = "\n--- CALCULATED EXPERIENCE GAPS (MATH ENGINE) ---\n" + "\n".join(det_analysis["experience_gaps"])

    user_prompt = f"""Evaluate the candidate against the rules subset dynamically based on the provided JD, Resume, and Math Analysis.

--- JOB DESCRIPTION ---
{jd_text}

--- RULES SUBSET ---
{json.dumps(rule_chunk, indent=2)}

--- RESUME (PII MASKED) ---
{cv_text}
{gaps_summary}

IMPORTANT: If there are major career gaps listed above, 'Career Stability' MUST NOT be marked as Pass with 'no major gaps'. It should be 'Partial Match' or 'Fail'.

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
            temperature=0.0,
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

def apply_fully_dynamic_guardrails(jd_analysis, det_analysis, combined_rule_evals, overall_score, recommendation):
    min_req_exp = float(jd_analysis.get('minimum_experience_years', 0.0))
    candidate_tot_exp = float(det_analysis.get('total_experience_years', 0.0))
    exp_gaps = det_analysis.get("experience_gaps", [])
    has_real_gaps = any("Gap of" in gap for gap in exp_gaps)

    for r_name in combined_rule_evals:
        if "career stability" in r_name.lower() or "stability" in r_name.lower():
            if has_real_gaps:
                combined_rule_evals[r_name] = {
                    "result": "Partial Match",
                    "confidence": "85%",
                    "reasoning": f"Flagged: Candidate has strong senior experience but {len(exp_gaps)} employment gaps detected."
                }

    experience_failed = False
    if min_req_exp > 0.0 and candidate_tot_exp < min_req_exp:
        experience_failed = True
        for r_name in combined_rule_evals:
            if "experience" in r_name.lower() or "work" in r_name.lower():
                combined_rule_evals[r_name] = {
                    "result": "Fail",
                    "confidence": "100%",
                    "reasoning": f"Dynamic Guardrail: Exp ({candidate_tot_exp}y) < JD requirement ({min_req_exp}y)."
                }

    failed_rule_count = sum(1 for r_name, r_data in combined_rule_evals.items() if str(r_data.get("result", "")).lower() == "fail")
    total_rules = len(combined_rule_evals)
    
    if experience_failed or (failed_rule_count >= max(1, total_rules // 2)):
        recommendation = "Reject"
        overall_score = min(overall_score, 39.0 if experience_failed else 45.0)
        summary = "Candidate rejected via dynamic guardrails (Exp shortfall or major rule failures)."
    elif has_real_gaps:
        recommendation = "Consider / Request Updated CV"
        overall_score = min(overall_score, 75.0)
        summary = f"Candidate has strong overall experience ({candidate_tot_exp} yrs), but {len(exp_gaps)} career gaps were detected that require recruiter clarification."
    elif overall_score >= 80 and failed_rule_count == 0:
        recommendation = "Strong Hire"
        summary = "Candidate meets all criteria excellently and qualifies as a strong hire."
    elif overall_score >= 40:
        recommendation = "Consider / Request Updated CV"
        summary = "Candidate demonstrates a moderate match and warrants further evaluation or clarification."
    else:
        recommendation = "Reject"
        overall_score = min(overall_score, 40.0)
        summary = "Candidate score fell below viable threshold."

    if recommendation == "Reject":
        for r_name in combined_rule_evals:
            if "overall profile fit" in r_name.lower() or "overall fit" in r_name.lower():
                combined_rule_evals[r_name] = {
                    "result": "Fail",
                    "confidence": "100%",
                    "reasoning": "Overridden to Fail due to final profile rejection/guardrails."
                }

    return combined_rule_evals, overall_score, recommendation, summary

# ------------------------------------------------------------------------------
# STREAMLIT UI SETUP & STEP-BY-STEP WIZARD
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Universal Enterprise Candidate Evaluator", layout="wide")

st.title("⚡ Universal Enterprise Candidate Evaluation System")
st.caption("100% Universal Semantic Extraction + Deterministic Math Engine + Global PII Protection")

# Initialize Session State
if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = [
        {"id": 1, "name": "Technical & Domain Competency", "type": "AI Evaluation", "criteria": "Candidate possesses required technical stack AND mandatory domain/industry experience mentioned in the JD."},
        {"id": 2, "name": "Work Experience", "type": "Deterministic", "criteria": "Meets or exceeds minimum required professional experience."},
        {"id": 3, "name": "Project Relevance", "type": "AI Evaluation", "criteria": "Previous project exposure aligns with job responsibilities."},
        {"id": 4, "name": "Career Stability", "type": "Deterministic", "criteria": "No unexplained erratic career switches or major gaps."},
        {"id": 5, "name": "Overall Profile Fit", "type": "AI Evaluation", "criteria": "Strong overall suitability for the role."}
    ]

if "editing_rule_id" not in st.session_state:
    st.session_state.editing_rule_id = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "form_reset_counter" not in st.session_state:
    st.session_state.form_reset_counter = 0
if "rule_form_reset_counter" not in st.session_state:
    st.session_state.rule_form_reset_counter = 0

# Evaluation locking states
if "eval_completed" not in st.session_state:
    st.session_state.eval_completed = False
if "last_evaluated_jd" not in st.session_state:
    st.session_state.last_evaluated_jd = ""
if "last_evaluated_cv" not in st.session_state:
    st.session_state.last_evaluated_cv = ""

# Sidebar Security & Credentials
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
        groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key_input, type="password")
        st.session_state.groq_api_key_input = groq_api_key
    else:
        st.success("✅ Groq API Key loaded from Secrets")
        
    st.markdown("---")
    st.info(f"**Model:** {MODEL_VERSION}\n\n**Logic:** {LOGIC_VERSION}")

# ------------------------------------------------------------------------------
# STEP-BY-STEP WIZARD NAVIGATION
# ------------------------------------------------------------------------------
counter = st.session_state.form_reset_counter

step1, step2, step3, step4, step5 = st.tabs([
    "Step 1: 📄 Job Description",
    "Step 2: 👤 Candidate CV",
    "Step 3: 🎛 Evaluation Rules",
    "Step 4: 🚀 Run & Results",
    "Step 5: 📋 Audit Trail"
])

# --- STEP 1: JOB DESCRIPTION ---
with step1:
    st.subheader("Step 1: Upload or Paste Job Description (JD)")
    jd_input_type = st.radio("JD Input Method", ["File Upload", "Paste Text"], key=f"jd_type_{counter}", horizontal=True)
    jd_file = None
    jd_text = ""
    if jd_input_type == "File Upload":
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key=f"jd_file_{counter}")
        if jd_file:
            jd_text = extract_text_with_ocr(jd_file)
            st.success(f"✅ Loaded JD: **{jd_file.name}**")
    else:
        jd_text = st.text_area("Paste JD Content", height=200, key=f"jd_textarea_{counter}")

    st.session_state.jd_text_val = jd_text
    st.session_state.jd_name_val = jd_file.name if (jd_file and hasattr(jd_file, 'name')) else "Pasted JD Text"

# --- STEP 2: CANDIDATE CV ---
with step2:
    st.subheader("Step 2: Upload or Paste Candidate Resume (CV)")
    candidate_name = st.text_input("Candidate Full Name (Masked for Storage)", value="Candidate Name", key=f"cand_name_{counter}")
    cv_input_type = st.radio("CV Input Method", ["File Upload", "Paste Text"], key=f"cv_type_{counter}", horizontal=True)
    cv_file = None
    cv_text = ""
    if cv_input_type == "File Upload":
        cv_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT, Image)", type=["pdf", "docx", "txt", "png", "jpg"], key=f"cv_file_{counter}")
        if cv_file:
            cv_text = extract_text_with_ocr(cv_file)
            st.success(f"✅ Loaded Resume: **{cv_file.name}**")
    else:
        cv_text = st.text_area("Paste Candidate Resume Content", height=200, key=f"cv_textarea_{counter}")

    st.session_state.cv_text_val = cv_text
    st.session_state.candidate_name_val = candidate_name

# --- STEP 3: RULES BUILDER ---
with step3:
    st.subheader("Step 3: Configure Evaluation Rules")
    with st.expander("➕ Add New Evaluation Rule", expanded=False):
        rc = st.session_state.rule_form_reset_counter
        new_name = st.text_input("Rule Name", key=f"new_r_name_{rc}")
        new_type = st.selectbox("Rule Type", ["Deterministic", "Skill Check", "Compliance", "Custom"], key=f"new_r_type_{rc}")
        new_criteria = st.text_area("Rule Description / Criteria", key=f"new_r_crit_{rc}")
        
        if st.button("Add Rule"):
            if new_name and new_criteria:
                max_id = max([r['id'] for r in st.session_state.custom_rules]) if st.session_state.custom_rules else 0
                st.session_state.custom_rules.append({"id": max_id + 1, "name": new_name, "type": new_type, "criteria": new_criteria})
                st.session_state.rule_form_reset_counter += 1
                st.success("Rule added successfully!")
                st.rerun()
            else:
                st.warning("Please fill both Rule Name and Criteria.")

    st.markdown("#### Active Evaluation Rules")
    updated_rules = []
    for idx, rule in enumerate(st.session_state.custom_rules):
        c1, c2, c3, c4 = st.columns([1, 5, 1, 1])
        c1.markdown(f"**Rule {idx+1}**")
        c2.markdown(f"**{rule['name']}** ({rule.get('type', 'Custom')}): *{rule['criteria']}*")
        if c3.button("✏️", key=f"edit_{rule['id']}"):
            st.session_state.editing_rule_id = rule['id']
            st.rerun()
        if c4.button("❌", key=f"del_{rule['id']}"):
            if st.session_state.editing_rule_id == rule['id']:
                st.session_state.editing_rule_id = None
            continue
        updated_rules.append(rule)

        if st.session_state.editing_rule_id == rule['id']:
            with st.container():
                st.markdown(f"**✏️ Editing Rule #{idx+1}**")
                edited_name = st.text_input("Edit Rule Name", value=rule['name'], key=f"ename_{rule['id']}")
                edited_type = st.selectbox("Edit Rule Type", ["Deterministic", "Skill Check", "Compliance", "Custom"], key=f"etype_{rule['id']}")
                edited_criteria = st.text_area("Edit Criteria", value=rule['criteria'], key=f"ecrit_{rule['id']}")
                col_s1, col_s2 = st.columns(2)
                if col_s1.button("💾 Save Changes", key=f"save_{rule['id']}"):
                    rule['name'], rule['type'], rule['criteria'] = edited_name, edited_type, edited_criteria
                    st.session_state.editing_rule_id = None
                    st.rerun()
                if col_s2.button("Cancel", key=f"cancel_{rule['id']}"):
                    st.session_state.editing_rule_id = None
                    st.rerun()

    if len(updated_rules) != len(st.session_state.custom_rules):
        st.session_state.custom_rules = updated_rules
        st.rerun()

# --- STEP 4: RUN EVALUATION & VIEW RESULTS ---
with step4:
    st.subheader("Step 4: Run Evaluation & View Results")
    
    current_jd = st.session_state.get("jd_text_val", "").strip()
    current_cv = st.session_state.get("cv_text_val", "").strip()
    
    jd_ready = bool(current_jd)
    cv_ready = bool(current_cv)
    
    # Check if evaluation has already been completed for the current input
    is_same_input = (
        st.session_state.eval_completed
        and current_jd == st.session_state.last_evaluated_jd
        and current_cv == st.session_state.last_evaluated_cv
    )
    
    col_chk1, col_chk2, col_chk3 = st.columns(3)
    col_chk1.metric("Step 1: JD Status", "✅ Ready" if jd_ready else "⚠️ Missing")
    col_chk2.metric("Step 2: CV Status", "✅ Ready" if cv_ready else "⚠️ Missing")
    col_chk3.metric("Step 3: Active Rules", f"{len(st.session_state.custom_rules)} Rules")

    if is_same_input:
        st.info("🔒 Evaluation already completed for this exact JD & CV. Upload or paste a new JD/CV in Step 1 & 2 to re-enable the button.")

    btn_disabled = (not jd_ready) or (not cv_ready) or is_same_input

    if st.button("🚀 Run Fully Dynamic Evaluation Engine", type="primary", use_container_width=True, disabled=btn_disabled):
        if not groq_api_key:
            st.error("❌ Groq API Key is required in the sidebar.")
        elif not cv_ready or not jd_ready:
            st.warning("⚠️ Please complete Step 1 (JD) and Step 2 (Candidate Resume) first.")
        else:
            raw_cv = current_cv
            raw_jd = current_jd
            cand_name_input = st.session_state.candidate_name_val
            jd_name = st.session_state.jd_name_val

            # 1. Fast Language Detection Check
            cv_lang_check = detect_language_fast(raw_cv, groq_api_key)
            jd_lang_check = detect_language_fast(raw_jd, groq_api_key)
            is_non_english = ("english" not in cv_lang_check.lower()) or ("english" not in jd_lang_check.lower())

            # 2. Conditional Translation Spinner
            if is_non_english:
                with st.spinner(f"🌐 Non-English document detected ({cv_lang_check} / {jd_lang_check}). Language translation in progress..."):
                    translated_cv, detected_cv_lang, timeline_data = translate_and_extract_cv_structure(raw_cv, groq_api_key)
                    translated_jd, detected_jd_lang = detect_and_translate_jd(raw_jd, groq_api_key)
            else:
                with st.spinner("⚙️ Extracting structured timelines and running deterministic evaluation..."):
                    translated_cv, detected_cv_lang, timeline_data = translate_and_extract_cv_structure(raw_cv, groq_api_key)
                    translated_jd, detected_jd_lang = raw_jd, "English"

            # 3. Deterministic Math & PII Masking
            with st.spinner("🔒 Masking PII and calculating deterministic career metrics..."):
                det_analysis = calculate_deterministic_timeline(timeline_data)
                det_analysis["detected_cv_language"] = detected_cv_lang
                det_analysis["detected_jd_language"] = detected_jd_lang

                masked_cv_text = mask_pii_in_text(translated_cv)
                jd_analysis = extract_jd_requirements(translated_jd, groq_api_key)
                evidence_map = extract_universal_evidence(masked_cv_text)

                chunk_size = 4
                rule_chunks = [st.session_state.custom_rules[i:i + chunk_size] for i in range(0, len(st.session_state.custom_rules), chunk_size)]
                combined_rule_evals = {}
                score_points = 0.0
                total_rules = len(st.session_state.custom_rules)

                for chunk in rule_chunks:
                    res = evaluate_batch_chunk(masked_cv_text, translated_jd, chunk, groq_api_key, det_analysis=det_analysis)
                    evals = res.get("Rule Evaluations", {})
                    for r_name, r_data in evals.items():
                        combined_rule_evals[r_name] = r_data
                        res_str = str(r_data.get("result", "")).lower()
                        if "pass" in res_str and "partial" not in res_str:
                            score_points += 1.0
                        elif "partial" in res_str:
                            score_points += 0.7

                overall_score = round((score_points / max(total_rules, 1)) * 100, 1)
                combined_rule_evals, overall_score, rec, summary_text = apply_fully_dynamic_guardrails(
                    jd_analysis, det_analysis, combined_rule_evals, overall_score, "Consider"
                )

                anonymized_candidate_name = f"Candidate_{abs(hash(cand_name_input)) % 10000}"
                word_file_io = generate_in_memory_word_report(
                    anonymized_candidate_name, evaluator_id, overall_score, rec, det_analysis, combined_rule_evals, evidence_map
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
                    anonymized_candidate_name,
                    overall_score,
                    rec,
                    "Contextual Dynamic Guardrail Evaluation",
                    "",
                    "Anonymized_CV",
                    jd_name,
                    json.dumps(st.session_state.custom_rules),
                    json.dumps(det_analysis),
                    json.dumps({"Rule Evaluations": combined_rule_evals, "Overall Candidate Match Score": overall_score}),
                    json.dumps(evidence_map),
                    MODEL_VERSION,
                    LOGIC_VERSION
                ))
                conn.commit()
                conn.close()

                st.session_state.evaluation_results = {
                    "candidate_name": anonymized_candidate_name,
                    "overall_score": overall_score,
                    "recommendation": rec,
                    "det_analysis": det_analysis,
                    "rule_evals": combined_rule_evals,
                    "evidence_map": evidence_map,
                    "summary": summary_text,
                    "word_file_io": word_file_io
                }
                
                # Lock button for current inputs
                st.session_state.eval_completed = True
                st.session_state.last_evaluated_jd = raw_jd
                st.session_state.last_evaluated_cv = raw_cv
                
                st.session_state.form_reset_counter += 1
                st.success("✅ Evaluation completed!")
                st.rerun()

    # Render Results
    if st.session_state.evaluation_results is not None:
        res = st.session_state.evaluation_results
        det_analysis = res["det_analysis"]
        overall_score = res["overall_score"]
        rec = res["recommendation"]
        rule_evals = res["rule_evals"]
        summary_text = res.get("summary", "")

        st.markdown("---")
        st.download_button(
            label="📥 Download Evaluation Report (.docx)",
            data=res["word_file_io"],
            file_name=f"Candidate_Report_{res['candidate_name']}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )

        st.markdown("### 📌 AI Recommendation & Verdict")
        if "Strong Hire" in rec:
            st.success(f"**Recommended Status:** {rec}")
        elif "Consider" in rec:
            st.warning(f"**Recommended Status:** {rec}")
        else:
            st.error(f"**Recommended Status:** {rec}")

        st.markdown("### 📋 Evaluation Summary Note, Conclusion & Recruiter Suggestions")
        
        exp_gaps_list = det_analysis.get("experience_gaps", [])
        has_gaps = any("Gap of" in gap for gap in exp_gaps_list)
        cv_lang = det_analysis.get("detected_cv_language", "English")
        jd_lang = det_analysis.get("detected_jd_language", "English")

        if overall_score >= 80 and not has_gaps:
            conclusion_html = "✅ High Potential / Ready for Interview"
            next_step_msg = "Strong alignment with core JD requirements. Recommend scheduling an initial technical interview."
        elif has_gaps or overall_score >= 40:
            conclusion_html = "⚠️ Moderate Match / Needs Clarification (Gaps Detected)"
            gap_str = ", ".join(exp_gaps_list) if has_gaps else "employment timeline"
            next_step_msg = f"Candidate demonstrates strong technical alignment. Do not reject immediately, but probe into the following gaps during initial screening: {gap_str}."
        else:
            conclusion_html = "❌ Low Alignment / Not Recommended"
            next_step_msg = "Significant gaps found against core JD requirements. Recommend sending a polite rejection notice."

        tot_exp_formatted = format_time_gap(det_analysis.get('total_experience_years', 0.0))
        edu_gap_formatted = det_analysis.get('education_to_job_gap', '0 Months')

        st.info(
            f"**Conclusion Status:** {conclusion_html}\n\n"
            f"**Detected CV Language:** {cv_lang} | **Detected JD Language:** {jd_lang}\n\n"
            f"**Overall Score:** {overall_score}/100 | **Total Experience:** {tot_exp_formatted} | **Education-to-Job Gap:** {edu_gap_formatted}\n\n"
            f"**Summary:** {summary_text}"
        )

        st.markdown("#### 💡 Actionable Suggestions for Recruiter")
        st.markdown(f"1. **Next Step:** {next_step_msg}")
        
        failed_rules = [r_name for r_name, r_data in rule_evals.items() if str(r_data.get("result", "")).lower() in ["fail", "partial match"]]
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

        st.markdown("### 🔍 Career & Education Gap Breakdown (Math Calculated)")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### 💼 Experience Gaps")
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

# --- STEP 5: AUDIT TRAIL ---
with step5:
    st.subheader("Step 5: Complete Audit Trail & History")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, recruiter_id, candidate_name, overall_score, recommendation, cv_source_name, jd_source_name, model_version FROM evaluations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        for row in rows:
            eval_id, timestamp, rec_id, cand_name, score, rec, cv_src, jd_src, model_v = row
            with st.expander(f"Record #{eval_id} | {cand_name} - Score: {score}/100 ({timestamp})"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Recruiter:** {rec_id}")
                c2.write(f"**Recommendation:** {rec}")
                c3.write(f"**Model:** {model_v}")
                st.text(f"CV Source: {cv_src} | JD Source: {jd_src}")
    else:
        st.info("No evaluation records found yet.")