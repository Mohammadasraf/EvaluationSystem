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
LOGIC_VERSION = "v10.45-Universal-MultiFormat-Gap-Fix"

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
    
    date_str = str(date_str).replace("'", "").replace("'", "").replace("`", "").strip().lower()
    
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
    cleaned_cv = cv_text.replace("'", "").replace("'", "")
    
    # Universal pattern to catch multiple date range formats across any CV template
    date_range_pattern = r'([A-Za-z0-9\/\-\.\s]{3,12})\s*[\–\-\to]+\s*([A-Za-z0-9\/\-\.\s]{3,12}|Present|Current|Till Date|Now)'
    matches = re.findall(date_range_pattern, cleaned_cv, re.IGNORECASE)
    
    internship_months = 0
    fulltime_months = 0
    
    work_periods = []
    for start_str, end_str in matches:
        pos = cleaned_cv.find(start_str)
        snippet_context = cleaned_cv[max(0, pos - 120): pos + 50].lower() if pos != -1 else ""
        
        edu_keywords = ['b.tech', 'b.e', 'm.tech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba', 'graduation', 'degree', 'cgpa', 'college', 'university', 'school', 'education', 'hsc', 'ssc']
        if any(kw in snippet_context for kw in edu_keywords):
            continue

        start_dt = parse_date_str(start_str)
        end_dt = parse_date_str(end_str)
        
        if start_dt and end_dt and start_dt <= end_dt:
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
            if months < 0:
                continue
            
            if "intern" in snippet_context or "trainee" in snippet_context or "internship" in snippet_context:
                internship_months += months
                work_periods.append((start_dt, end_dt, "internship"))
            else:
                fulltime_months += months
                work_periods.append((start_dt, end_dt, "fulltime"))

    # Sort work periods chronologically
    work_periods = sorted(work_periods, key=lambda x: x[0])

    # Remove overlapping or duplicate intervals to prevent double counting
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

    # --- UNIVERSAL EDUCATION END DATE PARSING ---
    edu_end_date = None
    edu_pos = cleaned_cv.lower().find('education')
    if edu_pos != -1:
        edu_subtext = cleaned_cv[edu_pos:edu_pos + 600]
        sub_ranges = re.findall(r'([A-Za-z0-9\/\-\.\s]{3,12})\s*[\–\-\to]+\s*([A-Za-z0-9\/\-\.\s]{3,12})', edu_subtext, re.IGNORECASE)
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
            else:
                education_to_job_gap_str = "0 Months (Seamless)"

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
            content = content.split("