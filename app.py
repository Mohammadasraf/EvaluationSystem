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
LOGIC_VERSION = "v10.38-Deterministic-Gap-Calculation"

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
    date_str = date_str.strip().lower()
    if "present" in date_str or "current" in date_str or "till date" in date_str:
        return datetime.datetime.now()
    
    match = re.search(r'([a-z]+)\s*[\'']?(\d{2,4})', date_str)
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
    date_range_pattern = r'([A-Za-z]+\s*[\'']?\d{2,4})\s*[\–\-\to]\s*([A-Za-z]+\s*[\'']?\d{2,4}|Present|Current|Till Date)'
    matches = re.findall(date_range_pattern, cv_text, re.IGNORECASE)
    
    total_months = 0
    internship_months = 0
    fulltime_months = 0
    
    work_periods = []
    for start_str, end_str in matches:
        start_dt = parse_date_str(start_str)
        end_dt = parse_date_str(end_str)
        if start_dt and end_dt and start_dt <= end_dt:
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
            pos = cv_text.find(start_str)
            snippet = cv_text[max(0, pos - 80): pos].lower() if pos != -1 else ""
            
            if "intern" in snippet or "trainee" in snippet or "internship" in snippet:
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

    edu_pattern = r'(B\.?Tech|B\.?E\.?|M\.?Tech|B\.?Sc|M\.?Sc|Graduation|Degree).*?([A-Za-z]+\s*[\'']?\d{2,4})'
    edu_matches = re.findall(edu_pattern, cv_text, re.IGNORECASE)
    
    edu_end_date = None
    for edu_title, date_str in edu_matches:
        dt = parse_date_str(date_str)
        if dt:
            edu_end_date = dt
            break

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
            content = content.split("