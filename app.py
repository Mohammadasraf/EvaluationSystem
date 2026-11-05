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
LOGIC_VERSION = "v15.0-Universal-Multilingual-Engine"

# ------------------------------------------------------------------------------
# UNIVERSAL LANGUAGE DETECTION & TRANSLATION ENGINE
# ------------------------------------------------------------------------------
def detect_and_translate_to_english(text: str, groq_api_key: str) -> tuple[str, str]:
    """
    Detects the input text language and translates non-English text into English.
    Returns a tuple of (translated_text, detected_language).
    """
    if not text or not text.strip():
        return "", "English"

    if not groq_api_key:
        return text, "Unknown (No API Key)"

    system_prompt = (
        "You are an expert multilingual document translator and language detector. "
        "Analyze the input text. Detect its primary language. "
        "If the text is NOT in English, translate it accurately to professional English, maintaining all numbers, dates, technical skills, job titles, and formatting context. "
        "If the text is already in English, return it unchanged. "
        "Return ONLY valid JSON matching this schema: {\"detected_language\": \"Spanish\", \"translated_text\": \"...\"}"
    )

    sample_text = text[:3000]

    user_prompt = f"""Analyze and translate the following document text if it is not in English:

--- DOCUMENT TEXT ---
{sample_text}

Return JSON strictly:
{{
  "detected_language": "Language Name (e.g., Spanish, French, German, English)",
  "translated_text": "Full English translated text..."
}}
"""

    try:
        client = Groq(api_key=groq_api_key.strip())
        completion = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=2500
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(repair_json(content))
        detected_lang = parsed.get("detected_language", "English")
        
        if len(text) <= 3000:
            translated_text = parsed.get("translated_text", text)
        else:
            if "english" not in detected_lang.lower():
                full_prompt = f"Translate the following {detected_lang} document to English. Output ONLY the translated text without extra commentary:\n\n{text}"
                full_completion = client.chat.completions.create(
                    model=MODEL_VERSION,
                    messages=[{"role": "system", "content": "You are a professional translator. Translate text accurately to English."}, {"role": "user", "content": full_prompt}],
                    temperature=0.0,
                    max_tokens=4000
                )
                translated_text = full_completion.choices[0].message.content.strip()
            else:
                translated_text = text

        return translated_text, detected_lang
    except Exception:
        return text, "English (Fallback)"

# ------------------------------------------------------------------------------
# COMPLETE PII ANONYMIZATION & MASKING ENGINE (UNIFIED ADDRESS TAG)
# ------------------------------------------------------------------------------
def mask_pii_in_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Mask Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    
    # 2. Mask Phone Numbers
    text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[REDACTED_PHONE]', text)
    text = re.sub(r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'\[REDACTED_PHONE\](?:\s*[\/\-,]\s*\[REDACTED_PHONE\])+', '[REDACTED_PHONE]', text)
    
    # 3. Mask URLs
    text = re.sub(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '[REDACTED_URL]', text)
    text = re.sub(r'\b(?:linkedin\.com|github\.com)/[a-zA-Z0-9_\-/]+', '[REDACTED_URL]', text, flags=re.IGNORECASE)

    # 4. Mask Government IDs
    text = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b', '[REDACTED_PAN_CARD]', text)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[REDACTED_AADHAAR]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)

    # 5. Mask Pincodes & Zip Codes
    text = re.sub(r'\b\d{6}\b', '[REDACTED_ADDRESS]', text)
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[REDACTED_ADDRESS]', text)

    # 6. Mask States & Provinces
    states_list = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 
        'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 
        'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 
        'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Delhi', 'Chandigarh', 'California', 'New York',
        'Madrid', 'Catalonia', 'Bavaria', 'Ontario', 'Quebec', 'Ile-de-France'
    ]
    for state in states_list:
        text = re.sub(rf'\b{state}\b', '[REDACTED_ADDRESS]', text, flags=re.IGNORECASE)

    # 7. Mask International Cities
    cities_list = [
        'Bengaluru', 'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Ahmedabad', 'Chennai', 
        'Kolkata', 'Pune', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 
        'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 
        'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Kalyan', 'Vasai', 'Varanasi',
        'New York', 'San Francisco', 'London', 'Singapore', 'Dubai', 'Barcelona', 'Madrid',
        'Paris', 'Berlin', 'Munich', 'Tokyo', 'Toronto', 'Mexico City', 'Bogota', 'Buenos Aires'
    ]
    for city in cities_list:
        text = re.sub(rf'\b{city}\b', '[REDACTED_ADDRESS]', text, flags=re.IGNORECASE)
    
    # 8. Explicit Address Key-Value Pairs (Multilingual)
    text = re.sub(
        r'\b(?:Address|Location|Residing at|Residential Address|Permanent Address|Current Address|Street|Road|Nagar|Marg|Sector|Plot|House No|Flat No|Apartment|Society|City|State|Pincode|Zip Code|Dirección|Ubicación|Domicilio)\s*:\s*[^\n]+',
        '[REDACTED_ADDRESS]',
        text,
        flags=re.IGNORECASE
    )

    # 9. Multilingual Landmark & Street Keywords
    lines = text.split('\n')
    masked_lines = []
    address_keywords = [
        'flat no', 'flat #', 'apartment', 'society', 'residency', 'layout', 
        'block no', 'plot no', 'house no', 'door no', 'street no', 
        'near opp', 'cross street', 'extension nagar', 'greenfield', 
        'enclave', 'towers', 'heights', 'villa', 'colony', 'phase',
        'cross', 'junction', 'circle', 'bypass', 'byepass', 'chowk', 'square', 'main road',
        'calle', 'avenida', 'carrera', 'barrio', 'colonia', 'piso', 'departamento', 'codigo postal'
    ]
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in address_keywords):
            masked_lines.append('[REDACTED_ADDRESS]')
        else:
            masked_lines.append(line)
    text = '\n'.join(masked_lines)

    text = re.sub(r'(\[REDACTED_ADDRESS\]\s*[\/\-,]?\s*)+', '[REDACTED_ADDRESS] ', text)

    # 10. Mask Candidate Names
    text = re.sub(r'\b(?:Candidate Name|Full Name|Name|Nombre|Nombre Completo)\s*:\s*[A-Za-z\s.]+', '[REDACTED_NAME]', text, flags=re.IGNORECASE)
    
    # 11. Mask Date of Birth & Demographics
    text = re.sub(r'\b(?:DOB|Date of Birth|Birth Date|Fecha de Nacimiento)\s*:\s*[^\n]+', '[REDACTED_DOB]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Age|Edad)\s*:\s*\d{1,2}\s*(?:years|yrs|años)?', '[REDACTED_AGE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Gender|Sex|Género|Sexo)\s*:\s*(?:Male|Female|Other|M|F|Masculino|Femenino)', '[REDACTED_GENDER]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Marital Status|Estado Civil)\s*:\s*(?:Single|Married|Unmarried|Divorced|Soltero|Casado|Divorciado)', '[REDACTED_MARITAL_STATUS]', text, flags=re.IGNORECASE)

    return text

def evaluate_hybrid_system_batched(cv_text, jd_text, rules_list, groq_api_key):
    # STEP 1: Universal Auto-Translation for Multilingual Resumes & JDs
    translated_cv, detected_cv_lang = detect_and_translate_to_english(cv_text, groq_api_key)
    translated_jd, detected_jd_lang = detect_and_translate_to_english(jd_text, groq_api_key)

    # STEP 2: Apply Full PII Masking
    masked_cv_text = mask_pii_in_text(translated_cv)

    jd_analysis = extract_jd_requirements(translated_jd, groq_api_key)
    profile_details = calculate_deterministic_timeline(masked_cv_text, groq_api_key)
    profile_details["detected_cv_language"] = detected_cv_lang
    profile_details["detected_jd_language"] = detected_jd_lang

    evidence_map = extract_universal_evidence(masked_cv_text)

    det_analysis = profile_details

    chunk_size = 4
    rule_chunks = [rules_list[i:i + chunk_size] for i in range(0, len(rules_list), chunk_size)]
    
    combined_rule_evals = {}
    score_points = 0.0
    total_rules = len(rules_list)

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

    combined_rule_evals, overall_score, recommendation, summary = apply_fully_dynamic_guardrails(
        jd_analysis, det_analysis, combined_rule_evals, overall_score, "Consider"
    )

    final_output = {
        "Rule Evaluations": combined_rule_evals,
        "Overall Candidate Match Score": overall_score,
        "Derived Recommendation": recommendation,
        "AI Contextual Summary": summary
    }

    return det_analysis, evidence_map, final_output