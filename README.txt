# ⚡ Enterprise Hybrid Candidate Evaluation System

An advanced, enterprise-grade AI-powered recruitment screening tool combining **deterministic pre-parsing** with **Large Language Models (Groq API)** to evaluate candidate resumes against custom job descriptions and dynamic evaluation rules.

---

## 🚀 Key Features

* **Deterministic & AI Hybrid Engine**: Pre-parses skills normalization, professional experience duration (excluding education and birth years), education mapping, and career gaps deterministically.
* **Dynamic N-Rules Builder**: Create, modify, and manage custom evaluation criteria on-the-fly.
* **OCR Support**: Built-in support for scanning image-based PDFs, PNGs, and JPG resumes using Tesseract OCR and PyMuPDF.
* **Groq API Integration**: High-performance evaluation leveraging the Qwen model (`qwen/qwen3.8-27b`) with structured JSON outputs.
* **Automated Word (.docx) Reporting**: Generates a downloadable executive summary and evaluation matrix report.
* **Persistent Audit Trail**: Automatically logs all evaluations, timestamps, scores, and recommendations into an SQLite database (`candidate_evaluator.db`).

---

## 🛠️ Tech Stack

* **Frontend / UI**: [Streamlit](https://streamlit.io/)
* **AI Provider**: Groq API (`qwen/qwen3.8-27b`)
* **Database**: SQLite3
* **Document Processing**: PyMuPDF (`fitz`), `python-docx`, `Pillow`
* **Data Processing**: Pandas

---

## 📦 Prerequisites & Installation

1. **Install Dependencies**:
   ```bash
   pip install streamlit fitz python-docx pandas pillow requests pytesseract