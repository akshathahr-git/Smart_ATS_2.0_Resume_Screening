# 📊 Smart ATS 2.0 - Resume Screening & Candidate Recommendation System

An intelligent recruitment system that analyzes resumes, extracts candidate skills, compares them with job descriptions, and recommends the most suitable candidates using NLP and Machine Learning techniques.

---

## 🌐 Live Demo

🔗 **Try it here:** [https://smartats20resumescreening-h7tv3zmdbdxwcjidewenyf.streamlit.app/](https://smartats20resumescreening-h7tv3zmdbdxwcjidewenyf.streamlit.app/)

**Demo Credentials:**
- **HR Login:** `admin` / `admin123`
- **Candidate:** No login required - just select role and upload resume

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Resume Parsing** | Extract text from PDF and TXT resumes |
| 🧠 **NLP Processing** | Tokenization, stopword removal, lemmatization using NLTK |
| 🔧 **Skill Extraction** | Identify technical skills from text using predefined database |
| 📊 **Candidate Ranking** | Score and rank candidates based on JD match |
| 🔍 **Similarity Analysis** | Compare resumes with job descriptions using skill intersection |
| 🎯 **Recommendation Engine** | Suggest top candidates for each role |
| 📧 **Email Automation** | Send personalized feedback emails to candidates |
| 📁 **JD Management** | Save, edit, and delete job descriptions |
| 🎨 **Visualization** | Color-coded match score distribution graph with thresholds |
| 🌐 **Web Application** | Interactive interface with Streamlit |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.13 |
| **Web Framework** | Streamlit |
| **NLP** | NLTK (punkt, stopwords, wordnet, averaged_perceptron_tagger) |
| **PDF Processing** | PyPDF2 |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **PDF Generation** | ReportLab |
| **Email** | SMTP (smtplib) |
| **Animations** | Streamlit-Lottie |

---
## 📧 SMTP Email Configuration

To send emails from the application:

1. **Create a Gmail App Password:**
   - Go to Google Account → Security → App Passwords
   - Select "Mail" and "Windows Computer"
   - Click Generate and copy the 16-character password

2. **Add to the app:**
   - Enter your Gmail and the app password in the "Send Feedback" tab
   - Or add to `.streamlit/secrets.toml`:
   ```toml
   SMTP_USER = "your-email@gmail.com"
   SMTP_PASSWORD = "your-app-password"
   

## 📦 Installation

### Prerequisites
- Python 3.8+ installed
- Git installed
- 2GB+ RAM recommended

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/akshathahr-git/Smart_ATS_2.0_Resume_Screening.git
cd Smart_ATS_2.0_Resume_Screening

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
