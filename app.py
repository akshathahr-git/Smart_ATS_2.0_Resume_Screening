# app.py - Smart ATS — Professional Version (Final)
import streamlit as st
import os, re, io, hashlib, json, math
import PyPDF2
import pandas as pd
from difflib import SequenceMatcher
import nltk

# Download NLTK resources
resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/stopwords", "stopwords"),
]

for path, name in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie
from datetime import datetime
import smtplib
from email.message import EmailMessage
import textwrap
import warnings
warnings.filterwarnings("ignore")

# Optional OpenAI integration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        OPENAI_API_KEY = None

# ---------- NLTK setup ----------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
STOP = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- storage ----------
UPLOAD_DIR = "uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PDFS_DIR = "generated_feedbacks"
os.makedirs(PDFS_DIR, exist_ok=True)
USER_DB_FILE = "candidate_users.json"
if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as f:
        json.dump({}, f)

# ---------- synonym mapping ----------
SYNONYMS = {
    "artificial intelligence": ["ai"],
    "machine learning": ["ml"],
    "internet of things": ["iot"],
    "natural language processing": ["nlp"],
    "deep learning": ["dl"],
    "tensorflow": ["tf"],
    "scikit-learn": ["sklearn"],
    "tableau": ["power bi dashboards", "powerbi", "power bi"],
    "power bi": ["powerbi", "tableau"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform"],
}

# ---------- constants ----------
MAX_RESUMES = 100

# ---------- Professional UI CSS ----------
st.set_page_config(
    page_title="HireVision AI - Smart ATS", 
    page_icon="🎯",
    layout="wide"
)

CSS = """
<style>
    /* Remove default Streamlit padding */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Professional white background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: #ffffff;
        font-weight: 700;
        font-size: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
    }
    
    .header span {
        color: #e94560;
    }
    
    /* Card styling */
    .card {
        background: #f8f9fa;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 4px solid #0f3460;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 52, 96, 0.3);
        background: linear-gradient(135deg, #16213e, #0f3460);
    }
    
    /* Password input without suggestions */
    input[type="password"] {
        -webkit-text-security: disc !important;
    }
    
    /* Hide browser password manager icons */
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        background-color: transparent;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f3460 !important;
        color: white !important;
    }
    
    /* Success/Warning/Info messages */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        font-size: 12px;
        color: #6c757d;
        padding: 20px 0;
        border-top: 1px solid #e9ecef;
        margin-top: 30px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #0f3460;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #16213e;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='header'>🎯 HireVision <span>AI</span> – Intelligent Resume Analyzer</div>",
    unsafe_allow_html=True
)

# ---------- helper: lottie ----------
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

anim_search = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_4kx2q32n.json")
anim_send = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json")

# ---------- pdf/text helpers ----------
def extract_text_from_pdf(file_like):
    try:
        reader = PyPDF2.PdfReader(file_like)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return text, len(reader.pages)
    except Exception:
        return "", 0

def normalize(word):
    return lemmatizer.lemmatize(re.sub(r"[^a-z0-9 ]+", "", word.lower()))

def extract_email(text):
    if not text: return None
    matches = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return matches[0].strip() if matches else None

def extract_phone(text):
    if not text: return None
    m = re.findall(r"(\+?\d[\d\-\s\(\)]{7,}\d)", text)
    return m[0].strip() if m else None

def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: return None
    candidates = lines[:5]
    for c in candidates:
        tokens = c.split()
        if 1 < len(tokens) <= 4 and all(t[0].isupper() for t in tokens if t):
            return c
    return candidates[0]

def extract_jd_keywords(text):
    words = word_tokenize(text)
    tags = pos_tag(words)
    kws = set()
    curr = []
    for w, t in tags:
        if t.startswith(('N', 'V', 'J')) and len(w) > 2 and w.lower() not in STOP:
            curr.append(w.lower())
        else:
            if curr:
                kws.add(normalize(" ".join(curr))); curr = []
    if curr: kws.add(normalize(" ".join(curr)))
    for w in words:
        if w.isalpha() and len(w) > 2 and w.lower() not in STOP:
            kws.add(normalize(w))
    return set(kws)

def tokenize_resume(text):
    return set([normalize(w) for w in word_tokenize(text) if w.isalpha()])

# ---------- synonym-aware matching ----------
def match_skills(jd_tokens, resume_tokens):
    matched = set()
    for s in jd_tokens:
        if s in resume_tokens:
            matched.add(s)
            continue
        syns = SYNONYMS.get(s, [])
        if any(normalize(syn) in resume_tokens for syn in syns):
            matched.add(s)
            continue
        for k, v in SYNONYMS.items():
            if s == normalize(k):
                if any(normalize(x) in resume_tokens for x in v):
                    matched.add(s)
                    break
        if s in matched:
            continue
        for r in resume_tokens:
            if SequenceMatcher(None, s, r).ratio() >= 0.82:
                matched.add(s)
                break
    return matched, jd_tokens - matched

# ---------- resume quality heuristics ----------
POSITIVE_WORDS = {"lead", "develop", "achieved", "improved", "reduced", "optimized", "built", "designed", "architected", "implemented"}
NEGATIVE_WORDS = {"responsible", "assisted", "helped", "duties", "tasked"}

def resume_score_and_suggestions(text, jd_tokens):
    score = 50
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    wc = max(1, len(words))
    kw_matches = sum(1 for k in jd_tokens if any(k_word in words for k_word in k.split()))
    score += int(min(30, (kw_matches / max(1, len(jd_tokens))) * 30)) if jd_tokens else 0
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    score += min(10, pos_count * 2)
    if wc < 150: score -= 5
    if wc > 3000: score -= 10
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    score -= min(10, neg_count * 2)
    score = max(5, min(100, score))
    suggestions = []
    if jd_tokens and kw_matches / max(1, len(jd_tokens)) < 0.5:
        suggestions.append("Include more JD-specific keywords and technologies near the top.")
    if pos_count < 2:
        suggestions.append("Use stronger action verbs like 'developed', 'optimized', 'led'.")
    if wc < 150:
        suggestions.append("Add more detail about projects and results (numbers help).")
    return score, suggestions

# ---------- scam detection ----------
SUSPICIOUS_PATTERNS = [
    r"\bcompleted\s+in\s+(one|1)\s+day\b",
    r"\bteam\s+of\s+1\b",
    r"\bbuilt\s+from\s+scratch\s+in\s+\d+\s+days\b",
]
def scam_score(text):
    s = 0
    for p in SUSPICIOUS_PATTERNS:
        if re.search(p, text, flags=re.I):
            s += 1
    buzz = len(re.findall(r"\b(ai|blockchain|deep learning|nlp|big data|microservices|docker|kubernetes)\b", text, flags=re.I))
    if buzz > 12: s += 1
    return s

# ---------- role classification & experience ----------
ROLE_KEYWORDS = {
    "Data Scientist": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "sklearn", "data science"],
    "Data Analyst": ["sql", "tableau", "powerbi", "excel", "data analysis", "dashboard"],
    "Machine Learning Engineer": ["tensorflow", "pytorch", "keras", "ml", "model deployment", "aws"],
    "Data Engineer": ["spark", "etl", "airflow", "data pipeline", "big data"],
    "AI Researcher": ["research", "nlp", "transformers", "publication", "llm"],
    "Cloud Engineer": ["aws", "gcp", "docker", "kubernetes", "terraform", "ci/cd"],
    "Software Engineer": ["java", "spring", "mysql", "git", "html", "css"],
    "Statistician": ["r", "spss", "statistics", "regression", "probability"],
    "Business Analyst": ["excel", "power bi", "requirements", "kpi", "business analysis"]
}

def classify_role(tokens):
    scores = {}
    for role, kws in ROLE_KEYWORDS.items():
        scores[role] = sum(1 for k in kws if any(k in t for t in tokens))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "General/Other"

def predict_experience_level(text):
    m = re.findall(r"(\d{1,2})\+?\s+years", text, flags=re.I)
    if m:
        years = max(int(x) for x in m)
    else:
        yrs = re.findall(r"(19|20)\d{2}", text)
        if len(yrs) >= 2:
            years = abs(int(yrs[-1]) - int(yrs[0]))
        else:
            years = 0
    if years < 1: return "Fresher"
    if years < 3: return "Junior"
    if years < 6: return "Mid-level"
    return "Senior"

def estimate_salary(role, experience_level):
    # Salary in Lakhs per annum (LPA)
    salary_ranges = {
        "Data Scientist": {
            "Fresher": "5-8 LPA",
            "Junior": "8-12 LPA", 
            "Mid-level": "12-18 LPA",
            "Senior": "18-25 LPA"
        },
        "Data Analyst": {
            "Fresher": "3-5 LPA",
            "Junior": "5-7 LPA",
            "Mid-level": "7-10 LPA",
            "Senior": "10-15 LPA"
        },
        "Machine Learning Engineer": {
            "Fresher": "6-10 LPA",
            "Junior": "10-15 LPA",
            "Mid-level": "15-22 LPA",
            "Senior": "22-30 LPA"
        },
        "Data Engineer": {
            "Fresher": "5-8 LPA",
            "Junior": "8-12 LPA",
            "Mid-level": "12-18 LPA",
            "Senior": "18-25 LPA"
        },
        "AI Researcher": {
            "Fresher": "7-12 LPA",
            "Junior": "12-18 LPA",
            "Mid-level": "18-25 LPA",
            "Senior": "25-35 LPA"
        },
        "Cloud Engineer": {
            "Fresher": "5-8 LPA",
            "Junior": "8-12 LPA",
            "Mid-level": "12-18 LPA",
            "Senior": "18-25 LPA"
        },
        "Software Engineer": {
            "Fresher": "4-6 LPA",
            "Junior": "6-10 LPA",
            "Mid-level": "10-15 LPA",
            "Senior": "15-22 LPA"
        },
        "Statistician": {
            "Fresher": "3-5 LPA",
            "Junior": "5-8 LPA",
            "Mid-level": "8-12 LPA",
            "Senior": "12-18 LPA"
        },
        "Business Analyst": {
            "Fresher": "3-5 LPA",
            "Junior": "5-8 LPA",
            "Mid-level": "8-12 LPA",
            "Senior": "12-18 LPA"
        },
        "General/Other": {
            "Fresher": "2-4 LPA",
            "Junior": "4-6 LPA",
            "Mid-level": "6-10 LPA",
            "Senior": "10-15 LPA"
        }
    }
    
    if role in salary_ranges:
        return salary_ranges[role].get(experience_level, "2-4 LPA")
    return "2-4 LPA"

# ---------- duplicate detection ----------
def file_hash_bytes(b):
    return hashlib.sha256(b).hexdigest()

# ---------- feedback PDF creation ----------
def create_feedback_pdf(name, score, matched, missing, suggestions, scam_flag, role, exp_level):
    fname = os.path.join(PDFS_DIR, f"{name.replace(' ','_')}_feedback_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    c = canvas.Canvas(fname, pagesize=A4)
    texty = 780
    c.setFont("Helvetica-Bold", 18)
    c.drawString(60, texty, "SMART ATS — Feedback Report")
    texty -= 28
    c.setFont("Helvetica", 11)
    c.drawString(60, texty, f"Candidate: {name}")
    texty -= 16
    c.drawString(60, texty, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    texty -= 22
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, texty, f"JD Match Score: {score}%")
    texty -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, texty, f"Predicted Role: {role} | Experience Level: {exp_level}")
    texty -= 18
    c.drawString(60, texty, "Matched Skills:")
    texty -= 14
    c.setFont("Helvetica", 10)
    c.drawString(70, texty, matched if matched else "None")
    texty -= 16
    c.setFont("Helvetica", 11)
    c.drawString(60, texty, "Missing / Recommended Skills:")
    texty -= 14
    c.setFont("Helvetica", 10)
    c.drawString(70, texty, missing if missing else "None")
    texty -= 16
    c.setFont("Helvetica-Bold", 11)
    c.drawString(60, texty, "Suggestions:")
    texty -= 14
    c.setFont("Helvetica", 10)
    for s in suggestions:
        wrapped = textwrap.wrap(s, width=80)
        for line in wrapped:
            c.drawString(70, texty, "- " + line)
            texty -= 12
            if texty < 80:
                c.showPage(); texty = 780
    c.setFont("Helvetica", 10)
    texty -= 6
    if scam_flag:
        c.setFillColorRGB(0.9, 0.2, 0.2)
        c.drawString(60, texty, "⚠️ This resume contains suspicious patterns. Recommend manual review.")
        c.setFillColorRGB(0,0,0)
    c.showPage()
    c.save()
    return fname

# ---------- email sending ----------
def send_feedback_email(smtp_user, smtp_password, to_email, pdf_path, candidate_name):
    msg = EmailMessage()
    msg["Subject"] = "Smart ATS — Your Feedback Report"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(f"Dear {candidate_name},\n\nPlease find attached your Smart ATS feedback report.\n\nRegards,\nSmart ATS Team")
    with open(pdf_path, "rb") as f:
        data = f.read()
        msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)

# ---------- manage user DB ----------
def load_users():
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

def save_users(d):
    with open(USER_DB_FILE, "w") as f:
        json.dump(d, f, indent=2)

# ---------- helper: storage capacity ----------
def can_upload_more():
    return len(os.listdir(UPLOAD_DIR)) < MAX_RESUMES

# ---------- session state ----------
if "login" not in st.session_state:
    st.session_state["login"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None
if "just_registered" not in st.session_state:
    st.session_state["just_registered"] = False

# ---------- Sidebar login/signup ----------
with st.sidebar:
    st.markdown("### 🔐 Login / Signup")
    st.markdown("---")
    
    role = st.radio("Select Role", ["Candidate", "HR"], index=0)
    username = st.text_input("Username", placeholder="Enter your username", key="username_input")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input", autocomplete="off")
    
    # Custom CSS to hide password suggestions
    st.markdown("""
    <style>
        input[type="password"] {
            -webkit-text-security: disc !important;
        }
        input:-webkit-autofill {
            -webkit-box-shadow: 0 0 0px 1000px white inset !important;
        }
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if user exists (including HR admin)
    users = load_users()
    HR_USERNAME = "admin"
    HR_PASSWORD = "admin123"
    
    is_hr_admin = (username == HR_USERNAME)
    is_candidate_user = (username and username in users)
    user_exists = is_hr_admin or is_candidate_user
    
    if user_exists:
        if st.button("🔑 Login", use_container_width=True):
            if not password:
                st.warning("⚠️ Please enter your password")
            else:
                if role == "Candidate":
                    if is_candidate_user and users[username]["password"] == password:
                        st.session_state["login"] = True
                        st.session_state["role"] = "Candidate"
                        st.session_state["user"] = username
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    if is_hr_admin and password == HR_PASSWORD:
                        st.session_state["login"] = True
                        st.session_state["role"] = "HR"
                        st.session_state["user"] = username
                        st.success("✅ HR login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid HR credentials")
    else:
        if st.button("📝 Sign Up", use_container_width=True):
            if not username or not password:
                st.warning("⚠️ Please enter both username and password")
            else:
                if role != "Candidate":
                    st.error("❌ Only candidates can sign up. HR accounts are pre-configured.")
                elif username in users:
                    st.error("❌ Username already exists. Please choose a different username.")
                elif username == HR_USERNAME:
                    st.error("❌ 'admin' is a reserved username. Please choose a different one.")
                else:
                    if not can_upload_more():
                        st.error(f"❌ Upload capacity reached ({MAX_RESUMES}). Contact admin.")
                    else:
                        users[username] = {
                            "password": password,
                            "uploaded_resume": None,
                            "created_at": datetime.now().isoformat()
                        }
                        save_users(users)
                        st.session_state["login"] = True
                        st.session_state["role"] = "Candidate"
                        st.session_state["user"] = username
                        st.success(f"✅ Account created successfully! Welcome, {username}!")
                        st.rerun()
    
    if st.session_state["login"]:
        st.markdown("---")
        st.markdown(f"""
        <div style='background: #e8f4f8; padding: 12px; border-radius: 8px;'>
            <b>👤 Logged in as:</b><br>
            {st.session_state.get('user')} 
            <span style='color: #0f3460; font-weight: 600;'>({st.session_state.get('role')})</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.rerun()

# ---------- Main Content ----------
if not st.session_state["login"]:
    st.info("👈 Please login or sign up from the sidebar to continue.")
    st.stop()

# ---------- Candidate Panel ----------
if st.session_state["role"] == "Candidate":
    st.header("📄 Candidate Dashboard")
    st.markdown("---")
    
    current_user = st.session_state["user"]
    users = load_users()
    user_record = users.get(current_user, {})
    uploaded_fname = user_record.get("uploaded_resume")
    
    if uploaded_fname:
        st.success(f"✅ You have uploaded your resume: **{uploaded_fname}**")
        path = os.path.join(UPLOAD_DIR, uploaded_fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "📥 Download your resume",
                    f,
                    file_name=uploaded_fname,
                    mime="application/pdf",
                    use_container_width=True
                )
        st.info("💡 Note: Each username can upload only one resume. Create a new account for additional uploads.")
        st.stop()
    
    st.write("📌 **Upload your resume to get started**")
    st.write("Make sure your resume contains your email address so HR can send feedback.")
    
    uploaded = st.file_uploader(
        "Upload Resume (PDF format only)",
        type="pdf",
        help="Upload a single PDF file. Max size: 5MB"
    )
    
    if uploaded:
        if not can_upload_more():
            st.error(f"❌ Upload capacity reached ({MAX_RESUMES}). Contact admin.")
        else:
            data = uploaded.read()
            safe_name = f"{current_user}_{uploaded.name}"
            save_path = os.path.join(UPLOAD_DIR, safe_name)
            
            with open(save_path, "wb") as f:
                f.write(data)
            
            users[current_user]["uploaded_resume"] = safe_name
            users[current_user]["uploaded_at"] = datetime.now().isoformat()
            save_users(users)
            
            st.success("✅ Resume uploaded successfully!")
            
            text, pages = extract_text_from_pdf(io.BytesIO(data))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📛 Name", extract_name(text) or "Not found")
            with col2:
                st.metric("📧 Email", extract_email(text) or "Not found")
            with col3:
                st.metric("📱 Phone", extract_phone(text) or "Not found")
            
            st.subheader("📄 Resume Preview")
            with st.expander("Click to view resume content"):
                st.code(text[:1000] + ("..." if len(text) > 1000 else ""), language="text")
            
            st.rerun()

# ---------- HR Panel ----------
if st.session_state["role"] == "HR":
    st.header("👔 HR Dashboard")
    st.markdown("---")
    
    tabs = st.tabs(["📁 View All", "🏆 Rank & Analyze", "🔍 Filter", "📧 Send Feedback", "⚙️ Admin"])
    users = load_users()
    stored_files = os.listdir(UPLOAD_DIR)

    # Tab 1: View All
    with tabs[0]:
        st.subheader("📊 Candidate Database")
        if users:
            rows = []
            for uname, meta in users.items():
                rows.append({
                    "Username": uname,
                    "Resume": meta.get("uploaded_resume", "❌ Not uploaded"),
                    "Created": meta.get("created_at", "").split("T")[0] if meta.get("created_at") else "",
                    "Uploaded": meta.get("uploaded_at", "").split("T")[0] if meta.get("uploaded_at") else ""
                })
            df_users = pd.DataFrame(rows)
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No candidates registered yet.")
        
        st.subheader("📁 Uploaded Resumes")
        if anim_search: 
            st_lottie(anim_search, height=100, key="lottie_search")
        
        if stored_files:
            for f in stored_files[:10]:
                st.markdown(f"<div class='card'>📄 {f}</div>", unsafe_allow_html=True)
            if len(stored_files) > 10:
                st.info(f"... and {len(stored_files) - 10} more files")
        else:
            st.info("ℹ️ No resumes uploaded yet.")

    # Tab 2: Rank & Analyze
    with tabs[1]:
        st.subheader("🎯 Resume Ranking & Analysis")
        jd = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the complete job description here...",
            key="jd_input"
        )
        
        if st.button("🚀 Analyze Resumes", use_container_width=True):
            if not jd:
                st.warning("⚠️ Please paste a job description first.")
            elif not stored_files:
                st.info("ℹ️ No resumes uploaded to analyze.")
            else:
                with st.spinner("Analyzing resumes... Please wait..."):
                    results = []
                    seen_hashes = {}
                    jd_tokens = extract_jd_keywords(jd)
                    
                    for f in stored_files:
                        path = os.path.join(UPLOAD_DIR, f)
                        with open(path, "rb") as fh:
                            b = fh.read()
                        h = file_hash_bytes(b)
                        text, pages = extract_text_from_pdf(io.BytesIO(b))
                        tokens = tokenize_resume(text)
                        matched, missing = match_skills(jd_tokens, tokens)
                        similarity = int((len(matched) / max(1, len(jd_tokens))) * 100) if jd_tokens else 0
                        score, suggestions = resume_score_and_suggestions(text, jd_tokens)
                        scam_flag = scam_score(text)
                        role_pred = classify_role(tokens)
                        exp_level = predict_experience_level(text)
                        salary = estimate_salary(role_pred, exp_level)
                        name = extract_name(text) or f.split("_",1)[-1]
                        email = extract_email(text)
                        phone = extract_phone(text)
                        duplicate_of = seen_hashes.get(h, "")
                        if not duplicate_of:
                            seen_hashes[h] = f
                        
                        results.append({
                            "Resume": f,
                            "Name": name,
                            "Email": email or "",
                            "Phone": phone or "",
                            "Match %": similarity,
                            "Score": score,
                            "Pages": pages,
                            "Words": len(text.split()),
                            "Matched": ", ".join(sorted(matched)),
                            "Missing": ", ".join(sorted(missing)),
                            "Suggestions": "; ".join(suggestions),
                            "ScamFlag": scam_flag,
                            "Role": role_pred,
                            "ExpLevel": exp_level,
                            "Salary": salary
                        })
                    
                    df = pd.DataFrame(results).sort_values("Match %", ascending=False).reset_index(drop=True)
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Candidates", len(df))
                    with col2:
                        avg_match = int(df["Match %"].mean()) if len(df) else 0
                        st.metric("Avg Match", f"{avg_match}%")
                    with col3:
                        top_role = df['Role'].mode()[0] if len(df) else "—"
                        st.metric("Top Role", top_role)
                    with col4:
                        top_exp = df['ExpLevel'].mode()[0] if len(df) else "—"
                        st.metric("Top Experience", top_exp)
                    
                    st.subheader("📊 Match Score Distribution")
                    if not df.empty:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(df['Name'], df['Match %'], color='#0f3460', alpha=0.7)
                        ax.axhline(y=70, color='green', linestyle='--', label='70% Target')
                        ax.set_ylabel("Match %")
                        ax.set_xlabel("Candidates")
                        ax.set_title("JD Match Scores")
                        plt.xticks(rotation=45, ha='right')
                        plt.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df[['Name', 'Role', 'Match %', 'Score', 'Email', 'ExpLevel']], 
                                use_container_width=True, hide_index=True)
                    
                    st.session_state["latest_df"] = df
                    
                    for i, row in df.iterrows():
                        with st.expander(f"📄 {row['Name']} — {row['Match %']}% — {row['Role']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Email:** {row['Email'] or 'N/A'}")
                                st.write(f"**Phone:** {row['Phone'] or 'N/A'}")
                                st.write(f"**Experience:** {row['ExpLevel']}")
                                st.write(f"**Salary Estimate:** {row['Salary']}")
                            with col2:
                                st.write(f"**Pages:** {row['Pages']}")
                                st.write(f"**Words:** {row['Words']}")
                                st.write(f"**Score:** {row['Score']}/100")
                            
                            st.write("**✅ Matched Skills:**")
                            st.code(row['Matched'] or "None", language="text")
                            
                            st.write("**⚠️ Missing Skills:**")
                            st.code(row['Missing'] or "None", language="text")
                            
                            if row['Suggestions']:
                                st.warning(f"💡 **Suggestions:** {row['Suggestions']}")
                            
                            if row['ScamFlag']:
                                st.error("⚠️ **Scam Alert:** This resume contains suspicious patterns. Manual review recommended.")
                            
                            if st.button(f"Generate Feedback PDF", key=f"gen_pdf_{i}"):
                                pdf_file = create_feedback_pdf(
                                    row['Name'], row['Match %'], 
                                    row['Matched'], row['Missing'],
                                    row['Suggestions'].split('; ') if row['Suggestions'] else [],
                                    row['ScamFlag'], row['Role'], row['ExpLevel']
                                )
                                with open(pdf_file, "rb") as fh:
                                    st.download_button(
                                        "📥 Download PDF",
                                        fh,
                                        file_name=os.path.basename(pdf_file),
                                        mime="application/pdf",
                                        use_container_width=True
                                    )

   # Tab 3: Filter (Simple - Only >50% Match)
with tabs[2]:
    st.subheader("🔍 Filter Candidates (≥50% Match)")
    jd_filter = st.text_area("Paste JD for filtering", height=150, key="filter_jd")
    
    if st.button("🔎 Apply Filter", use_container_width=True):
        if not jd_filter:
            st.warning("⚠️ Please paste a job description.")
        elif not stored_files:
            st.info("ℹ️ No resumes to filter.")
        else:
            with st.spinner("Filtering candidates..."):
                jd_tokens = extract_jd_keywords(jd_filter)
                
                results = []
                for f in stored_files:
                    path = os.path.join(UPLOAD_DIR, f)
                    with open(path, "rb") as fh:
                        b = fh.read()
                    text, pages = extract_text_from_pdf(io.BytesIO(b))
                    tokens = tokenize_resume(text)
                    
                    matched, missing = match_skills(jd_tokens, tokens)
                    
                    if jd_tokens:
                        match_pct = int((len(matched) / len(jd_tokens)) * 100)
                    else:
                        match_pct = 0
                    
                    name = extract_name(text) or f.split("_", 1)[-1]
                    email = extract_email(text) or ""
                    
                    results.append({
                        "Name": name,
                        "Email": email,
                        "Resume": f,
                        "Match %": match_pct,
                        "Matched": ", ".join(sorted(matched)) if matched else "None",
                        "Missing": ", ".join(sorted(missing)) if missing else "None"
                    })
                
                # Filter to >50% match
                filtered_results = [r for r in results if r["Match %"] >= 50]
                
                if filtered_results:
                    # Sort by match percentage
                    filtered_results = sorted(filtered_results, key=lambda x: x["Match %"], ascending=False)
                    st.success(f"✅ Found {len(filtered_results)} candidates with ≥50% match")
                    
                    # Show results in a clean table
                    df_filtered = pd.DataFrame(filtered_results)
                    st.dataframe(
                        df_filtered[['Name', 'Email', 'Match %', 'Matched']], 
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    # Show cards with details
                    for row in filtered_results:
                        st.markdown(f"""
                        <div class='card' style='border-left-color: #28a745;'>
                            <b>📌 {row['Name']}</b> — <span style='color: #0f3460; font-weight: bold;'>{row['Match %']}% Match</span><br>
                            <span style='color: #28a745;'>✅ Matched: {row['Matched'][:150]}</span><br>
                            <span style='color: #dc3545;'>⚠️ Missing: {row['Missing'][:150]}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Export option
                    csv = df_filtered.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        file_name="filtered_candidates.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("ℹ️ No candidates with ≥50% match found.")

    # Tab 4: Send Feedback
    with tabs[3]:
        st.subheader("📧 Send Feedback Emails")
        if anim_send:
            st_lottie(anim_send, height=100, key="lottie_send")
        
        st.info("💡 Use Gmail App Password for SMTP (recommended)")
        
        smtp_user = st.text_input("SMTP Email", placeholder="yourorg@gmail.com", key="smtp_user")
        smtp_password = st.text_input("App Password", type="password", placeholder="Enter app password", key="smtp_pass")
        
        if "latest_df" not in st.session_state:
            st.warning("⚠️ Please run 'Rank & Analyze' first to prepare the feedback list.")
        else:
            df = st.session_state["latest_df"]
            st.write(f"📋 Ready to send feedback for **{len(df)}** candidates")
            
            if st.button("📤 Send Feedback to All", use_container_width=True):
                if not smtp_user or not smtp_password:
                    st.error("❌ Please provide SMTP credentials")
                else:
                    with st.spinner("Sending emails..."):
                        sent = 0
                        failed = 0
                        for i, row in df.iterrows():
                            pdf_file = create_feedback_pdf(
                                row['Name'], row['Match %'], 
                                row['Matched'], row['Missing'],
                                row['Suggestions'].split('; ') if row['Suggestions'] else [],
                                row['ScamFlag'], row['Role'], row['ExpLevel']
                            )
                            email = row['Email'] if row['Email'] else None
                            if not email:
                                path = os.path.join(UPLOAD_DIR, row['Resume'])
                                with open(path, "rb") as fh:
                                    txt, _ = extract_text_from_pdf(fh)
                                email = extract_email(txt)
                            
                            if email:
                                try:
                                    send_feedback_email(smtp_user, smtp_password, email, pdf_file, row['Name'])
                                    sent += 1
                                except Exception as e:
                                    failed += 1
                                    st.error(f"Failed to send to {row['Name']}: {e}")
                            else:
                                failed += 1
                                st.warning(f"No email found for {row['Name']}")
                        
                        st.success(f"✅ Email process complete! Sent: {sent}, Failed: {failed}")

    # Tab 5: Admin
    with tabs[4]:
        st.subheader("⚙️ System Administration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Uploaded Resumes", len(os.listdir(UPLOAD_DIR)), delta=f"Max: {MAX_RESUMES}")
        with col2:
            st.metric("Generated PDFs", len(os.listdir(PDFS_DIR)))
        with col3:
            st.metric("Registered Users", len(load_users()))
        
        st.markdown("---")
        st.warning("⚠️ Admin actions are irreversible. Please use with caution.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Uploaded Resumes", use_container_width=True):
                for f in os.listdir(UPLOAD_DIR):
                    os.remove(os.path.join(UPLOAD_DIR, f))
                users = load_users()
                for u in users:
                    users[u]["uploaded_resume"] = None
                    users[u].pop("uploaded_at", None)
                save_users(users)
                st.success("✅ Resumes cleared. Refreshing...")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Generated PDFs", use_container_width=True):
                for f in os.listdir(PDFS_DIR):
                    os.remove(os.path.join(PDFS_DIR, f))
                st.success("✅ PDFs cleared.")
        
        with col3:
            if st.button("❌ Reset All Data", use_container_width=True):
                for f in os.listdir(UPLOAD_DIR):
                    os.remove(os.path.join(UPLOAD_DIR, f))
                for f in os.listdir(PDFS_DIR):
                    os.remove(os.path.join(PDFS_DIR, f))
                if os.path.exists(USER_DB_FILE):
                    os.remove(USER_DB_FILE)
                st.session_state.clear()
                st.success("✅ All data reset. Refreshing...")
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <b>HireVision AI</b> — Smart ATS System • <span style='color: #0f3460;'>Professional Edition</span><br>
    <small>© 2024 All Rights Reserved • For demonstration purposes only</small>
</div>
""", unsafe_allow_html=True)
