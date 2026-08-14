# app.py - Smart ATS — Professional Version (FINAL)
import streamlit as st
import os, re, io, hashlib, json, math
import PyPDF2
import pandas as pd
from difflib import SequenceMatcher
import nltk

# ---------- SESSION STATE INITIALIZATION - MUST BE FIRST ----------
if "login" not in st.session_state:
    st.session_state["login"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None
if "just_registered" not in st.session_state:
    st.session_state["just_registered"] = False
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "Candidate"

# ---------- JD MANAGEMENT ----------
JD_DB_FILE = "jds.json"

def load_jds():
    """Load JDs from JSON file"""
    if os.path.exists(JD_DB_FILE):
        try:
            with open(JD_DB_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_jds(jds):
    """Save JDs to JSON file"""
    with open(JD_DB_FILE, "w") as f:
        json.dump(jds, f, indent=4)

# ---------- TECHNICAL SKILLS DATABASE ----------
TECHNICAL_SKILLS = {
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", 
    "Go", "Rust", "Swift", "Kotlin", "PHP", "Scala", "R", "MATLAB",
    "Perl", "Lua", "Dart", "Elixir", "Clojure", "Haskell", "Groovy",
    
    # Frameworks & Libraries
    "React", "Angular", "Vue.js", "Django", "Flask", "Spring Boot", 
    "Node.js", "Express.js", "ASP.NET", "Rails", "Laravel", "FastAPI",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "JQuery", "Bootstrap", "Tailwind", "Sass", "Webpack",
    "Next.js", "Nuxt.js", "Svelte", "Gatsby", "Remix",
    "Keras", "OpenCV", "NLTK", "spaCy", "Transformers",
    # ✅ ADDED
    "scikit-learn", "sklearn", "scikit learn",
    "tensorflow", "pytorch",
    
    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", 
    "SQL Server", "Firebase", "Cassandra", "Elasticsearch", "DynamoDB",
    "Neo4j", "InfluxDB", "CouchDB", "MariaDB", "SQLite",
    
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", 
    "GitLab CI", "GitHub Actions", "Terraform", "Ansible", "Chef",
    "Puppet", "SaltStack", "OpenShift", "CloudFormation", "CircleCI",
    # ✅ ADDED
    "aws", "azure", "gcp",
    
    # Data Science & ML
    "NLP", "Computer Vision", "Deep Learning", "Machine Learning",
    "Data Mining", "Data Visualization", "Statistics", "Big Data",
    "Hadoop", "Spark", "Kafka", "Airflow", "Tableau", "Power BI",
    # ✅ ADDED
    "nlp", "deep learning", "machine learning", "statistics", 
    "data visualization", "tableau", "power bi",
    
    # Tools
    "Git", "Linux", "Jira", "Confluence", "Agile", "Scrum",
    "Kafka", "RabbitMQ", "Postman", "Swagger", "Figma",
    "Adobe XD", "Photoshop", "Illustrator", "Sketch",
    "Jenkins", "SonarQube", "Grafana", "Prometheus",
    
    # Testing
    "Selenium", "JUnit", "PyTest", "Mocha", "Cypress", 
    "Jest", "TestNG", "Cucumber", "Postman", "SoapUI",
    "Mockito", "Chai", "Jasmine", "Karma",
    
    # Mobile
    "React Native", "Flutter", "Xamarin", "iOS", "Android",
    "SwiftUI", "Jetpack Compose", "Ionic", "Cordova",
    
    # Certifications & Others
    "AWS Certified", "Azure Certified", "Google Cloud Certified",
    "PMP", "Scrum Master", "CISSP", "CCNA", "MCSE", "OCP",
    
    # Architecture & Design
    "Microservices", "REST API", "GraphQL", "gRPC", "SOAP",
    "Event Driven", "Serverless", "MQTT", "WebSocket",
    # ✅ ADDED
    "rest api", "microservices",
}

def extract_technical_skills(text):
    """Extract only technical skills from text"""
    if not text:
        return set()
    text_lower = text.lower()
    found_skills = set()
    for skill in TECHNICAL_SKILLS:
        if skill.lower() in text_lower:
            found_skills.add(skill)
    return found_skills

# ---------- role classification from skills ----------
def classify_role_from_skills(skills):
    """Classify role based on technical skills found in resume"""
    if not skills:
        return "General/Other"
    
    # Non-technical skills to ignore
    NON_TECHNICAL = {
        "communication", "team management", "ms office", "excel", "powerpoint", 
        "basic computer knowledge", "team player", "quick learner", "adaptable",
        "good communication", "leadership", "problem solving", "time management",
        "organizational skills", "interpersonal skills", "microsoft office",
        "word", "outlook", "presentation skills", "verbal communication"
    }
    
    # Filter out non-technical skills
    technical_skills = []
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower not in NON_TECHNICAL:
            technical_skills.append(skill_lower)
    
    # If no technical skills, return General/Other
    if not technical_skills:
        return "General/Other"
    
    ROLE_KEYWORDS = {
        "Python Developer": ["python", "django", "flask", "fastapi"],
        "Java Developer": ["java", "spring", "hibernate", "maven"],
        "Frontend Developer": ["react", "angular", "vue", "html", "css", "javascript"],
        "Full Stack Developer": ["react", "node", "python", "java", "javascript", "html", "css"],
        "Data Scientist": ["python", "r", "tensorflow", "pytorch", "sklearn", "pandas", "numpy", "machine learning", "deep learning", "nlp", "statistics"],
        "Data Analyst": ["sql", "tableau", "power bi", "python", "r", "data analysis"],
        "Machine Learning Engineer": ["tensorflow", "pytorch", "keras", "scikit-learn", "ml", "deep learning"],
        "Data Engineer": ["spark", "hadoop", "etl", "airflow", "kafka", "python", "sql"],
        "DevOps Engineer": ["docker", "kubernetes", "jenkins", "aws", "azure", "terraform", "ci/cd"],
        "Cloud Engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
        "AI Researcher": ["python", "tensorflow", "pytorch", "nlp", "deep learning", "research"],
        "Statistician": ["r", "python", "statistics", "regression", "spss", "sas"],
        "Business Analyst": ["sql", "power bi", "tableau", "business analysis", "requirements"],
        "Software Engineer": ["python", "java", "c++", "javascript", "git", "sql"],
        "QA Engineer": ["selenium", "junit", "pytest", "testng", "cypress"],
        "Backend Developer": ["python", "java", "node", "django", "flask", "spring", "sql"],
    }
    
    scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            for skill in technical_skills:
                if kw.lower() in skill or skill in kw.lower():
                    score += 1
                    break
        scores[role] = score
    
    best_role = max(scores, key=lambda k: scores[k])
    
    if scores[best_role] == 0:
        return "General/Other"
    
    return best_role

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

# ---------- text extraction helpers ----------
def extract_text_from_pdf(file_like):
    try:
        reader = PyPDF2.PdfReader(file_like)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return text, len(reader.pages)
    except Exception:
        return "", 0

def extract_text_from_file(file_like, file_extension):
    """Extract text from PDF or TXT file"""
    if file_extension == "pdf":
        return extract_text_from_pdf(file_like)
    else:  # txt
        try:
            if hasattr(file_like, 'read'):
                content = file_like.read()
                if isinstance(content, bytes):
                    text = content.decode("utf-8", errors="ignore")
                else:
                    text = str(content)
            else:
                text = str(file_like)
            return text, 1
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
    salary_ranges = {
        "Data Scientist": {"Fresher": "5-8 LPA", "Junior": "8-12 LPA", "Mid-level": "12-18 LPA", "Senior": "18-25 LPA"},
        "Data Analyst": {"Fresher": "3-5 LPA", "Junior": "5-7 LPA", "Mid-level": "7-10 LPA", "Senior": "10-15 LPA"},
        "Machine Learning Engineer": {"Fresher": "6-10 LPA", "Junior": "10-15 LPA", "Mid-level": "15-22 LPA", "Senior": "22-30 LPA"},
        "Data Engineer": {"Fresher": "5-8 LPA", "Junior": "8-12 LPA", "Mid-level": "12-18 LPA", "Senior": "18-25 LPA"},
        "AI Researcher": {"Fresher": "7-12 LPA", "Junior": "12-18 LPA", "Mid-level": "18-25 LPA", "Senior": "25-35 LPA"},
        "Cloud Engineer": {"Fresher": "5-8 LPA", "Junior": "8-12 LPA", "Mid-level": "12-18 LPA", "Senior": "18-25 LPA"},
        "Software Engineer": {"Fresher": "4-6 LPA", "Junior": "6-10 LPA", "Mid-level": "10-15 LPA", "Senior": "15-22 LPA"},
        "Statistician": {"Fresher": "3-5 LPA", "Junior": "5-8 LPA", "Mid-level": "8-12 LPA", "Senior": "12-18 LPA"},
        "Business Analyst": {"Fresher": "3-5 LPA", "Junior": "5-8 LPA", "Mid-level": "8-12 LPA", "Senior": "12-18 LPA"},
        "General/Other": {"Fresher": "2-4 LPA", "Junior": "4-6 LPA", "Mid-level": "6-10 LPA", "Senior": "10-15 LPA"}
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
def send_feedback_email(smtp_user, smtp_password, to_email, pdf_path, candidate_name, subject=None, body=None):
    """Send feedback email with custom subject and body"""
    msg = EmailMessage()
    
    if subject:
        msg["Subject"] = subject
    else:
        msg["Subject"] = "Smart ATS — Your Feedback Report"
    
    msg["From"] = smtp_user
    msg["To"] = to_email
    
    if body:
        msg.set_content(body)
    else:
        msg.set_content(f"Dear {candidate_name},\n\nPlease find attached your Smart ATS feedback report.\n\nRegards,\nSmart ATS Team")
    
    if pdf_path and os.path.exists(pdf_path):
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

# ---------- SIDEBAR LOGIN ----------
with st.sidebar:
    st.markdown("### 🔐 Login / Signup")
    st.markdown("---")
    
    if st.session_state.get("login", False):
        st.markdown(f"""
        <div style='background: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 4px solid #D4AF37;'>
            <b>👤 Logged in as:</b><br>
            {st.session_state.get('user')} 
            <span style='color: #2C3E6B; font-weight: 600;'>({st.session_state.get('role')})</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.rerun()
    else:
        st.markdown("### 👥 Select Your Role")
        st.markdown("---")
        
        st.markdown("""
        <style>
        div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 5px 0;
        }
        div[role="radiogroup"] label {
            padding: 10px 15px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            cursor: pointer;
            font-weight: 500;
        }
        div[role="radiogroup"] label:hover {
            border-color: #2196F3;
            background: #f5f9ff;
        }
        div[role="radiogroup"] label[data-checked="true"] {
            border-color: #2196F3;
            background: #E3F2FD;
            color: #1565C0;
            font-weight: 600;
        }
        div[role="radiogroup"] label[data-checked="true"] .st-emotion-cache-1g26t5d {
            background-color: #2196F3 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        selected_role = st.radio(
            "",
            options=["👤 Candidate", "🏢 HR"],
            index=0,
            label_visibility="collapsed",
            horizontal=False
        )
        
        # Save selected role to session state
        st.session_state.selected_role = selected_role
        
        if selected_role == "👤 Candidate":
            selected_role = "Candidate"
        else:
            selected_role = "HR"
        
        st.markdown("---")
        
        if selected_role == "Candidate":
            st.info("📄 Upload your resume on the main page")
        else:
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            
            HR_USERNAME = "admin"
            HR_PASSWORD = "admin123"
            
            if st.button("🔑 Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.warning("Please enter username and password")
                elif username == HR_USERNAME and password == HR_PASSWORD:
                    st.session_state["login"] = True
                    st.session_state["role"] = "HR"
                    st.session_state["user"] = username
                    st.success("HR login successful!")
                    st.rerun()
                else:
                    st.error("Invalid HR credentials")

# ---------- MAIN CONTENT ----------
if not st.session_state.get("login", False):
    # Show header always
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1B2A4A 0%, #2C3E6B 50%, #1B2A4A 100%);
        padding: 25px 35px;
        border-radius: 15px;
        color: #ffffff;
        font-weight: 700;
        font-size: 26px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(27, 42, 74, 0.3);
        letter-spacing: 0.5px;
        border-bottom: 4px solid #D4AF37;
    '>
        🎯 HireVision <span style='color: #D4AF37;'>AI</span> – Intelligent Resume Analyzer
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Candidate role is selected in sidebar
    if st.session_state.get("selected_role") == "👤 Candidate":
        st.subheader("📄 Upload Your Resume")
        st.write("Upload your resume (PDF or TXT format) to get started. HR will review and contact you via email.")
        
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=["pdf", "txt"],
            help="Upload a PDF or TXT file. Max size: 5MB"
        )
        
        if uploaded_file:
            if not can_upload_more():
                st.error(f"❌ Upload capacity reached ({MAX_RESUMES}). Contact admin.")
            else:
                data = uploaded_file.read()
                file_extension = uploaded_file.name.split('.')[-1].lower()
                safe_name = f"candidate_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
                save_path = os.path.join(UPLOAD_DIR, safe_name)
                
                with open(save_path, "wb") as f:
                    f.write(data)
                
                if file_extension == "pdf":
                    text, pages = extract_text_from_pdf(io.BytesIO(data))
                else:
                    try:
                        text = data.decode("utf-8", errors="ignore")
                        pages = 1
                    except Exception:
                        text = ""
                        pages = 0
                
                name = extract_name(text) or "Unknown"
                email = extract_email(text) or "Not found"
                phone = extract_phone(text) or "Not found"
                
                st.success("✅ Resume uploaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📛 Name", name)
                with col2:
                    st.metric("📧 Email", email)
                with col3:
                    st.metric("📱 Phone", phone)
                
                st.balloons()
    else:
        # Show message when no role selected or HR is selected
        st.markdown("""
        <div style='
            text-align: center; 
            padding: 60px 20px; 
            background: #f8f9fc; 
            border-radius: 15px; 
            border: 2px dashed #2C3E6B; 
            margin-top: 20px;
        '>
            <p style='color: #6c757d; font-size: 18px;'>👈 Please select a role (<b>Candidate</b> or <b>HR</b>) from the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ---------- CSS ----------
CSS = """
<style>
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .stApp {
        background-color: #ffffff;
    }
    .header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2C3E6B 50%, #1B2A4A 100%);
        padding: 25px 35px;
        border-radius: 15px;
        color: #ffffff;
        font-weight: 700;
        font-size: 26px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(27, 42, 74, 0.3);
        letter-spacing: 0.5px;
        border-bottom: 4px solid #D4AF37;
    }
    .header span {
        color: #D4AF37;
    }
    .card {
        background: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 14px;
        border-left: 5px solid #2C3E6B;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border: 1px solid #eef2f7;
    }
    .card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 20px rgba(44, 62, 107, 0.12);
        border-color: #D4AF37;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2C3E6B, #1B2A4A);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(44, 62, 107, 0.35);
    }
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f8f9fc;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #eef2f7;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        background-color: transparent;
        color: #495057;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2C3E6B, #1B2A4A) !important;
        color: white !important;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #6c757d;
        padding: 25px 0;
        border-top: 2px solid #eef2f7;
        margin-top: 35px;
        background: #f8f9fc;
        border-radius: 12px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .dataframe thead {
        background: #2C3E6B !important;
        color: white !important;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# Header (only shown when logged in)
st.markdown(
    "<div class='header'>🎯 HireVision <span>AI</span> – Intelligent Resume Analyzer</div>",
    unsafe_allow_html=True
)

# ---------- HR Panel ----------
if st.session_state["role"] == "HR":
    st.header("👔 HR Dashboard")
    st.markdown("---")
    
    tabs = st.tabs(["📁 View All", "🏆 Rank & Analyze", "🔍 Filter", "📧 Send Feedback", "⚙️ Admin"])
    users = load_users()
    stored_files = os.listdir(UPLOAD_DIR)

    # Tab 1: View All - UPDATED (Username removed)
with tabs[0]:
    st.subheader("📊 Candidate Database")
    
    # Get all uploaded resumes with extracted info
    resume_data = []
    for f in stored_files:
        file_path = os.path.join(UPLOAD_DIR, f)
        file_extension = f.split('.')[-1].lower()
        
        with open(file_path, "rb") as fh:
            b = fh.read()
        
        if file_extension == "pdf":
            text, pages = extract_text_from_pdf(io.BytesIO(b))
        else:
            try:
                text = b.decode("utf-8", errors="ignore")
                pages = 1
            except Exception:
                text = ""
                pages = 0
        
        name = extract_name(text) or f.split("_")[-1] if "_" in f else f
        email = extract_email(text) or "Not found"
        phone = extract_phone(text) or "Not found"
        
        resume_data.append({
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Resume": f,
            "Uploaded": datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d")
        })
    
    if resume_data:
        # ✅ Show only Name, Resume, and Uploaded columns (Username removed)
        df_resumes = pd.DataFrame(resume_data)
        display_df = df_resumes[['Name', 'Resume', 'Uploaded']]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Delete functionality
        st.subheader("🗑️ Delete Resume")
        col1, col2 = st.columns([3, 1])
        with col1:
            resume_to_delete = st.selectbox(
                "Select resume to delete",
                options=[r["Resume"] for r in resume_data],
                help="Select a resume file to permanently delete"
            )
        with col2:
            if st.button("🗑️ Delete Selected", use_container_width=True, type="secondary"):
                if resume_to_delete:
                    file_path = os.path.join(UPLOAD_DIR, resume_to_delete)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        st.success(f"✅ Deleted: {resume_to_delete}")
                        st.rerun()
    else:
        st.info("ℹ️ No resumes uploaded yet.")
    
    st.subheader("📁 Uploaded Resumes")
if anim_search: 
    st_lottie(anim_search, height=100, key="lottie_search")

if stored_files:
    for f in stored_files[:10]:
        file_path = os.path.join(UPLOAD_DIR, f)
        
        # Expandable card with view option
        with st.expander(f"📄 {f}"):
            # Read and display file content
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    st.text_area("Resume Content", content, height=200, key=f"view_{f}")
            except:
                # For PDF files - show download option
                st.warning("📄 PDF file - Click download to view")
            
            # Download button
            with open(file_path, "rb") as file:
                st.download_button(
                    label="📥 Download Resume",
                    data=file,
                    file_name=f,
                    mime="text/plain" if f.endswith('.txt') else "application/pdf",
                    key=f"download_{f}",
                    use_container_width=True
                )
    
    if len(stored_files) > 10:
        st.info(f"... and {len(stored_files) - 10} more files")
else:
    st.info("ℹ️ No resumes uploaded yet.")

    # Tab 2: Rank & Analyze - COMPLETELY UPDATED
# Tab 2: Rank & Analyze - UPDATED with Non-Technical Filter
with tabs[1]:
    st.subheader("🎯 Resume Ranking & Analysis")
    
    # JD Selection - Built-in or Paste
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Load saved JDs
        jds = load_jds()
        jd_options = ["📝 Paste New JD"] + list(jds.keys())
        selected_jd_title = st.selectbox("Select JD or Paste New", options=jd_options)
    
    with col2:
        if selected_jd_title != "📝 Paste New JD":
            if st.button("✏️ Edit Selected JD", use_container_width=True):
                st.session_state.editing_jd = selected_jd_title
                st.rerun()
    
    # JD Text Area
    if selected_jd_title != "📝 Paste New JD" and selected_jd_title in jds:
        jd_text = jds[selected_jd_title]
        st.info(f"📄 Using saved JD: **{selected_jd_title}**")
    else:
        jd_text = ""
    
    # Show JD in text area (editable)
    jd = st.text_area(
        "Job Description",
        value=jd_text if selected_jd_title != "📝 Paste New JD" else "",
        height=200,
        placeholder="Paste the complete job description here...",
        key="jd_input"
    )
    
    # Save/Update JD Buttons
    if selected_jd_title != "📝 Paste New JD":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("💾 Update JD", use_container_width=True):
                if jd:
                    jds[selected_jd_title] = jd
                    save_jds(jds)
                    st.success(f"✅ JD '{selected_jd_title}' updated!")
                    st.rerun()
        with col2:
            if st.button("🗑️ Delete JD", use_container_width=True):
                if selected_jd_title in jds:
                    del jds[selected_jd_title]
                    save_jds(jds)
                    st.success(f"✅ JD '{selected_jd_title}' deleted!")
                    st.rerun()
    else:
        # Save new JD
        new_jd_title = st.text_input("JD Title (to save this JD)", placeholder="e.g., Senior Python Developer")
        if st.button("💾 Save New JD", use_container_width=True):
            if jd and new_jd_title:
                jds[new_jd_title] = jd
                save_jds(jds)
                st.success(f"✅ JD '{new_jd_title}' saved!")
                st.rerun()
            else:
                st.warning("⚠️ Please enter both JD content and title")
    
    if st.button("🚀 Analyze Resumes", use_container_width=True, type="primary"):
        if not jd:
            st.warning("⚠️ Please paste a job description first.")
        elif not stored_files:
            st.info("ℹ️ No resumes uploaded to analyze.")
        else:
            with st.spinner("Analyzing resumes... Please wait..."):
                results = []
                seen_hashes = {}
                
                # Extract technical skills from JD
                jd_skills = extract_technical_skills(jd)
                
                # ✅ Non-technical skills to filter out
                NON_TECHNICAL = {
                    "excel", "powerpoint", "ms office", "microsoft office", "word", "outlook",
                    "communication", "team management", "team player", "quick learner",
                    "adaptable", "good communication", "leadership", "problem solving",
                    "time management", "organizational skills", "interpersonal skills",
                    "presentation skills", "verbal communication", "basic computer knowledge",
                    "computer knowledge", "soft skills", "management skills", "office",
                    "power bi", "tableau", "sql", "python", "r", "statistics", "data analysis"
                }
                
                for f in stored_files:
                    path = os.path.join(UPLOAD_DIR, f)
                    file_extension = f.split('.')[-1].lower()
                    
                    with open(path, "rb") as fh:
                        b = fh.read()
                    
                    # Extract text based on file type
                    if file_extension == "pdf":
                        text, pages = extract_text_from_pdf(io.BytesIO(b))
                    else:  # txt
                        try:
                            text = b.decode("utf-8", errors="ignore")
                            pages = 1
                        except Exception:
                            text = ""
                            pages = 0
                    
                    # Extract technical skills from resume
                    resume_skills = extract_technical_skills(text)
                    
                    # ✅ Filter out non-technical skills
                    filtered_resume_skills = {skill for skill in resume_skills if skill.lower() not in NON_TECHNICAL}
                    
                                        # Calculate technical skills match
                    matched_skills = jd_skills.intersection(filtered_resume_skills)
                    missing_skills = jd_skills - filtered_resume_skills

                    # Technical Skills Match % (40%)
                    tech_match_pct = int((len(matched_skills) / max(1, len(jd_skills))) * 100)

                    # Experience Level (30%)
                    exp_level = predict_experience_level(text)
                    exp_map = {"Fresher": 50, "Junior": 70, "Mid-level": 85, "Senior": 100}
                    exp_match = exp_map.get(exp_level, 70)

                    # Role Match (20%) - Use filtered skills
                    role_pred = classify_role_from_skills(filtered_resume_skills)
                    if "Data Scientist" in role_pred or "Machine Learning" in role_pred:
                        role_match = 100
                    elif "Data Analyst" in role_pred or "Software Engineer" in role_pred:
                        role_match = 70
                    else:
                        role_match = 50

                    # Education (10%)
                    edu_score = 50
                    edu_keywords = ["phd", "doctorate", "master", "m.sc", "m.s", "bachelor", "b.sc", "b.s", "b.tech", "m.tech", "mca", "bca"]
                    for keyword in edu_keywords:
                        if keyword in text.lower():
                            if "phd" in text.lower() or "doctorate" in text.lower():
                                edu_score = 100
                            elif "master" in text.lower() or "m.sc" in text.lower() or "m.tech" in text.lower():
                                edu_score = 80
                            elif "bachelor" in text.lower() or "b.sc" in text.lower() or "b.tech" in text.lower():
                                edu_score = 60
                            break

                    # Calculate Total Score (Weighted)
                    total_score = int(
                        (tech_match_pct * 0.4) +
                        (exp_match * 0.3) +
                        (role_match * 0.2) +
                        (edu_score * 0.1)
                    )

                    # Name extraction
                    name = extract_name(text) or f.split("_", 1)[-1]
                    email = extract_email(text) or ""
                    phone = extract_phone(text) or ""
                    
                    
                    # Name extraction
                    name = extract_name(text) or f.split("_", 1)[-1]
                    email = extract_email(text) or ""
                    phone = extract_phone(text) or ""
                    
                    # Salary estimation
                    salary = estimate_salary(role_pred, exp_level)
                    
                    # Scam flag
                    scam_flag = scam_score(text)
                    
                    # Hash for duplicates
                    h = file_hash_bytes(b)
                    duplicate_of = seen_hashes.get(h, "")
                    if not duplicate_of:
                        seen_hashes[h] = f
                    
                    results.append({
                        "Resume": f,
                        "Name": name,
                        "Email": email or "",
                        "Phone": phone or "",
                        "Match %": total_score,
                        "Pages": pages,
                        "Role": role_pred,
                        "ExpLevel": exp_level,
                        "Salary": salary,
                        "Matched Skills": ", ".join(sorted(matched_skills)) if matched_skills else "None",
                        "Missing Skills": ", ".join(sorted(missing_skills)) if missing_skills else "None",
                        "ScamFlag": scam_flag,
                        "DuplicateOf": duplicate_of
                    })
                
                df = pd.DataFrame(results).sort_values("Match %", ascending=False).reset_index(drop=True)
                st.success("✅ Analysis Complete!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Candidates", len(df))
                with col2:
                    avg_match = int(df["Match %"].mean()) if len(df) else 0
                    st.metric("Avg Match", f"{avg_match}%")
                with col3:
                    top_role = df['Role'].mode()[0] if len(df) and not df['Role'].empty else "—"
                    st.metric("Top Role", top_role)
                with col4:
                    top_exp = df['ExpLevel'].mode()[0] if len(df) and not df['ExpLevel'].empty else "—"
                    st.metric("Top Experience", top_exp)
                
                # Color-coded graph with threshold
                st.subheader("📊 Match Score Distribution")
                if not df.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create color map based on match %
                    colors = []
                    for val in df['Match %']:
                        if val >= 70:
                            colors.append('#2ECC71')  # Green - Excellent
                        elif val >= 50:
                            colors.append('#F1C40F')  # Yellow - Good
                        elif val >= 30:
                            colors.append('#E67E22')  # Orange - Average
                        else:
                            colors.append('#E74C3C')  # Red - Poor
                    
                    # Create bars
                    bars = ax.bar(df['Name'], df['Match %'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{int(height)}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
                    # Add threshold lines
                    ax.axhline(y=70, color='#27AE60', linestyle='--', linewidth=3, label='Target (70%)', alpha=0.8)
                    ax.axhline(y=50, color='#E74C3C', linestyle='--', linewidth=3, label='Threshold (50%)', alpha=0.8)
                    
                    ax.set_ylabel("Match %", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Candidates", fontsize=12, fontweight='bold')
                    ax.set_title("JD Match Scores - Color Coded", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylim(0, 110)
                    plt.tight_layout()
                    
                    # Add legend for colors
                    legend_elements = [
                        plt.Rectangle((0,0),1,1, fc='#2ECC71', label='Excellent (≥70%)'),
                        plt.Rectangle((0,0),1,1, fc='#F1C40F', label='Good (50-69%)'),
                        plt.Rectangle((0,0),1,1, fc='#E67E22', label='Average (30-49%)'),
                        plt.Rectangle((0,0),1,1, fc='#E74C3C', label='Poor (<30%)')
                    ]
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
                    
                    st.pyplot(fig)
                
                # Display results table without Score and Words columns
                st.subheader("📋 Detailed Results")
                display_df = df[['Name', 'Role', 'Match %', 'Email', 'ExpLevel', 'Salary', 'Pages', 'Matched Skills', 'Missing Skills']]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.session_state["latest_df"] = df
                
                # Expanded view for each candidate
                for i, row in df.iterrows():
                    with st.expander(f"📄 {row['Name']} — {row['Match %']}% — {row['Role']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Email:** {row['Email'] or 'N/A'}")
                            st.write(f"**Phone:** {row['Phone'] or 'N/A'}")
                            st.write(f"**Experience:** {row['ExpLevel']}")
                            st.write(f"**Salary Estimate:** {row['Salary']}")
                            if row['DuplicateOf']:
                                st.warning(f"⚠️ Duplicate of: {row['DuplicateOf']}")
                        with col2:
                            st.write(f"**Pages:** {row['Pages']}")
                            st.write(f"**Match %:** {row['Match %']}%")
                            st.write(f"**Role:** {row['Role']}")
                        
                        st.write("**✅ Matched Technical Skills:**")
                        st.code(row['Matched Skills'] or "None", language="text")
                        
                        st.write("**⚠️ Missing Technical Skills:**")
                        st.code(row['Missing Skills'] or "None", language="text")
                        
                        if row['ScamFlag']:
                            st.error("⚠️ **Scam Alert:** This resume contains suspicious patterns. Manual review recommended.")
                            
# Tab 3: Filter - UPDATED with Role Filter and Removed Skills Columns
with tabs[2]:
    st.subheader("🔍 Filter Candidates by Role")
    
    # Check if analysis has been run
    if "latest_df" not in st.session_state:
        st.warning("⚠️ Please run 'Rank & Analyze' first to prepare the data.")
    else:
        df = st.session_state["latest_df"]
        
        # Get all unique roles from the analysis
        all_roles = sorted(df['Role'].unique().tolist())
        role_options = ["All Roles"] + all_roles
        
        selected_role_filter = st.selectbox("Select Role to Filter", options=role_options)
        
        # Optional: Minimum match % filter
        min_match = st.slider("Minimum Match %", min_value=0, max_value=100, value=50, step=5)
        
        if st.button("🔎 Apply Filter", use_container_width=True):
            # Filter by role
            if selected_role_filter == "All Roles":
                filtered_df = df
            else:
                filtered_df = df[df['Role'] == selected_role_filter]
            
            # Filter by minimum match %
            filtered_df = filtered_df[filtered_df['Match %'] >= min_match]
            
            # Sort by Match % (highest first)
            filtered_df = filtered_df.sort_values("Match %", ascending=False)
            
            if not filtered_df.empty:
                st.success(f"✅ Found {len(filtered_df)} candidates matching criteria")
                
                # Display only relevant columns (REMOVED Matched Skills and Missing Skills)
                display_df = filtered_df[['Name', 'Role', 'Match %', 'Email', 'ExpLevel', 'Salary']]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Show cards with candidate info
                for i, row in filtered_df.iterrows():
                    st.markdown(f"""
                    <div class='card' style='border-left-color: #28a745;'>
                        <b>📌 {row['Name']}</b> — <span style='color: #2C3E6B; font-weight: bold;'>{row['Match %']}% Match</span><br>
                        <span style='color: #6c757d;'>Role: {row['Role']} | Experience: {row['ExpLevel']} | Salary: {row['Salary']}</span><br>
                        <span style='color: #6c757d;'>Email: {row['Email'] or 'N/A'}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download option
                csv = filtered_df[['Name', 'Role', 'Match %', 'Email', 'ExpLevel', 'Salary']].to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    file_name=f"filtered_{selected_role_filter.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info(f"ℹ️ No candidates found matching the criteria")

# Tab 4: Send Feedback - UPDATED with Selected Candidates Only
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
        
        # Show summary of candidates
        shortlisted = len(df[df['Match %'] >= 50])
        rejected = len(df[df['Match %'] < 50])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", len(df))
        with col2:
            st.metric("✅ Shortlisted (>50%)", shortlisted)
        with col3:
            st.metric("📝 Need Improvement (<50%)", rejected)
        
        st.write("📋 **Select candidates to send feedback**")
        
        # Create checkboxes for each candidate
        selected_indices = []
        for i, row in df.iterrows():
            # Show checkbox with candidate info
            if st.checkbox(f"📧 {row['Name']} — {row['Role']} — {row['Match %']}% Match", key=f"select_{i}"):
                selected_indices.append(i)
        
        if selected_indices:
            st.info(f"✅ Selected {len(selected_indices)} candidates for feedback")
            
            if st.button(f"📤 Send Feedback to {len(selected_indices)} Selected Candidates", use_container_width=True, type="primary"):
                if not smtp_user or not smtp_password:
                    st.error("❌ Please provide SMTP credentials")
                else:
                    with st.spinner(f"Sending emails to {len(selected_indices)} candidates..."):
                        sent = 0
                        failed = 0
                        for idx in selected_indices:
                            row = df.iloc[idx]
                            email = row['Email'] if row['Email'] else None
                            
                            if not email:
                                path = os.path.join(UPLOAD_DIR, row['Resume'])
                                file_extension = row['Resume'].split('.')[-1].lower()
                                with open(path, "rb") as fh:
                                    b = fh.read()
                                if file_extension == "pdf":
                                    txt, _ = extract_text_from_pdf(io.BytesIO(b))
                                else:
                                    try:
                                        txt = b.decode("utf-8", errors="ignore")
                                    except Exception:
                                        txt = ""
                                email = extract_email(txt)
                            
                            if email:
                                try:
                                    if row['Match %'] >= 50:
                                        subject = f"🎉 Congratulations! You've been shortlisted for {row['Role']}"
                                        body = f"""
Dear {row['Name']},

Congratulations! Your profile matches {row['Match %']}% of our requirements for the {row['Role']} position.

We are impressed with your qualifications and would like to invite you for the next round of interviews. Our HR team will reach out to you shortly to schedule an interview at your convenience.

Looking forward to speaking with you!

Best regards,
Hiring Team
"""
                                    else:
                                        missing_skills = row['Missing Skills']
                                        subject = f"📝 Feedback on your application for {row['Role']}"
                                        body = f"""
Dear {row['Name']},

Thank you for applying for the {row['Role']} position at our company. We appreciate your interest and the time you took to submit your application.

After carefully reviewing your profile against our requirements, we noticed some skill gaps that need improvement:

❌ Missing Technical Skills:
{missing_skills}

💡 Suggestions to improve your profile:
1. Focus on acquiring the above technical skills through online courses or certifications
2. Build projects using these technologies to gain practical experience
3. Update your resume to highlight any relevant experience or projects

📚 Recommended resources:
- Coursera / Udemy for technical courses
- GitHub for open-source contributions
- LinkedIn Learning for professional development

We encourage you to upskill in these areas and reapply in the future. We'd love to see your improved profile!

Best regards,
Hiring Team
"""
                                    send_feedback_email(smtp_user, smtp_password, email, None, row['Name'], subject, body)
                                    sent += 1
                                except Exception as e:
                                    failed += 1
                                    st.error(f"Failed to send to {row['Name']}: {e}")
                            else:
                                failed += 1
                                st.warning(f"No email found for {row['Name']}")
                        
                        st.success(f"✅ Email process complete! Sent: {sent}, Failed: {failed}")
        else:
            st.info("☝️ Select at least one candidate to send feedback")

# Tab 5: Admin - UNCHANGED
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
    <b>HireVision AI</b> — Smart ATS System • <span style='color: #D4AF37;'>Professional Edition</span><br>
    <small>© 2024 All Rights Reserved • For demonstration purposes only</small>
</div>
""", unsafe_allow_html=True)