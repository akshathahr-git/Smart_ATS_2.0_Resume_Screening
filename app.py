# app.py - Smart ATS — Final (full version with requested enhancements)
import streamlit as st
import os, re, io, hashlib, json, math
import PyPDF2
import pandas as pd
from difflib import SequenceMatcher
import nltk
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

# Optional OpenAI integration (if you set OPENAI_API_KEY in env)
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
    # expand as needed
}

# ---------- constants ----------
MAX_RESUMES = 100  # capacity limit

# ---------- UI CSS ----------
st.set_page_config(page_title="Smart ATS — Enhanced", layout="wide")
CSS = """
:root{
  --accent1: #6C63FF;
  --accent2: #FF5A7A;
}
.header { background: linear-gradient(90deg, var(--accent1), var(--accent2)); padding:18px; border-radius:12px; color:#ffffff; font-weight:700; font-size:22px; text-align:center; margin-bottom:18px;}
.card { background: #f5f7ff10; padding:12px; border-radius:10px; margin-bottom:10px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.markdown(
    "<div class='header'>HireVision AI – Intelligent Resume Analyzer</div>",
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
        # exact match
        if s in resume_tokens:
            matched.add(s)
            continue
        # synonyms mapping
        syns = SYNONYMS.get(s, [])
        if any(normalize(syn) in resume_tokens for syn in syns):
            matched.add(s)
            continue
        # reverse check
        for k, v in SYNONYMS.items():
            if s == normalize(k):
                if any(normalize(x) in resume_tokens for x in v):
                    matched.add(s)
                    break
        if s in matched:
            continue
        # fuzzy match fallback
        for r in resume_tokens:
            if SequenceMatcher(None, s, r).ratio() >= 0.82:
                matched.add(s)
                break
    return matched, jd_tokens - matched

# ---------- resume quality heuristics ----------
POSITIVE_WORDS = {"lead", "develop", "achieved", "improved", "reduced", "optimized", "built", "designed"}
NEGATIVE_WORDS = {"responsible", "assisted", "helped", "duties"}

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
    "Data Scientist": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "sklearn"],
    "Data Analyst": ["sql", "tableau", "powerbi", "excel", "data analysis"],
    "Backend Engineer": ["django", "flask", "api", "microservice", "postgres", "mysql"],
    "Frontend Engineer": ["react", "angular", "vue", "javascript", "css", "html"],
    "DevOps": ["docker", "kubernetes", "ci/cd", "aws", "azure"],
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
    base = {
        "Data Scientist": 800000,
        "Data Analyst": 400000,
        "Backend Engineer": 600000,
        "Frontend Engineer": 500000,
        "DevOps": 650000,
        "General/Other": 350000
    }
    mult = {"Fresher": 0.6, "Junior": 0.8, "Mid-level": 1.0, "Senior": 1.4}
    salary = int(base.get(role, 350000) * mult.get(experience_level, 1.0))
    return f"₹{salary//1000}k - ₹{int(salary*1.3)//1000}k"

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

# ---------- Sidebar login/signup ----------
with st.sidebar:
    st.markdown("## 🔐 Login / Signup")
    role = st.radio("Login as", ["Candidate", "HR"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login / Signup"):
        users = load_users()

        if role == "Candidate":
            if not username or not password:
                st.warning("Enter username and password to create candidate account.")
            else:
                if username in users:
                    st.warning(
                        "Username already exists. "
                        "(One resume per username rule). "
                        "Create a new username to upload another resume."
                    )
                else:
                    if not can_upload_more():
                        st.error(
                            f"Upload capacity reached ({MAX_RESUMES}). "
                            "Admin must clear storage first."
                        )
                    else:
                        users[username] = {
                            "password": password,
                            "uploaded_resume": None,
                            "created_at": datetime.now().isoformat()
                        }

                        save_users(users)

                        st.success(
                            f"Candidate account created: {username}. "
                            "You can now upload a single resume."
                        )

                        st.session_state["login"] = True
                        st.session_state["role"] = "Candidate"
                        st.session_state["user"] = username

        else:
            # HR login - fixed username and password
            HR_USERNAME = "admin"
            HR_PASSWORD = "admin123"

            if not username or not password:
                st.warning("Enter HR username and password.")

            elif username == HR_USERNAME and password == HR_PASSWORD:
                st.session_state["login"] = True
                st.session_state["role"] = "HR"
                st.session_state["user"] = username
                st.success("HR login successful!")

            else:
                st.error("Invalid HR username or password.")

    if st.session_state["login"]:
        st.markdown(
            f"**Logged in as:** "
            f"{st.session_state.get('user')} "
            f"({st.session_state.get('role')})"
        )

        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged out. Please refresh page.")
            st.stop()


if not st.session_state["login"]:
    st.info("Please login as Candidate or HR from the sidebar.")
    st.stop()

# ---------- Candidate Panel ----------
if st.session_state["role"] == "Candidate":
    st.header("🙋 Candidate Panel — Upload Resume")
    st.write("Upload exactly one resume (PDF). To upload another resume later, create a new username.")
    current_user = st.session_state["user"]
    users = load_users()
    user_record = users.get(current_user, {})
    uploaded_fname = user_record.get("uploaded_resume")
    if uploaded_fname:
        st.success(f"You have already uploaded a resume: **{uploaded_fname}**")
        st.info("To upload a different resume create another username.")
        # show download link
        path = os.path.join(UPLOAD_DIR, uploaded_fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button("⬇️ Download your uploaded resume", f, file_name=uploaded_fname, mime="application/pdf")
        st.stop()

    st.write("Make sure your resume contains your email so HR can send feedback.")
    uploaded = st.file_uploader("Upload PDF Resume", type="pdf")
    if uploaded:
        if not can_upload_more():
            st.error(f"Upload capacity reached ({MAX_RESUMES}). Contact admin to clear older resumes.")
        else:
            data = uploaded.read()
            # save file with username prefix to avoid duplicates
            safe_name = f"{current_user}_{uploaded.name}"
            save_path = os.path.join(UPLOAD_DIR, safe_name)
            with open(save_path, "wb") as f:
                f.write(data)
            # update user record
            users[current_user]["uploaded_resume"] = safe_name
            users[current_user]["uploaded_at"] = datetime.now().isoformat()
            save_users(users)
            st.success(f"✅ Resume saved as {safe_name}")
            text, pages = extract_text_from_pdf(io.BytesIO(data))
            st.write("**Extracted Contact Info:**")
            st.write(f"- Name: {extract_name(text) or 'Not found'}")
            st.write(f"- Email: {extract_email(text) or 'Not found'}")
            st.write(f"- Phone: {extract_phone(text) or 'Not found'}")
            st.write(f"- Pages: {pages}")
            # optional: preview first 400 chars
            st.subheader("Resume Preview (first 800 chars)")
            st.code(text[:800] + ("" if len(text) < 800 else "\n\n..."), language="text")

# ---------- HR Panel ----------
if st.session_state["role"] == "HR":
    st.header("🧑‍💼 HR Dashboard — Smart ATS")
    tabs = st.tabs(["📁 View All", "🏆 Rank & Analyze", "🔍 Filter & Shortlist", "📧 Send Feedback", "⚙️ Admin"])
    users = load_users()
    stored_files = os.listdir(UPLOAD_DIR)

    # Tab 1: View All
    with tabs[0]:
        st.subheader("Stored Candidate Accounts (username / password / uploaded file)")
        if users:
            rows = []
            for uname, meta in users.items():
                rows.append({
                    "Username": uname,
                    "Password": meta.get("password", ""),
                    "UploadedResume": meta.get("uploaded_resume", ""),
                    "CreatedAt": meta.get("created_at", ""),
                    "UploadedAt": meta.get("uploaded_at", "")
                })
            df_users = pd.DataFrame(rows)
            st.dataframe(df_users, use_container_width=True)
        else:
            st.info("No candidate accounts yet.")

        st.subheader("Uploaded Resumes")
        if anim_search: st_lottie(anim_search, height=140, key="lottie_search")
        if stored_files:
            for f in stored_files:
                st.markdown(f"<div class='card'><b>{f}</b></div>", unsafe_allow_html=True)
        else:
            st.info("No resumes uploaded.")

    # Tab 2: Rank & Analyze
    with tabs[1]:
        st.subheader("Rank Top Candidates")
        jd = st.text_area("Paste Job Description (JD)", height=200, placeholder="Paste JD text here...")
        if st.button("🚀 Rank Resumes", key="rank"):
            if not jd:
                st.warning("Please paste a JD first.")
            elif not stored_files:
                st.info("No resumes uploaded.")
            else:
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
                        "Similarity": similarity,
                        "Score": score,
                        "Pages": pages,
                        "Words": len(text.split()),
                        "Matched": ", ".join(sorted(matched)),
                        "Missing": ", ".join(sorted(missing)),
                        "Suggestions": "; ".join(suggestions),
                        "ScamFlag": scam_flag,
                        "Role": role_pred,
                        "ExpLevel": exp_level,
                        "SalaryEstimate": salary,
                        "DuplicateOf": duplicate_of
                    })
                df = pd.DataFrame(results).sort_values("Similarity", ascending=False).reset_index(drop=True)
                st.success("✅ Analysis Completed")
                # vertical line graph
                st.subheader("📊 Similarity Trend (Vertical Line Graph)")
                if not df.empty:
                    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * len(df)))
                    x = df['Resume']
                    y = df['Similarity']
                    ax.plot(x, y, marker='o', linestyle='-', linewidth=2)
                    ax.set_xlabel("Resume")
                    ax.set_ylabel("Similarity %")
                    ax.set_title("JD Similarity Score per Resume")
                    plt.xticks(rotation=45, ha='right')
                    plt.ylim(0, 100)
                    st.pyplot(fig)
                # metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Candidates", len(df))
                with col2:
                    avg_sim = int(df["Similarity"].mean()) if len(df) else 0
                    st.metric("Avg Similarity", f"{avg_sim}%")
                with col3:
                    top_role = df['Role'].mode()[0] if len(df) else "—"
                    st.metric("Top Role", top_role)
                st.subheader("Detailed Results")
                st.dataframe(df.drop(columns=['Suggestions']), use_container_width=True)
                # expanders per resume
                for i, row in df.iterrows():
                    with st.expander(f"{row['Resume']} — {row['Similarity']}% — {row['Role']}"):
                        st.write(f"*Name:* {row['Name']}  |  *Email:* {row['Email']}  |  *Phone:* {row['Phone']}")
                        st.write(f"*Pages:* {row['Pages']}  |  *Words:* {row['Words']}")
                        st.write(f"*Matched Skills:* {row['Matched']}")
                        st.write(f"*Missing Skills:* {row['Missing']}")
                        recs = []
                        if row['Missing']:
                            miss_list = [m.strip() for m in row['Missing'].split(",") if m.strip()]
                            for s in miss_list:
                                for key in ["tensorflow","pytorch","sql","nlp","react","docker","aws","sql","pandas"]:
                                    if key in s:
                                        recs.append((s, "Online course suggestion", "https://www.coursera.org/"))
                                        break
                        st.write(f"*Skill Suggestions:* {recs if recs else 'None'}")
                        st.write(f"*Resume Suggestions:* {row['Suggestions'] if row['Suggestions'] else 'None'}")
                        if row['ScamFlag']:
                            st.warning("⚠️ This resume contains suspicious patterns. Manual review recommended.")
                        if st.button(f"Generate Feedback PDF: {row['Resume']}", key=f"fb_{i}"):
                            pdf_file = create_feedback_pdf(row['Name'], row['Similarity'], row['Matched'], row['Missing'], row['Suggestions'].split('; ') if row['Suggestions'] else [], row['ScamFlag'], row['Role'], row['ExpLevel'])
                            with open(pdf_file, "rb") as fh:
                                st.download_button("⬇️ Download Feedback PDF", fh, file_name=os.path.basename(pdf_file), mime="application/pdf")
                st.session_state["latest_df"] = df

    # Tab 3: Filter & Shortlist
    with tabs[2]:
        st.subheader("Filter by Skill Match / Shortlist")
        jd2 = st.text_area("Paste JD for filtering", height=150)
        if st.button("🔎 Filter Now"):
            if not jd2:
                st.warning("Paste a JD.")
            elif not stored_files:
                st.info("No resumes.")
            else:
                jd_tokens = extract_jd_keywords(jd2)
                filtered = []
                for f in stored_files:
                    path = os.path.join(UPLOAD_DIR, f)
                    with open(path, "rb") as fh:
                        b = fh.read()
                    text, pages = extract_text_from_pdf(io.BytesIO(b))
                    tokens = tokenize_resume(text)
                    matched, missing = match_skills(jd_tokens, tokens)
                    if len(matched) / max(1, len(jd_tokens)) >= 0.5:
                        filtered.append({"Resume": f, "Matched": ", ".join(sorted(matched)), "Missing": ", ".join(sorted(missing))})
                if filtered:
                    st.write("### Resumes matching ≥50%")
                    for r in filtered:
                        st.markdown(f"<div class='card'><b>{r['Resume']}</b><br>Matched: {r['Matched']}<br>Missing: {r['Missing']}</div>", unsafe_allow_html=True)
                else:
                    st.info("No matches ≥50%")

    # Tab 4: Send Feedback
    with tabs[3]:
        st.subheader("Send Feedback Emails to Candidates")
        if anim_send: 
            st_lottie(anim_send, height=140, key="lottie_send")
        st.write("⚠️ Use Gmail App Password for SMTP (recommended).")
        
        smtp_user = st.text_input("SMTP Email (from)", value="", placeholder="yourorg@gmail.com", key="smtp_user")
        smtp_password = st.text_input("SMTP App Password", type="password", key="smtp_pass")
        
        if "latest_df" not in st.session_state:
            st.info("Run 'Rank Resumes' first to prepare feedback list.")
        else:
            df = st.session_state["latest_df"]
            st.write(f"Ready to send feedback for {len(df)} candidates.")
            
            if st.button("📧 Send Feedback to All (generate PDFs + email)"):
                if not smtp_user or not smtp_password:
                    st.warning("Provide SMTP credentials.")
                else:
                    sent = 0
                    for i, row in df.iterrows():
                        pdf_file = create_feedback_pdf(
                            row['Name'], row['Similarity'], row['Matched'], row['Missing'],
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
                                st.error(f"Failed to send to {row['Resume']}: {e}")
                        else:
                            st.warning(f"No email found for {row['Resume']}; skipped.")
                    st.success(f"Email send complete. {sent} emails sent.")

    # Tab 5: Admin / Utilities
    with tabs[4]:
        st.subheader("Admin / Utilities")
        st.write(f"Current uploads: {len(os.listdir(UPLOAD_DIR))} / {MAX_RESUMES}")

        if st.button("🧹 Clear uploaded resumes"):
            for f in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, f))
            users = load_users()
            for u in users:
                users[u]["uploaded_resume"] = None
                users[u].pop("uploaded_at", None)
            save_users(users)
            st.session_state.clear()
            st.success("Uploaded resumes cleared. Please refresh page.")
            st.stop()

        if st.button("🗑 Clear generated PDFs"):
            for f in os.listdir(PDFS_DIR):
                os.remove(os.path.join(PDFS_DIR, f))
            st.success("Generated PDFs cleared.")

        if st.button("❌ Clear all users & resumes"):
            for f in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, f))
            for f in os.listdir(PDFS_DIR):
                os.remove(os.path.join(PDFS_DIR, f))
            if os.path.exists(USER_DB_FILE):
                os.remove(USER_DB_FILE)
            st.session_state.clear()
            st.success("All users, resumes, and PDFs cleared. Refresh page.")
            st.stop()

        st.markdown("**Optional:** Provide OPENAI_API_KEY in environment to enable AI-enhanced resume rewriting.")
        st.write("Saved folders (on server):")
        st.write(f"- Uploads: {os.path.abspath(UPLOAD_DIR)}")
        st.write(f"- PDFs: {os.path.abspath(PDFS_DIR)}")
        st.write(f"- User DB: {os.path.abspath(USER_DB_FILE)}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:12px; color:#666;'>Smart ATS — Enhanced • Demo / Academic use • Do not expose passwords in production</div>", unsafe_allow_html=True)
