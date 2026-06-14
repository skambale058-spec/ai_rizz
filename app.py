from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import fitz
import re

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except:
    MODEL = None

app = FastAPI(title="Professional Resume Analyzer")

SKILLS = [
    "python","java","javascript","typescript",
    "react","nextjs","nodejs","express",
    "fastapi","django","flask",
    "aws","azure","gcp",
    "docker","kubernetes",
    "mongodb","postgresql","mysql",
    "redis","tensorflow","pytorch",
    "machine learning","deep learning",
    "git","linux","microservices"
]


def extract_pdf_text(file_bytes):
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for page in pdf:
        text += page.get_text()

    return text


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill.lower() in text:
            found.append(skill)

    return sorted(list(set(found)))


def estimate_experience(text):
    years = re.findall(r"(19\d{2}|20\d{2})", text)

    if len(years) < 2:
        return 0

    years = sorted([int(y) for y in years])

    return max(0, years[-1] - years[0])


def semantic_match(resume, jd):
    if MODEL is None:
        return 60

    r = MODEL.encode([resume])
    j = MODEL.encode([jd])

    score = cosine_similarity(r, j)[0][0]

    return round(score * 100, 2)


def ats_format_score(text):
    score = 100

    sections = [
        "experience",
        "education",
        "skills",
        "project"
    ]

    for s in sections:
        if s not in text.lower():
            score -= 10

    if len(text) < 500:
        score -= 20

    return max(score, 0)


def achievement_score(text):
    matches = re.findall(
        r"\d+%|\$\d+|\d+\+|\d+x",
        text
    )

    return min(len(matches) * 5, 100)


def calculate_ats(
    semantic,
    skills,
    format_score,
    achievements
):
    return round(
        semantic * 0.40 +
        skills * 0.30 +
        format_score * 0.20 +
        achievements * 0.10,
        2
    )


def recommendations(
    ats,
    missing,
    exp
):
    tips = []

    if ats < 70:
        tips.append(
            "Resume needs optimization."
        )

    if missing:
        tips.append(
            "Add skills: " +
            ", ".join(missing)
        )

    if exp < 2:
        tips.append(
            "Show internships, projects and achievements."
        )

    if not tips:
        tips.append(
            "Strong resume. Minor refinements only."
        )

    return tips


@app.get("/", response_class=HTMLResponse)
async def home():

    return """
    <html>
    <head>
    <title>AI Resume Analyzer</title>

    <style>
    body{
        font-family:Arial;
        background:#0f172a;
        color:white;
        padding:40px;
    }

    .card{
        background:#1e293b;
        padding:25px;
        border-radius:16px;
        margin-top:20px;
    }

    textarea{
        width:100%;
        height:180px;
    }

    button{
        padding:12px 20px;
        background:#3b82f6;
        color:white;
        border:none;
        border-radius:10px;
    }
    </style>
    </head>

    <body>

    <h1>Professional Resume Analyzer</h1>

    <form action="/analyze"
          enctype="multipart/form-data"
          method="post">

        <div class="card">

        <h3>Upload Resume PDF</h3>

        <input type="file"
               name="resume">

        <h3>Job Description</h3>

        <textarea
        name="jd"></textarea>

        <br><br>

        <button>
        Analyze Resume
        </button>

        </div>

    </form>

    </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    resume: UploadFile = File(...),
    jd: str = Form(...)
):

    data = await resume.read()

    text = extract_pdf_text(data)

    resume_skills = extract_skills(text)
    jd_skills = extract_skills(jd)

    missing = list(
        set(jd_skills) -
        set(resume_skills)
    )

    semantic = semantic_match(text, jd)

    skill_score = 0

    if jd_skills:
        skill_score = (
            len(set(resume_skills)
            & set(jd_skills))
            /
            len(jd_skills)
        ) * 100

    format_score = ats_format_score(text)

    achievements = achievement_score(text)

    ats = calculate_ats(
        semantic,
        skill_score,
        format_score,
        achievements
    )

    exp = estimate_experience(text)

    tips = recommendations(
        ats,
        missing,
        exp
    )

    return f"""
    <html>
    <body style='font-family:Arial;
                 background:#0f172a;
                 color:white;
                 padding:40px;'>

    <h1>Resume Analysis Report</h1>

    <div style='background:#1e293b;
                padding:20px;
                border-radius:15px;'>

    <h2>ATS Score: {ats}%</h2>

    <h3>Semantic Match: {semantic}%</h3>

    <h3>Skill Match: {round(skill_score,2)}%</h3>

    <h3>Formatting: {format_score}%</h3>

    <h3>Achievements: {achievements}%</h3>

    <h3>Experience: {exp} Years</h3>

    <h3>Detected Skills</h3>
    {", ".join(resume_skills)}

    <h3>Missing Skills</h3>
    {", ".join(missing)}

    <h3>Recommendations</h3>
    <ul>
    {''.join([f"<li>{x}</li>" for x in tips])}
    </ul>

    </div>

    </body>
    </html>
    """
