from flask import Flask, render_template, request
import pdfplumber
import os
from transformers import pipeline
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load zero-shot classifier from Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Possible job roles
job_roles = ["Software Engineer", "Data Analyst", "Project Manager", "Digital Marketer", "Graphic Designer"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['resume']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Extract text
        text = extract_text(filepath)

        # Classify text
        classified_roles = classify_resume(text)

        # Fetch real jobs from Adzuna API
        jobs = []
        for role, score in classified_roles:
            jobs += fetch_jobs_adzuna(role)

        return render_template('result.html', roles=classified_roles, jobs=jobs)
    return "No file uploaded"

def extract_text(filepath):
    text = ""
    if filepath.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
    elif filepath.endswith(".docx"):
        from docx import Document
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text
    else:
        text = "Unsupported file format"
    return text

def classify_resume(text):
    candidate_labels = job_roles
    output = classifier(text, candidate_labels)
    roles = list(zip(output['labels'], output['scores']))
    return roles

def fetch_jobs_adzuna(job_title):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": 3,
        "what": job_title,
        "content-type": "application/json"
    }
    response = requests.get(url, params=params)
    jobs = []
    if response.status_code == 200:
        data = response.json()
        for job in data.get("results", []):
            jobs.append({
                "title": job.get("title"),
                "company": job.get("company", {}).get("display_name"),
                "location": job.get("location", {}).get("display_name"),
                "link": job.get("redirect_url")
            })
    else:
        print("Adzuna API error:", response.status_code, response.text)
    return jobs

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render dynamic port
    app.run(host="0.0.0.0", port=port)
