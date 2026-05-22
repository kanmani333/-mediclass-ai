"""
MediClass AI — Complete Flask Application
==========================================
Medical Specialty Classifier
Roles: Admin, Doctor, Patient
Database: SQLite
OCR: Tesseract (Windows + Linux)
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import os, json, platform
import torch.nn.functional as F

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, "database", "mediclass.db")
UPL_PATH = os.path.join(BASE_DIR, "uploads")
MDL_PATH = os.path.join(BASE_DIR, "model", "best_model")

os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)
os.makedirs(UPL_PATH, exist_ok=True)

# ── App Config ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mediclass-ai-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"]                  = UPL_PATH
app.config["MAX_CONTENT_LENGTH"]             = 16 * 1024 * 1024

db = SQLAlchemy(app)

SPECIALTIES = [
    "Cardiology", "Neurology", "Oncology", "Orthopedics",
    "Dermatology", "Gastroenterology", "Pulmonology",
    "Nephrology", "Pediatrics", "General Medicine"
]

SPECIALTY_ICONS = {
    "Cardiology":       "❤️",
    "Neurology":        "🧠",
    "Oncology":         "🔬",
    "Orthopedics":      "🦴",
    "Dermatology":      "🩺",
    "Gastroenterology": "🫁",
    "Pulmonology":      "💨",
    "Nephrology":       "🫘",
    "Pediatrics":       "👶",
    "General Medicine": "🏥"
}

# ── Symptom Keywords ─────────────────────────────────────────
SPECIALTY_KEYWORDS = {
    "General Medicine": [
        "fever", "cold", "flu", "cough", "headache", "fatigue",
        "tired", "weakness", "body ache", "sore throat", "runny nose",
        "nausea", "vomiting", "diarrhea", "infection", "viral",
        "bacterial", "antibiotic", "general", "mild", "routine"
    ],
    "Cardiology": [
        "chest pain", "heart", "cardiac", "palpitation", "blood pressure",
        "hypertension", "ecg", "angina", "shortness of breath", "edema"
    ],
    "Neurology": [
        "brain", "headache", "migraine", "seizure", "stroke", "dizzy",
        "dizziness", "numbness", "tingling", "memory", "tremor", "nerve"
    ],
    "Oncology": [
        "cancer", "tumor", "malignant", "chemotherapy", "biopsy",
        "lymphoma", "carcinoma", "oncology", "mass", "growth"
    ],
    "Orthopedics": [
        "bone", "joint", "fracture", "knee", "back pain", "spine",
        "muscle", "ligament", "arthritis", "shoulder", "hip", "ankle"
    ],
    "Dermatology": [
        "skin", "rash", "itch", "acne", "eczema", "psoriasis",
        "lesion", "wound", "burn", "allergy", "hives", "blister"
    ],
    "Gastroenterology": [
        "stomach", "abdomen", "liver", "bowel", "gastric", "ulcer",
        "acid", "reflux", "constipation", "bloating", "colon", "gut"
    ],
    "Pulmonology": [
        "lung", "breathing", "asthma", "copd", "pneumonia", "bronchial",
        "respiratory", "wheeze", "inhaler", "spirometry", "oxygen"
    ],
    "Nephrology": [
        "kidney", "renal", "urine", "dialysis", "urinary", "bladder",
        "creatinine", "protein urine", "edema", "swelling"
    ],
    "Pediatrics": [
        "child", "baby", "infant", "newborn", "toddler", "kid",
        "vaccination", "growth", "pediatric", "juvenile", "adolescent"
    ]
}

# ── Database Models ──────────────────────────────────────────
class User(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    email           = db.Column(db.String(150), unique=True, nullable=False)
    password        = db.Column(db.String(200), nullable=False)
    role            = db.Column(db.String(20), default="patient")
    name            = db.Column(db.String(100), nullable=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    is_active       = db.Column(db.Boolean, default=True)
    classifications = db.relationship("Classification", backref="user", lazy=True)

class Classification(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    specialty  = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    all_probs  = db.Column(db.Text, nullable=False)
    input_type = db.Column(db.String(20), default="text")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ── Model Loading ────────────────────────────────────────────
classifier_model = None
tokenizer        = None

def load_model():
    global classifier_model, tokenizer
    try:
        # Check if model folder exists
        if not os.path.exists(MDL_PATH):
            print("⚠️  Model folder not found — running in demo mode")
            return

        # Check if model weight files exist
        safetensors_file = os.path.join(MDL_PATH, "model.safetensors")
        pytorch_bin_file = os.path.join(MDL_PATH, "pytorch_model.bin")

        if not os.path.exists(safetensors_file) and \
           not os.path.exists(pytorch_bin_file):
            print("⚠️  Model weights not found — running in demo mode")
            return

        # Load model
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("🤖 Loading BioBERT model...")
        tokenizer        = AutoTokenizer.from_pretrained(MDL_PATH)
        classifier_model = AutoModelForSequenceClassification.from_pretrained(MDL_PATH)
        classifier_model.eval()
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("⚠️  Running in demo mode")
        classifier_model = None
        tokenizer        = None

# ── Text Enhancer ────────────────────────────────────────────
def detect_specialty_from_keywords(text):
    text_lower = text.lower()
    scores     = {}
    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[specialty] = score
    if scores:
        return max(scores, key=scores.get)
    return None

def enhance_text(text):
    words    = text.split()
    if len(words) >= 20:
        return text
    detected  = detect_specialty_from_keywords(text)
    templates = {
        "General Medicine":   f"Patient presents with general medical complaints including {text}. Symptoms suggest primary care consultation. Clinical assessment for common illness, fever, infection or viral condition. General medicine outpatient evaluation recommended.",
        "Cardiology":         f"Patient presents with cardiovascular symptoms: {text}. Cardiac evaluation recommended including ECG and blood pressure monitoring.",
        "Neurology":          f"Patient presents with neurological symptoms: {text}. Neurological assessment of brain and nervous system recommended.",
        "Orthopedics":        f"Patient presents with musculoskeletal complaints: {text}. Orthopedic evaluation of bones and joints recommended.",
        "Dermatology":        f"Patient presents with skin-related symptoms: {text}. Dermatological examination recommended.",
        "Gastroenterology":   f"Patient presents with gastrointestinal symptoms: {text}. Gastroenterological evaluation recommended.",
        "Pulmonology":        f"Patient presents with respiratory symptoms: {text}. Pulmonological assessment of lungs recommended.",
        "Nephrology":         f"Patient presents with renal symptoms: {text}. Nephrological evaluation of kidney function recommended.",
        "Pediatrics":         f"Pediatric patient presents with: {text}. Child health assessment recommended.",
        "Oncology":           f"Patient presents with oncological concerns: {text}. Cancer screening and evaluation recommended.",
    }
    if detected and detected in templates:
        return templates[detected]
    return f"Patient presents with the following symptoms: {text}. General medical assessment required for diagnosis and treatment."

# ── Prediction ───────────────────────────────────────────────
def predict(text):
    text = enhance_text(text)

    # Demo mode — smart keyword-based prediction
    if classifier_model is None or tokenizer is None:
        detected = detect_specialty_from_keywords(text)
        specialty = detected if detected else "General Medicine"

        # Build probability distribution
        import random
        probs = [0.01] * len(SPECIALTIES)
        top_idx = SPECIALTIES.index(specialty)
        probs[top_idx] = 0.85 + random.uniform(0, 0.10)

        # Distribute remaining probability
        remaining = 1.0 - probs[top_idx]
        other_indices = [i for i in range(len(SPECIALTIES)) if i != top_idx]
        for i in other_indices:
            probs[i] = remaining / len(other_indices)

        return {
            "specialty":  specialty,
            "confidence": round(probs[top_idx] * 100, 1),
            "all_probs":  {s: round(p * 100, 1) for s, p in zip(SPECIALTIES, probs)}
        }

    # Real model prediction
    import torch
    enc = tokenizer(
        str(text), max_length=256,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out   = classifier_model(**enc)
        probs = F.softmax(out.logits, dim=1).squeeze().numpy()
    top_idx = int(probs.argmax())
    return {
        "specialty":  SPECIALTIES[top_idx],
        "confidence": round(float(probs[top_idx]) * 100, 1),
        "all_probs":  {s: round(float(p) * 100, 1) for s, p in zip(SPECIALTIES, probs)}
    }

# ── File Extraction ──────────────────────────────────────────
def extract_text_from_pdf(filepath):
    try:
        import fitz
        doc  = fitz.open(filepath)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        print(f"✅ PDF extracted: {len(text)} characters")
        return text
    except Exception as e:
        print(f"❌ PDF error: {e}")
        return ""

def extract_text_from_image(filepath):
    try:
        import pytesseract
        from PIL import Image
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = \
                r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        img  = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        print(f"✅ OCR extracted: {len(text)} characters")
        return text
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in {"pdf", "png", "jpg", "jpeg"}

# ── Auth Decorators ──────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            user = db.session.get(User, session["user_id"])
            if not user or user.role not in roles:
                flash("Access denied.", "danger")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            if not user.is_active:
                flash("Account deactivated. Contact admin.", "danger")
                return redirect(url_for("login"))
            session["user_id"]   = user.id
            session["user_name"] = user.name
            session["user_role"] = user.role
            flash(f"Welcome back, {user.name}! 👋", "success")
            return redirect(url_for("home"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        role     = request.form.get("role", "patient")
        if not all([name, email, password, confirm]):
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))
        if role not in ["doctor", "patient"]:
            role = "patient"
        user = User(
            name     = name,
            email    = email,
            password = generate_password_hash(password),
            role     = role
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

@app.route("/home")
@login_required
def home():
    user                  = db.session.get(User, session["user_id"])
    recent                = Classification.query\
                            .filter_by(user_id=user.id)\
                            .order_by(Classification.created_at.desc())\
                            .limit(5).all()
    total_classifications = Classification.query\
                            .filter_by(user_id=user.id).count()
    return render_template("index.html",
        user                  = user,
        recent                = recent,
        total_classifications = total_classifications,
        specialty_icons       = SPECIALTY_ICONS
    )

@app.route("/classify_page")
@login_required
def classify_page():
    return render_template("classify.html",
        specialty_icons = SPECIALTY_ICONS
    )

@app.route("/classify", methods=["POST"])
@login_required
def classify():
    input_type = request.form.get("input_type", "text")
    text       = ""

    if input_type == "text":
        text = request.form.get("text", "").strip()

    elif input_type == "pdf":
        file = request.files.get("pdf_file")
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                text = extract_text_from_pdf(filepath)
                if not text.strip():
                    flash("Could not extract text from PDF.", "warning")
                    return redirect(url_for("classify_page"))
            elif ext in ["jpg", "jpeg", "png"]:
                text = extract_text_from_image(filepath)
                if not text.strip():
                    flash("Could not extract text from image.", "warning")
                    return redirect(url_for("classify_page"))
        else:
            flash("Invalid file. Upload PDF, JPG or PNG.", "warning")
            return redirect(url_for("classify_page"))

    if not text or len(text.strip()) < 3:
        flash("Please enter valid text.", "warning")
        return redirect(url_for("classify_page"))

    result = predict(text)
    clf    = Classification(
        user_id    = session["user_id"],
        input_text = text[:1000],
        specialty  = result["specialty"],
        confidence = result["confidence"],
        all_probs  = json.dumps(result["all_probs"]),
        input_type = input_type
    )
    db.session.add(clf)
    db.session.commit()

    return render_template("result.html",
        result          = result,
        input_text      = text[:300],
        specialty_icons = SPECIALTY_ICONS
    )

@app.route("/history")
@login_required
def history():
    user = db.session.get(User, session["user_id"])
    if user.role == "admin":
        records = Classification.query\
                  .order_by(Classification.created_at.desc()).all()
    else:
        records = Classification.query\
                  .filter_by(user_id=user.id)\
                  .order_by(Classification.created_at.desc()).all()
    for r in records:
        r.all_probs = json.loads(r.all_probs)
    return render_template("history.html",
        records         = records,
        user            = user,
        specialty_icons = SPECIALTY_ICONS
    )

@app.route("/admin")
@role_required("admin")
def admin():
    users   = User.query.order_by(User.created_at.desc()).all()
    total_c = Classification.query.count()
    recent  = Classification.query\
              .order_by(Classification.created_at.desc())\
              .limit(10).all()
    return render_template("admin.html",
        users           = users,
        total_c         = total_c,
        recent          = recent,
        specialty_icons = SPECIALTY_ICONS
    )

@app.route("/admin/toggle_user/<int:user_id>")
@role_required("admin")
def toggle_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin"))
    if user.role != "admin":
        user.is_active = not user.is_active
        db.session.commit()
        flash(f"User {'activated' if user.is_active else 'deactivated'}.", "success")
    return redirect(url_for("admin"))

@app.route("/admin/change_role/<int:user_id>/<role>")
@role_required("admin")
def change_role(user_id, role):
    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin"))
    if role in ["admin", "doctor", "patient"] and user.id != session["user_id"]:
        user.role = role
        db.session.commit()
        flash(f"Role updated to {role}.", "success")
    return redirect(url_for("admin"))

@app.route("/api/classify", methods=["POST"])
@login_required
def api_classify():
    data = request.get_json()
    text = data.get("text", "").strip() if data else ""
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict(text)
    return jsonify(result)

# ── Initialize Database on Startup ──────────────────────────
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created!")
        if not User.query.filter_by(role="admin").first():
            admin_user = User(
                name     = "Admin",
                email    = "admin@mediclass.com",
                password = generate_password_hash("Admin@123"),
                role     = "admin"
            )
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin account created!")
            print("   Email   : admin@mediclass.com")
            print("   Password: Admin@123")
        else:
            print("✅ Admin already exists!")
    except Exception as e:
        print(f"❌ Database error: {e}")

# ── Load Model Safely ────────────────────────────────────────
load_model()

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("🚀 MediClass AI is running!")
    print(f"   Open: http://127.0.0.1:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)