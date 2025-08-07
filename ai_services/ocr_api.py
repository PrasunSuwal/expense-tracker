from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber, re, os, csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pytesseract

app = Flask(__name__)
CORS(app)

# -----------------------------
# Training Data for ML
# -----------------------------
training_data = [
    ("KFC Durbar Marg bill Rs 1200", "Food"),
    ("Pizza Hut Thamel receipt Rs 2500", "Food"),
    ("Daraz online shopping Rs 5200", "Shopping"),
    ("Amazon online order Rs 4500", "Shopping"),
    ("Hotel Everest stay Rs 12000", "Travel"),
    ("Taxi ride Tribhuvan Airport Rs 700", "Travel"),
    ("Nepal Electricity Authority bill Rs 3200", "Utilities"),
    ("Worldlink Internet monthly Rs 2100", "Utilities"),
]

texts = [t for t, l in training_data]
labels = [l for t, l in training_data]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=200)
model.fit(X_train, labels)

# -----------------------------
# Keyword Categories
# -----------------------------
category_keywords = {
    "Food": ["restaurant", "cafe", "coffee", "pizza", "burger", "bakery", "lunch", "dinner"],
    "Shopping": ["daraz", "bhatbhateni", "mall", "store", "amazon", "shopping", "clothing",
                 "furniture", "table", "marble", "decor", "gift", "hardware", "locks", "electronics"],
    "Travel": ["hotel", "flight", "bus", "taxi", "airlines", "pathao", "uber", "tour", "booking", "stay", "ticket"],
    "Utilities": ["electricity", "water", "internet", "recharge", "gas", "dishhome", "ntc", "bill", "mobile"],
}

# -----------------------------
# Amount Extraction Helpers
# -----------------------------
def normalize_amount_str(amount_str):
    s = amount_str.strip()
    s = re.sub(r'[^\d,.\s]', '', s)
    s = s.replace(" ", "")

    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")   # US style 12,345.67
        else:
            s = s.replace(".", "").replace(",", ".")  # EU style 12.345,67
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def extract_amount(text):
    candidates = []
    for line in text.splitlines():
        if any(k in line.lower() for k in ["total", "gross worth", "amount due", "summary", "grand total"]):
            matches = re.findall(r'[\$₹Rs\. ]?\s?[\d\s,.]+(?:[,.]\d{1,2})?', line)
            for m in matches:
                val = normalize_amount_str(m)
                if val is not None:
                    candidates.append(val)

    if candidates:
        return max(candidates)

    all_numbers = re.findall(r'[\$₹Rs\. ]?\s?[\d\s,.]+(?:[,.]\d{1,2})?', text)
    numeric_vals = [normalize_amount_str(n) for n in all_numbers if normalize_amount_str(n) is not None]

    return max(numeric_vals) if numeric_vals else None

# -----------------------------
# Text Extraction (PDF or Image)
# -----------------------------
def extract_text_from_file(file):
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image)
        return text

    else:
        raise ValueError("Unsupported file type. Please upload PDF or Image.")

# -----------------------------
# Feedback System CSV
# -----------------------------
FEEDBACK_FILE = "invoice_feedback.csv"

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_text", "correct_category", "amount"])

# -----------------------------
# Process Invoice Endpoint
# -----------------------------
@app.route("/process", methods=["POST"])
def process_invoice():
    try:
        file = request.files["file"]
        text = extract_text_from_file(file)

        print("=== Extracted Text ===")
        print(text)

        amount = extract_amount(text)
        text_lower = text.lower()

        # Step 1: Keyword Override
        category = None
        for cat, words in category_keywords.items():
            if any(w in text_lower for w in words):
                category = cat
                break

        # Step 2: ML Fallback
        if not category:
            vec = vectorizer.transform([text])
            proba = model.predict_proba(vec)[0]
            best_idx = proba.argmax()
            best_confidence = proba[best_idx]
            best_category = model.classes_[best_idx]

            if best_confidence >= 0.7:
                category = best_category
            else:
                category = "Miscellaneous"

        return jsonify({
            "raw_text": text,
            "amount": amount,
            "category": category
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Feedback Endpoint
# -----------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        raw_text = data.get("raw_text")
        correct_category = data.get("correct_category")
        amount = data.get("amount")

        if not raw_text or not correct_category:
            return jsonify({"error": "Missing raw_text or correct_category"}), 400

        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([raw_text, correct_category, amount])

        return jsonify({"message": "Feedback saved successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Optional: configure tesseract path if not auto-detected
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    app.run(port=8000)
