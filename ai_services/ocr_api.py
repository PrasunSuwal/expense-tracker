from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber, re, os, csv, io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pytesseract

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    "Medical": ["medical", "hospital", "doctor", "medication", "medicine", "pharmacy", "pharmaceuticals", "health", "medical insurance"],
    "Entertainment": ["movie", "cinema", "entertainment", "concert"],
    "Other": ["other", "miscellaneous", "misc", "other expenses", "other income"],
    "Salary": ["salary", "payroll", "income", "wages", "monthly pay"],
    "Bonus": ["bonus", "incentive", "reward"],
    "Investment": ["investment", "dividend", "interest", "stock", "mutual fund"],
    "Freelance": ["freelance", "contract", "gig", "project"],
    "Gift": ["gift", "present", "donation"],
    "Financial": ["financial", "bank", "loan", "credit", "debit", "credit card", "debit card", "bank statement", "bank account"],
    "Education": ["education", "school", "college", "university", "tuition", "fees", "scholarship", "study", "learning"],
 
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
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lower_lines = [l.lower() for l in lines]

    def nums_from_line(line):
        matches = list(re.finditer(r'[\$₹rs\. ]?\s?[\d\s,.]+(?:[,.]\d{1,2})?', line, flags=re.IGNORECASE))
        vals = []
        for m in matches:
            v = normalize_amount_str(m.group())
            if v is not None:
                vals.append((v, m.start(), m.end()))
        return vals

    # 1) Strong signals for final payable total (exclude subtotal/discount/shipping)
    strong_keywords = [
        "grand total", "total due", "amount due", "balance due", "total payable", "amount payable","amount"
    ]
    for i in range(len(lower_lines) - 1, -1, -1):
        ll = lower_lines[i]
        if any(k in ll for k in strong_keywords):
            vals = nums_from_line(lines[i])
            if vals:
                # Prefer the last number on the line, or the number appearing after the keyword
                # Determine the rightmost keyword position
                kw_pos = max((ll.rfind(k) for k in strong_keywords if k in ll), default=-1)
                after_kw = [v for v in vals if v[1] >= kw_pos]
                chosen = after_kw[-1][0] if after_kw else vals[-1][0]
                return chosen

    # 2) Generic 'total' but not 'subtotal'/'discount'/'shipping' (scan bottom-up)
    for i in range(len(lower_lines) - 1, -1, -1):
        ll = lower_lines[i]
        if "total" in ll and not any(bad in ll for bad in ["subtotal", "sub total", "discount", "shipping", "tax"]):
            vals = nums_from_line(lines[i])
            if vals:
                kw_pos = ll.rfind("total")
                after_kw = [v for v in vals if v[1] >= kw_pos]
                chosen = after_kw[-1][0] if after_kw else vals[-1][0]
                return chosen

    # 3) Derive from subtotal - discount + shipping if available
    def find_first(regex_list):
        for i, ll in enumerate(lower_lines):
            if any(rgx in ll for rgx in regex_list):
                vals = nums_from_line(lines[i])
                if vals:
                    return max(vals)
        return None

    subtotal = find_first(["subtotal", "sub total"])
    discount = find_first(["discount"])
    shipping = find_first(["shipping", "delivery"])
    if subtotal is not None:
        computed = subtotal - (discount or 0) + (shipping or 0)
        if computed > 0:
            return round(computed, 2)

    # 4) Fallback: choose the largest number near the bottom third of the doc
    cut = max(0, int(len(lines) * 0.66))
    bottom_text = "\n".join(lines[cut:])
    all_numbers = re.findall(r'[\$₹rs\. ]?\s?[\d\s,.]+(?:[,.]\d{1,2})?', bottom_text, flags=re.IGNORECASE)
    numeric_vals = [normalize_amount_str(n) for n in all_numbers if normalize_amount_str(n) is not None]
    if numeric_vals:
        return max(numeric_vals)

    # 5) Last resort: max anywhere
    all_numbers = re.findall(r'[\$₹rs\. ]?\s?[\d\s,.]+(?:[,.]\d{1,2})?', text, flags=re.IGNORECASE)
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
@app.post("/process")
async def process_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = extract_text_from_upload(file.filename, contents)

        amount = extract_amount(text)
        text_lower = text.lower()

        category = None
        for cat, words in category_keywords.items():
            if any(w in text_lower for w in words):
                category = cat
                break

        if not category:
            vec = vectorizer.transform([text])
            proba = model.predict_proba(vec)[0]
            best_idx = proba.argmax()
            best_confidence = proba[best_idx]
            best_category = model.classes_[best_idx]

            category = best_category if best_confidence >= 0.7 else "Miscellaneous"

        return {
            "raw_text": text,
            "amount": amount,
            "category": category
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Feedback Endpoint
# -----------------------------
class FeedbackBody(BaseModel):
    raw_text: str
    correct_category: str
    amount: float | None = None

@app.post("/feedback")
async def feedback(body: FeedbackBody):
    try:
        if not body.raw_text or not body.correct_category:
            raise HTTPException(status_code=400, detail="Missing raw_text or correct_category")

        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([body.raw_text, body.correct_category, body.amount])

        return {"message": "Feedback saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------
# Helpers (FastAPI upload)
# ---------------
def extract_text_from_upload(filename: str, contents: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    elif name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(contents))
        return pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or Image.")
