const xlsx = require("xlsx");
const Expense = require("../models/Expense");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

// Helper: Detect category from text
const detectCategory = (text) => {
  const keywords = {
    food: ["food", "restaurant", "meal", "grocery", "groceries"],
    rent: ["rent", "lease", "apartment", "housing"],
    utilities: ["electricity", "water", "gas", "utility", "utilities"],
    travel: ["travel", "flight", "train", "bus", "taxi"],
    salary: ["salary", "payroll", "income", "wages"],
    shopping: ["shopping", "clothes", "apparel", "store"],
    medical: ["medical", "doctor", "hospital", "medicine"],
    entertainment: ["movie", "cinema", "entertainment", "concert"],
    other: [],
  };

  const lowerText = text ? text.toLowerCase() : "";
  for (let category in keywords) {
    for (let word of keywords[category]) {
      if (lowerText.includes(word)) {
        return category;
      }
    }
  }
  return "other";
};

// Add Expense
exports.addExpense = async (req, res) => {
  const userId = req.user.id;
  try {
    const { icon, category, amount, date } = req.body;

    if (!category || !amount || !date) {
      return res.status(401).json({ message: "All fields are required" });
    }

    const newExpense = new Expense({
      userId,
      icon,
      category,
      amount: parseFloat(Number(amount).toFixed(2)),
      date: new Date(date),
    });

    await newExpense.save();
    res.status(200).json(newExpense);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

// Upload and categorize bill (via FastAPI OCR service)
exports.uploadAndCategorizeBill = async (req, res) => {
  const userId = req.user.id;
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No bill file uploaded" });
    }

    const billPath = req.file.path;
    const billUrl = `${req.protocol}://${req.get("host")}/uploads/${
      req.file.filename
    }`;
    // Forward the uploaded file to OCR microservice
    const buffer = fs.readFileSync(billPath);
    const form = new FormData();
    form.append("file", buffer, {
      filename: req.file.originalname || req.file.filename,
    });

    const OCR_API = process.env.OCR_API_URL || "http://localhost:8001";
    const ocrResponse = await axios.post(`${OCR_API}/process`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
    });

    const extractedText = ocrResponse.data?.raw_text || "";
    const detectedAmount = Number(ocrResponse.data?.amount) || 0;
    const detectedCategory = (
      ocrResponse.data?.category || "other"
    ).toLowerCase();
    const category =
      detectedCategory !== "miscellaneous"
        ? detectedCategory
        : detectCategory(extractedText);

    // Emoji mapping for categories
    const categoryEmojis = {
      food: "🍲",
      rent: "🏠",
      utilities: "💡",
      travel: "✈️",
      salary: "💰",
      shopping: "🛍️",
      medical: "🩺",
      entertainment: "🎬",
      other: "💸",
    };
    const icon = categoryEmojis[category] || "💸";

    // Extract amount: trust OCR value; only fallback if not present
    let amount = detectedAmount;
    if (!amount || Number.isNaN(amount)) {
      // Prefer strong total phrases; avoid matching inside 'subtotal'
      const strongTotal = extractedText.match(
        /(grand\s+total|amount\s+due|balance\s+due|total\s+due)\s*[:\-]?\s*(\$|Rs\.?|INR\.?|USD\.?|EUR\.?|GBP\.?\s*)?(\d+[.,]?\d*)/i
      );
      if (strongTotal && strongTotal[3]) {
        amount = parseFloat(strongTotal[3].replace(/,/g, ""));
      } else {
        const strictTotal = extractedText.match(
          /\btotal\b\s*[:\-]?\s*(\$|Rs\.?|INR\.?|USD\.?|EUR\.?|GBP\.?\s*)?(\d+[.,]?\d*)/i
        );
        if (strictTotal && strictTotal[2]) {
          amount = parseFloat(strictTotal[2].replace(/,/g, ""));
        } else {
          const generic = extractedText.match(
            /(?:\$|Rs\.?|INR\.?|USD\.?|EUR\.?|GBP\.?|Amount:?\s?)(\d+[.,]?\d*)/i
          );
          if (generic && generic[1]) {
            amount = parseFloat(generic[1].replace(/,/g, ""));
          }
        }
      }
    }

    // Always use current date for bill upload
    const date = new Date();

    // Only analyze and return extracted data, do not save expense
    // Round amount to 2 decimal places
    const roundedAmount = amount ? parseFloat(Number(amount).toFixed(2)) : null;

    // Backward-compatible shape: top-level category/amount for UI auto-fill
    res.status(200).json({
      message: "Bill analyzed and categorized",
      category,
      amount: roundedAmount,
      date,
      icon,
      billUrl,
      extractedText,
      expense: {
        category,
        amount: roundedAmount,
        date,
        icon,
        billUrl,
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};

// Get all expenses
exports.getAllExpense = async (req, res) => {
  const userId = req.user.id;
  const { month, year } = req.query;
  let filter = { userId };
  if (month && year) {
    // month: 1-12, year: YYYY
    const startDate = new Date(year, month - 1, 1);
    const endDate = new Date(year, month, 1);
    filter.date = { $gte: startDate, $lt: endDate };
  }
  try {
    const expense = await Expense.find(filter).sort({ date: -1 });
    res.json(expense);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

// Delete an expense
exports.deleteExpense = async (req, res) => {
  try {
    await Expense.findByIdAndDelete(req.params.id);
    res.json({ message: "Expense deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

// Download Excel of expenses
exports.downloadExpenseExcel = async (req, res) => {
  const userId = req.user.id;
  try {
    const expense = await Expense.find({ userId }).sort({ date: -1 });

    const data = expense.map((item) => ({
      category: item.category,
      Amount: item.amount,
      Date: item.date,
    }));

    const wb = xlsx.utils.book_new();
    const ws = xlsx.utils.json_to_sheet(data);
    xlsx.utils.book_append_sheet(wb, ws, "Expenses");
    const filePath = "expense_details.xlsx";
    xlsx.writeFile(wb, filePath);
    res.download(filePath);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};
