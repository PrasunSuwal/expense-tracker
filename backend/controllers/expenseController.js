const xlsx = require("xlsx");
const Expense = require("../models/Expense");
const Tesseract = require("tesseract.js");
const pdfParse = require("pdf-parse");
const fs = require("fs");

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
      amount,
      date: new Date(date),
    });

    await newExpense.save();
    res.status(200).json(newExpense);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

// Upload and categorize bill
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
    let extractedText = "";

    if (req.file.mimetype === "application/pdf") {
      const dataBuffer = fs.readFileSync(billPath);
      const pdfData = await pdfParse(dataBuffer);
      extractedText = pdfData.text;
    } else if (req.file.mimetype.startsWith("image/")) {
      const result = await Tesseract.recognize(billPath, "eng");
      extractedText = result.data.text;
    } else {
      return res.status(400).json({ message: "Unsupported file type" });
    }

    const category = detectCategory(extractedText);

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

    // Extract amount (prefer 'Total' line, else first currency match)
    let amount = 0;
    let totalMatch = extractedText.match(
      /Total\s*[:\-]?\s*(\$|Rs\.?|INR\.?|USD\.?|EUR\.?|GBP\.?|Amount:?\s?)?(\d+[.,]?\d*)/i
    );
    if (totalMatch && totalMatch[2]) {
      amount = parseFloat(totalMatch[2].replace(/,/g, ""));
    } else {
      const amountMatch = extractedText.match(
        /(?:\$|Rs\.?|INR\.?|USD\.?|EUR\.?|GBP\.?|Amount:?\s?)(\d+[.,]?\d*)/i
      );
      if (amountMatch && amountMatch[1]) {
        amount = parseFloat(amountMatch[1].replace(/,/g, ""));
      }
    }

    // Extract date (first occurrence of date-like pattern)
    let date = new Date();
    const dateMatch = extractedText.match(
      /(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}-\d{2}-\d{2})/
    );
    if (dateMatch && dateMatch[1]) {
      // Try to parse the date string
      const parsedDate = new Date(dateMatch[1]);
      if (!isNaN(parsedDate.getTime())) {
        date = parsedDate;
      }
    }

    // Only analyze and return extracted data, do not save expense
    res.status(200).json({
      message: "Bill analyzed and categorized",
      expense: {
        category,
        amount,
        date,
        icon,
        billUrl,
      },
      extractedText,
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};

// Get all expenses
exports.getAllExpense = async (req, res) => {
  const userId = req.user.id;
  try {
    const expense = await Expense.find({ userId }).sort({ date: -1 });
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
