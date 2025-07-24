const Tesseract = require("tesseract.js");
const pdfParse = require("pdf-parse");
const fs = require("fs");

// Helper: Detect income category from text
const detectIncomeCategory = (text) => {
  const keywords = {
    salary: ["salary", "payroll", "income", "wages", "monthly pay"],
    bonus: ["bonus", "incentive", "reward"],
    investment: ["investment", "dividend", "interest", "stock", "mutual fund"],
    freelance: ["freelance", "contract", "gig", "project"],
    gift: ["gift", "present", "donation"],
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

// Upload and categorize income bill
exports.uploadAndCategorizeIncomeBill = async (req, res) => {
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
    const category = detectIncomeCategory(extractedText);
    // Emoji mapping for income categories
    const categoryEmojis = {
      salary: "💰",
      bonus: "🎉",
      investment: "📈",
      freelance: "🧑‍💻",
      gift: "🎁",
      other: "💵",
    };
    const icon = categoryEmojis[category] || "💵";
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
      const parsedDate = new Date(dateMatch[1]);
      if (!isNaN(parsedDate.getTime())) {
        date = parsedDate;
      }
    }
    // Only analyze and return extracted data, do not save income
    res.status(200).json({
      message: "Income bill analyzed and categorized",
      income: {
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
const xlsx = require("xlsx");
const Income = require("../models/Income");

//Add Income Source
exports.addIncome = async (req, res) => {
  const userId = req.user.id;
  try {
    const { icon, source, amount, date } = req.body;

    //validate for missing fields
    if (!source || !amount || !date) {
      return res.status(401).json({ message: "All fields are required" });
    }
    const newIncome = new Income({
      userId,
      icon,
      source,
      amount,
      date: new Date(date),
    });
    await newIncome.save();
    res.status(200).json(newIncome);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

//get all Income Source
exports.getAllIncome = async (req, res) => {
  const userId = req.user.id;
  try {
    const income = await Income.find({ userId }).sort({ date: -1 });
    res.json(income);
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

//Delete Income Source
exports.deleteIncome = async (req, res) => {
  const userId = req.user.id;
  try {
    await Income.findByIdAndDelete(req.params.id);
    res.json({ message: "Income deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};

//Download excel
exports.downloadIncomeExcel = async (req, res) => {
  const userId = req.user.id;
  try {
    const income = await Income.find({ userId }).sort({ date: -1 });

    //Prepare data for Excel
    const data = income.map((item) => ({
      Source: item.source,
      Amount: item.amount,
      Date: item.date,
    }));
    const wb = xlsx.utils.book_new();
    const ws = xlsx.utils.json_to_sheet(data);
    xlsx.utils.book_append_sheet(wb, ws, "Income");
    xlsx.writeFile(wb, "income_details.xlsx");
    res.download("income_details.xlsx");
  } catch (error) {
    res.status(500).json({ message: "Server Error" });
  }
};
