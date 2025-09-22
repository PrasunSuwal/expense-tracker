const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

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

// Upload and categorize income bill (via FastAPI OCR service)
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
    // Forward file to OCR microservice
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
        : detectIncomeCategory(extractedText);
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
    // Extract amount: trust OCR value first; fallback to stricter patterns if missing
    let amount = detectedAmount;
    if (!amount || Number.isNaN(amount)) {
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
    // Only analyze and return extracted data, do not save income
    // Round amount to 2 decimal places
    const roundedAmount = amount ? parseFloat(Number(amount).toFixed(2)) : null;

    // Backward-compatible shape with top-level fields
    res.status(200).json({
      message: "Income bill analyzed and categorized",
      category,
      amount: roundedAmount,
      date,
      icon,
      billUrl,
      extractedText,
      income: {
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
      amount: parseFloat(Number(amount).toFixed(2)),
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
  const { month, year } = req.query;
  let filter = { userId };
  if (month && year) {
    // month: 1-12, year: YYYY
    const startDate = new Date(year, month - 1, 1);
    const endDate = new Date(year, month, 1);
    filter.date = { $gte: startDate, $lt: endDate };
  }
  try {
    const income = await Income.find(filter).sort({ date: -1 });
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
