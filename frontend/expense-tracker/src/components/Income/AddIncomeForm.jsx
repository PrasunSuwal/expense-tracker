import React, { useState } from "react";
import Input from "../Inputs/Input";
import EmojiPickerPopup from "../EmojiPickerPopup";

const AddIncomeForm = ({ onAddIncome }) => {
  const [income, setIncome] = useState({
    source: "",
    amount: "",
    date: "",
    icon: "",
  });
  const [billFile, setBillFile] = useState(null);
  const [detectedCategory, setDetectedCategory] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleChange = (key, value) => setIncome({ ...income, [key]: value });

  const handleBillChange = async (e) => {
    const file = e.target.files[0];
    setBillFile(file);
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("bill", file);
    try {
      const res = await import("../../utils/axiosInstance").then(
        ({ default: axiosInstance }) =>
          axiosInstance.post("/api/v1/income/upload-bill", formData, {
            headers: { "Content-Type": "multipart/form-data" },
          })
      );
      const { income: analyzedIncome, extractedText } = res.data;
      setDetectedCategory(analyzedIncome.category);
      setExtractedText(extractedText);
      // Emoji mapping by category
      const emojiMap = {
        salary: "💰",
        bonus: "🎉",
        investment: "📈",
        freelance: "🧑‍💻",
        gift: "🎁",
        other: "💵",
      };
      setIncome((prev) => ({
        ...prev,
        source: analyzedIncome.category || prev.source,
        amount: analyzedIncome.amount || prev.amount,
        date: analyzedIncome.date
          ? analyzedIncome.date.slice(0, 10)
          : prev.date,
        icon: emojiMap[analyzedIncome.category] || prev.icon,
      }));
    } catch (err) {
      setDetectedCategory("");
      setExtractedText("");
    }
    setUploading(false);
  };

  return (
    <div>
      <EmojiPickerPopup
        icon={income.icon}
        onSelect={(selectedIcon) => handleChange("icon", selectedIcon)}
      />
      <Input
        value={income.source}
        onChange={({ target }) => handleChange("source", target.value)}
        label="Income Source"
        placeholder="Freelance, Salary, etc"
        type="text"
      />
      <Input
        value={income.amount}
        onChange={({ target }) => handleChange("amount", target.value)}
        label="Amount"
        placeholder=""
        type="number"
      />
      <Input
        value={income.date}
        onChange={({ target }) => handleChange("date", target.value)}
        label="Date"
        placeholder=""
        type="date"
      />
      <Input
        label="Upload Bill (Image/PDF)"
        type="file"
        accept="image/*,application/pdf"
        onChange={handleBillChange}
        className=""
        style={{ padding: "8px 12px" }}
      />
      {uploading && (
        <p className="text-xs text-gray-500 mt-2">
          Uploading and analyzing bill...
        </p>
      )}
      {detectedCategory && (
        <div className="mt-2 p-2 bg-slate-100 rounded">
          <strong>Detected Category:</strong> {detectedCategory}
          <br />
          <strong>Extracted Text:</strong>
          <pre className="text-xs whitespace-pre-wrap">{extractedText}</pre>
        </div>
      )}
      <div className="flex justify-end mt-6">
        <button
          className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg"
          onClick={() => onAddIncome(income)}
          type="button"
        >
          Add Income
        </button>
      </div>
    </div>
  );
};

export default AddIncomeForm;
