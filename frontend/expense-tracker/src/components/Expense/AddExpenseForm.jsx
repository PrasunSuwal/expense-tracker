import React, { useState } from "react";
import axiosInstance from "../../utils/axiosInstance";
import { API_PATHS } from "../../utils/apiPaths";
import Input from "../Inputs/Input";
import EmojiPickerPopup from "../EmojiPickerPopup";

const AddExpenseForm = ({ onAddExpense }) => {
  const [income, setIncome] = useState({
    category: "",
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
      const res = await axiosInstance.post(
        API_PATHS.EXPENSE.UPLOAD_BILL,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      const { expense, extractedText } = res.data;
      setDetectedCategory(expense.category);
      setExtractedText(extractedText);
      // Emoji mapping by category
      const emojiMap = {
        food: "🍔",
        rent: "🏠",
        utilities: "💡",
        travel: "✈️",
        salary: "💰",
        shopping: "🛍️",
        medical: "🩺",
        entertainment: "🎬",
        other: "📝",
      };
      setIncome((prev) => ({
        ...prev,
        category: expense.category || prev.category,
        amount: expense.amount || prev.amount,
        date: expense.date ? expense.date.slice(0, 10) : prev.date,
        icon: emojiMap[expense.category] || prev.icon,
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
        value={income.category}
        onChange={({ target }) => handleChange("category", target.value)}
        label="Category"
        placeholder="Rent, Groceries, etc."
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
          className="add-btn add-btn-fill"
          type="button"
          onClick={() => onAddExpense(income)}
        >
          Add Expense
        </button>
      </div>
    </div>
  );
};

export default AddExpenseForm;
