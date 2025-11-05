import React, { useState, useEffect } from "react";
import axiosInstance, { ocrAxios } from "../../utils/axiosInstance";
import { API_PATHS } from "../../utils/apiPaths";
import Input from "../Inputs/Input";
import EmojiPickerPopup from "../EmojiPickerPopup";

const AddExpenseForm = ({ onAddExpense, editingExpense, onUpdateExpense }) => {
  const [expense, setExpense] = useState({
    category: "",
    amount: "",
    date: "",
    icon: "",
    notes: "",
  });

  // Populate form when editing
  useEffect(() => {
    if (editingExpense) {
      setExpense({
        category: editingExpense.category || "",
        amount: editingExpense.amount || "",
        date: editingExpense.date ? editingExpense.date.slice(0, 10) : "",
        icon: editingExpense.icon || "",
        notes: editingExpense.notes || "",
      });
    }
  }, [editingExpense]);

  const [, setBillFile] = useState(null);
  const [detectedCategory, setDetectedCategory] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [feedbackCategory, setFeedbackCategory] = useState("");
  const [feedbackAmount, setFeedbackAmount] = useState("");
  const [feedbackNote, setFeedbackNote] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

  const handleChange = (key, value) => setExpense({ ...expense, [key]: value });

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
      const payload = res.data || {};
      const exp = payload.expense || {};
      const cat = exp.category || payload.category || "";
      const amt = exp.amount ?? payload.amount ?? "";
      const dt = exp.date || payload.date || "";
      const ic = exp.icon || payload.icon || "";
      const text = payload.extractedText || "";

      setDetectedCategory(cat);
      setExtractedText(text);

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

      setExpense((prev) => ({
        ...prev,
        category: cat || prev.category,
        amount:
          amt !== undefined && amt !== null && amt !== "" ? amt : prev.amount,
        date: dt ? String(dt).slice(0, 10) : prev.date,
        icon: emojiMap[String(cat).toLowerCase()] || ic || prev.icon || "💸",
      }));
    } catch (err) {
      console.error("Error uploading bill:", err);
      setDetectedCategory("");
      setExtractedText("");
    }
    setUploading(false);
  };

  const handleSubmitFeedback = async () => {
    if (!extractedText || (!feedbackCategory && !feedbackAmount)) return;
    setSubmittingFeedback(true);
    try {
      await ocrAxios.post(API_PATHS.OCR.FEEDBACK, {
        raw_text: extractedText,
        correct_category:
          feedbackCategory || detectedCategory || expense.category || "",
        amount: feedbackAmount ? Number(feedbackAmount) : undefined,
      });
      setFeedbackNote("Thanks! Your feedback was saved.");
      setTimeout(() => setFeedbackNote(""), 2500);
    } catch (e) {
      console.error("Error submitting feedback:", e);
      setFeedbackNote("Could not save feedback. Try again later.");
      setTimeout(() => setFeedbackNote(""), 2500);
    }
    setSubmittingFeedback(false);
  };

  return (
    <div>
      <EmojiPickerPopup
        icon={expense.icon || "💸"}
        onSelect={(selectedIcon) => handleChange("icon", selectedIcon)}
      />
      <Input
        value={expense.category}
        onChange={({ target }) => handleChange("category", target.value)}
        label="Category"
        placeholder="Rent, Groceries, etc."
        type="text"
      />
      <Input
        value={expense.amount}
        onChange={({ target }) => handleChange("amount", target.value)}
        label="Amount"
        placeholder=""
        type="number"
      />
      <Input
        value={expense.date}
        onChange={({ target }) => handleChange("date", target.value)}
        label="Date"
        placeholder=""
        type="date"
      />
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Additional Notes
        </label>
        <textarea
          value={expense.notes}
          onChange={({ target }) => handleChange("notes", target.value)}
          placeholder="Add any additional details about this expense..."
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-vertical min-h-[80px]"
          rows={3}
        />
      </div>
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
          <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2">
            <input
              type="text"
              className="border rounded px-2 py-1"
              placeholder="Correct category (optional)"
              value={feedbackCategory}
              onChange={(e) => setFeedbackCategory(e.target.value)}
            />
            <input
              type="number"
              className="border rounded px-2 py-1"
              placeholder="Correct amount (optional)"
              value={feedbackAmount}
              onChange={(e) => setFeedbackAmount(e.target.value)}
            />
            <button
              type="button"
              disabled={submittingFeedback}
              onClick={handleSubmitFeedback}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white px-3 py-1 rounded"
            >
              {submittingFeedback ? "Submitting..." : "Send OCR Feedback"}
            </button>
          </div>
          {feedbackNote && (
            <p className="text-xs mt-1 text-gray-600">{feedbackNote}</p>
          )}
        </div>
      )}
      <div className="flex justify-end mt-6">
        <button
          className="add-btn add-btn-fill"
          type="button"
          onClick={() =>
            editingExpense ? onUpdateExpense(expense) : onAddExpense(expense)
          }
        >
          {editingExpense ? "Update Expense" : "Add Expense"}
        </button>
      </div>
    </div>
  );
};

export default AddExpenseForm;
