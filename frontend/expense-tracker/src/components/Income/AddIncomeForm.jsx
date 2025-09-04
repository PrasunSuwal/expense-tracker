import React, { useState } from "react";
import Input from "../Inputs/Input";
import EmojiPickerPopup from "../EmojiPickerPopup";
import { API_PATHS } from "../../utils/apiPaths";
import { ocrAxios } from "../../utils/axiosInstance";

const AddIncomeForm = ({ onAddIncome }) => {
  const [income, setIncome] = useState({
    source: "",
    amount: "",
    date: "",
    icon: "",
  });
  const [, setBillFile] = useState(null);
  const [detectedCategory, setDetectedCategory] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [feedbackCategory, setFeedbackCategory] = useState("");
  const [feedbackAmount, setFeedbackAmount] = useState("");
  const [feedbackNote, setFeedbackNote] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

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
      const payload = res.data || {};
      const inc = payload.income || {};
      const cat = inc.category || payload.category || "";
      const amt = inc.amount ?? payload.amount ?? "";
      const dt = inc.date || payload.date || "";
      const ic = inc.icon || payload.icon || "";
      const text = payload.extractedText || "";

      setDetectedCategory(cat);
      setExtractedText(text);

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
        source: cat || prev.source,
        amount:
          amt !== undefined && amt !== null && amt !== "" ? amt : prev.amount,
        date: dt ? String(dt).slice(0, 10) : prev.date,
        icon: emojiMap[String(cat).toLowerCase()] || ic || prev.icon || "💵",
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
          feedbackCategory || detectedCategory || income.source || "",
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
        icon={income.icon || "💵"}
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
