const mongoose = require("mongoose");

const ExpenseSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    icon: { type: String },
    category: { type: String, required: true }, //eg:food,rent,groceries
    amount: { type: Number, required: true },
    date: { type: Date, default: Date.now },
    billUrl: { type: String },
    category: { type: String },

  },
  { timestamps: true }
);

module.exports = mongoose.model("Expense", ExpenseSchema);
