const express = require("express");
const {
  addExpense,
  getAllExpense,
  deleteExpense,
  downloadExpenseExcel,
} = require("../controllers/expenseController");
const { protect } = require("../middleware/authMiddleware");

const router = express.Router();
const upload = require("../middleware/uploadMiddleware");
const { uploadAndCategorizeBill } = require("../controllers/expenseController");

router.post("/add", protect, addExpense);
router.get("/get", protect, getAllExpense);
router.get("/downloadexcel", protect, downloadExpenseExcel);
router.delete("/:id", protect, deleteExpense);

// Bill upload and categorization endpoint
router.post("/upload-bill", protect, upload.single("bill"), uploadAndCategorizeBill);

module.exports = router;
