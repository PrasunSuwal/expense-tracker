const express = require("express");
const {
  addIncome,
  getAllIncome,
  deleteIncome,
  downloadIncomeExcel,
  uploadAndCategorizeIncomeBill,
} = require("../controllers/incomeController");
const { protect } = require("../middleware/authMiddleware");

const router = express.Router();
const uploadMiddleware = require("../middleware/uploadMiddleware");

router.post("/add", protect, addIncome);
router.get("/get", protect, getAllIncome);
router.get("/downloadexcel", protect, downloadIncomeExcel);
router.delete("/:id", protect, deleteIncome);

// Bill upload and categorization endpoint for income
// Bill upload and categorization endpoint for income
router.post(
  "/upload-bill",
  protect,
  uploadMiddleware.single("bill"),
  uploadAndCategorizeIncomeBill
);

module.exports = router;
