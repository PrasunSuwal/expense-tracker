const Income = require("../models/Income");
const Expense = require("../models/Expense");
const { isValidObjectId, Types } = require("mongoose");

//Dashboard Data

exports.getDashboardData = async (req, res) => {
  try {
    const userId = req.user.id;
    const userObjectId = new Types.ObjectId(String(userId));
    const { month, year, overall } = req.query;

    // Build date filter if month/year provided and not overall
    let dateFilter = {};
    if (!overall && month && year) {
      const startDate = new Date(year, month - 1, 1);
      const endDate = new Date(year, month, 1);
      dateFilter.date = { $gte: startDate, $lt: endDate };
    }

    // Total Income
    const incomeMatch = { userId: userObjectId, ...dateFilter };
    const totalIncome = await Income.aggregate([
      { $match: incomeMatch },
      { $group: { _id: null, total: { $sum: "$amount" } } },
    ]);

    // Total Expense
    const expenseMatch = { userId: userObjectId, ...dateFilter };
    const totalExpense = await Expense.aggregate([
      { $match: expenseMatch },
      { $group: { _id: null, total: { $sum: "$amount" } } },
    ]);

    // Recent Transactions (last 5)
    const incomeTxns = await Income.find(incomeMatch)
      .sort({ date: -1 })
      .limit(5);
    const expenseTxns = await Expense.find(expenseMatch)
      .sort({ date: -1 })
      .limit(5);
    const lastTransactions = [
      ...incomeTxns.map((txn) => ({
        ...txn.toObject(),
        type: "income",
      })),
      ...expenseTxns.map((txn) => ({
        ...txn.toObject(),
        type: "expense",
      })),
    ].sort((a, b) => new Date(b.date) - new Date(a.date));

    // Last 30 days expenses (filtered by month/year if provided)
    let last30DaysExpenseTransactions = [];
    if (!overall && month && year) {
      last30DaysExpenseTransactions = await Expense.find({
        userId,
        ...dateFilter,
      }).sort({ date: -1 });
    } else {
      last30DaysExpenseTransactions = await Expense.find({
        userId,
        date: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
      }).sort({ date: -1 });
    }
    const expenseLast30Days = last30DaysExpenseTransactions.reduce(
      (sum, transaction) => sum + transaction.amount,
      0
    );

    // Last 30 days income (filtered by month/year if provided)
    let last30DaysIncomeTransactions = [];
    if (!overall && month && year) {
      last30DaysIncomeTransactions = await Income.find({
        userId,
        ...dateFilter,
      }).sort({ date: -1 });
    } else {
      last30DaysIncomeTransactions = await Income.find({
        userId,
        date: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
      }).sort({ date: -1 });
    }
    const incomeLast30Days = last30DaysIncomeTransactions.reduce(
      (sum, transaction) => sum + transaction.amount,
      0
    );

    res.json({
      totalBalance: parseFloat(
        ((totalIncome[0]?.total || 0) - (totalExpense[0]?.total || 0)).toFixed(
          2
        )
      ),
      totalIncome: parseFloat((totalIncome[0]?.total || 0).toFixed(2)),
      totalExpense: parseFloat((totalExpense[0]?.total || 0).toFixed(2)),
      last30DaysExpenses: {
        total: parseFloat(expenseLast30Days.toFixed(2)),
        transactions: last30DaysExpenseTransactions,
      },
      last30DaysIncome: {
        total: parseFloat(incomeLast30Days.toFixed(2)),
        transactions: last30DaysIncomeTransactions,
      },
      recentTransactions: lastTransactions,
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error });
  }
};
