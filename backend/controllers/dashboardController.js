const Income = require("../models/Income");
const Expense = require("../models/Expense");
const { isValidObjectId, Types } = require("mongoose");

//Dashboard Data

exports.getDashboardData = async (req, res) => {
  try {
    const userId = req.user.id;
    const userObjectId = new Types.ObjectId(String(userId));

    //Fetch total income & expenses
    const totalIncome = await Income.aggregate([
      { $match: { userId: userObjectId } },
      { $group: { _id: null, total: { $sum: "$amount" } } },
    ]);

    // console.log("totalIncome", {
    //   totalIncome,
    //   userId: isValidObjectId(userId),
    // });

    const totalExpense = await Expense.aggregate([
      { $match: { userId: userObjectId } },
      { $group: { _id: null, total: { $sum: "$amount" } } },
    ]);

    //GET income transactions in the last 60days

    const last60DaysIncomeTransactions = await Income.find({
      userId,
      date: { $gte: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000) },
    }).sort({ date: -1 });

    //Get total income for last 60 days
    const incomeLast60Days = last60DaysIncomeTransactions.reduce(
      (sum, transaction) => sum + transaction.amount,
      0
    );

    //Get expense transations in the last 30days
    const last30DaysExpenseTransactions = await Expense.find({
      userId,
      date: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
    }).sort({ date: -1 });

    //Get total expenses for last 30 days
    const expenseLast30Days = last30DaysExpenseTransactions.reduce(
      (sum, transaction) => sum + transaction.amount,
      0
    );

    //Fetch last 5transacctions (income+exxpenses)
    const incomeTxns = await Income.find({ userId })
      .sort({ date: -1 })
      .limit(5);
    const expenseTxns = await Expense.find({ userId })
      .sort({ date: -1 })
      .limit(5);

    const lastTransactions = [
      ...incomeTxns.map((txn) => ({
        ...txn.toObject(),
        type: "income", // ✅ correct label
      })),
      ...expenseTxns.map((txn) => ({
        ...txn.toObject(),
        type: "expense", // ✅ correct label
      })),
    ].sort((a, b) => new Date(b.date) - new Date(a.date)); // sort newest first

    //final response
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
      last60DaysIncome: {
        total: parseFloat(incomeLast60Days.toFixed(2)),
        transactions: last60DaysIncomeTransactions,
      },
      recentTransactions: lastTransactions,
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error });
  }
};
