import React, { useState } from "react";
import { useEffect } from "react";
import { LuPlus } from "react-icons/lu";
import { prepareExpenseLineChartData } from "../../utils/helper";
import CustomLineChart from "../Charts/CustomLineChart";

const ExpenseOverview = ({ transactions, onExpenseIncome }) => {
  const [charData, setCharData] = useState([]);
  const totalExpense = transactions.reduce(
    (sum, t) => sum + (t.amount || 0),
    0
  );
  useEffect(() => {
    const result = prepareExpenseLineChartData(transactions);
    setCharData(result);
    return () => {};
  }, [transactions]);

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div className="">
          <h5 className="text-lg">Expense Overview</h5>
          <p className="text-xs text-gray-400 mt-0.5">
            Track your spendings trends over time and gain insights into where
            your money goes.
          </p>
        </div>
        <button className="add-btn" onClick={onExpenseIncome}>
          <LuPlus className="text-lg" />
          Add Expense
        </button>
      </div>
      <div className="mt-4">
        <span className="font-semibold text-purple-700 text-lg">
          Total Expenses: Rs{totalExpense}
        </span>
      </div>
      <div className="mt-6">
        {charData.length === 0 ? (
          <div className="text-center text-gray-400 py-10">
            No expense data for this month.
          </div>
        ) : (
          <CustomLineChart data={charData} />
        )}
      </div>
    </div>
  );
};

export default ExpenseOverview;
