import React, { useEffect, useState } from "react";
import { LuPlus } from "react-icons/lu";
import CustomBarChart from "../Charts/CustomBarChart";
import { prepareIncomeBarChartData } from "../../utils/helper";

const IncomeOverview = ({ transactions, onAddIncome }) => {
  const [charData, setCharData] = useState([]);
  const totalIncome = transactions.reduce((sum, t) => sum + (t.amount || 0), 0);

  useEffect(() => {
    const result = prepareIncomeBarChartData(transactions);
    setCharData(result);
    return () => {};
  }, [transactions]);
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div className="">
          <h5 className="text-lg">Income Overview</h5>
          <p className="text-xs text-gray-400 mt-0.5">
            Track your earnings over time and analyze your income trends.
          </p>
        </div>
        <button className="add-btn" onClick={onAddIncome}>
          <LuPlus className="text-lg" />
          Add Income
        </button>
      </div>
      <div className="mt-4">
        <span className="font-semibold text-purple-700 text-lg">
          Total Income: Rs{totalIncome}
        </span>
      </div>
      <div className="mt-6">
        {charData.length === 0 ? (
          <div className="text-center text-gray-400 py-10">
            No income data for this month.
          </div>
        ) : (
          <CustomBarChart data={charData} />
        )}
      </div>
    </div>
  );
};

export default IncomeOverview;
