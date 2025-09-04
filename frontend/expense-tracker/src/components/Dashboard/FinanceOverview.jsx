import React from "react";
import CustomPieChart from "../Charts/CustomPieChart";

const COLORS = ["#875CF5", "#FA2C37", "#FF6900"];

const FinanceOverview = ({ totalBalance, totalIncome, totalExpense }) => {
  const balanceData = [
    {
      name: "Total Balance",
      amount: parseFloat(Number(totalBalance).toFixed(2)),
    },
    {
      name: "Total Expenses",
      amount: parseFloat(Number(totalExpense).toFixed(2)),
    },
    {
      name: "Total Income",
      amount: parseFloat(Number(totalIncome).toFixed(2)),
    },
  ];

  // console.log("balanceData", balanceData);

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <h5 className="text-lg">Financial Overview</h5>
      </div>
      <CustomPieChart
        data={balanceData}
        label="Total Balance"
        totalAmount={`Rs${parseFloat(Number(totalBalance).toFixed(2))}`}
        colors={COLORS}
        showTextAnchor
      />
    </div>
  );
};

export default FinanceOverview;
