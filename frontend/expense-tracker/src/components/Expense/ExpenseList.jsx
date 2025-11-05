import moment from "moment";
import React from "react";
import { LuDownload } from "react-icons/lu";
import TransactionInfoCard from "../Cards/TransactionInfoCard";

const ExpenseList = ({ transactions, onDelete, onEdit, onDownload }) => {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <h5 className="text-lg">All Expenses</h5>
        <button className="card-btn" onClick={onDownload}>
          <LuDownload className="text-base" />
          Download
        </button>
      </div>
      <div className="grid gris-cols-1 md:grid-cols-2">
        {transactions?.map((expense) => (
          <TransactionInfoCard
            key={expense._id}
            title={expense.category}
            icon={expense.icon}
            date={moment(expense.date).format("Do MMM YYYY")}
            amount={expense.amount}
            type="expense"
            notes={expense.notes}
            onDelete={() => onDelete(expense._id)}
            onEdit={() => onEdit(expense)}
          />
        ))}
      </div>
    </div>
  );
};

export default ExpenseList;
