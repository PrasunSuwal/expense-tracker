import React, { useEffect, useState } from "react";
import { useUserAuth } from "../../hooks/useUserAuth";
import { API_PATHS } from "../../utils/apiPaths";
import axiosInstance from "../../utils/axiosInstance";
import toast from "react-hot-toast";
import ExpenseOverview from "../../components/Expense/ExpenseOverview";
import Modal from "../../components/Modal";
import AddExpenseForm from "../../components/Expense/AddExpenseForm";
import ExpenseList from "../../components/Expense/ExpenseList";
import DeleteAlert from "../../components/DeleteAlert";
import DashboardLayout from "../../components/layouts/DashboardLayout";

const Expense = () => {
  useUserAuth();
  const [expenseData, setExpenseData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDeleteAlert, setOpenDeleteAlert] = useState({
    show: false,
    data: null,
  });
  const [openAddExpenseModal, setOpenAddExpenseModal] = useState(false);
  // Month/year selection
  const now = new Date();
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth() + 1); // 1-12
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());

  //Get all Expense Details
  const fetchExpenseDetails = async () => {
    if (loading) return;
    setLoading(true);
    try {
      const response = await axiosInstance.get(
        `${API_PATHS.EXPENSE.GET_ALL_EXPENSE}?month=${selectedMonth}&year=${selectedYear}`
      );
      if (response.data) {
        setExpenseData(response.data);
      }
    } catch (error) {
      console.log("Something went wrong.Please try againn", error);
    } finally {
      setLoading(false);
    }
  };

  //Handle All Expense
  const handleAddExpense = async (expense) => {
    const { category, amount, date, icon } = expense;

    //Validation checks
    if (!category.trim()) {
      toast.error("Category is required.");
    }
    if (!amount || isNaN(amount) || Number(amount) <= 0) {
      toast.error("Amount is required and should be a positive number.");
      return;
    }
    if (!date) {
      toast.error("Date is required.");
      return;
    }
    try {
      await axiosInstance.post(API_PATHS.EXPENSE.ADD_EXPENSE, {
        category,
        amount,
        date,
        icon,
      });
      setOpenAddExpenseModal(false);
      toast.success("Expense added successfully.");
      fetchExpenseDetails();
    } catch (error) {
      console.error(
        "Error adding expense",
        error.response?.data?.message || error.message
      );
    }
  };

  //Delete Expense
  const deleteExpense = async (id) => {
    try {
      await axiosInstance.delete(API_PATHS.EXPENSE.DELETE_EXPENSE(id));
      setOpenDeleteAlert({ show: false, data: null });
      toast.success("Expense deleted successfully.");
      fetchExpenseDetails();
    } catch (error) {
      console.error(
        "Error deleting expense",
        error.response?.data?.message || error.message
      );
    }
  };

  //handle download expense details
  const handleDownloadExpenseDetails = async () => {
    try {
      const response = await axiosInstance.get(
        `${API_PATHS.EXPENSE.DOWNLOAD_EXPENSE}`,
        {
          responseType: "blob",
        }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "expense-details.xlsx");
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading expense details", error);
      toast.error("Error downloading expense details");
    }
  };

  useEffect(() => {
    fetchExpenseDetails();
    return () => {};
  }, [selectedMonth, selectedYear]);

  return (
    <DashboardLayout activeMenu="Expense">
      <div className="my-5 mx-auto">
        {/* Month/year selector */}
        <div className="flex gap-2 mb-4 items-center">
          <button
            className="px-2 py-1 rounded bg-gray-200"
            onClick={() =>
              setSelectedMonth(selectedMonth === 1 ? 12 : selectedMonth - 1)
            }
            aria-label="Previous Month"
          >
            &#8592;
          </button>
          {[...Array(12)].map((_, i) => (
            <button
              key={i + 1}
              className={`px-3 py-1 rounded ${
                selectedMonth === i + 1
                  ? "bg-purple-600 text-white"
                  : "bg-gray-200"
              }`}
              onClick={() => setSelectedMonth(i + 1)}
            >
              {new Date(0, i).toLocaleString("default", { month: "short" })}
            </button>
          ))}
          <button
            className="px-2 py-1 rounded bg-gray-200"
            onClick={() =>
              setSelectedMonth(selectedMonth === 12 ? 1 : selectedMonth + 1)
            }
            aria-label="Next Month"
          >
            &#8594;
          </button>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="ml-2 px-2 py-1 rounded border"
          >
            {[...Array(5)].map((_, i) => (
              <option key={i} value={now.getFullYear() - i}>
                {now.getFullYear() - i}
              </option>
            ))}
          </select>
        </div>
        {loading ? (
          <div className="flex justify-center items-center py-20">
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-purple-600"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            <div className="">
              <ExpenseOverview
                transactions={expenseData}
                onExpenseIncome={() => setOpenAddExpenseModal(true)}
              />
            </div>
            <ExpenseList
              transactions={expenseData}
              onDelete={(id) => setOpenDeleteAlert({ show: true, data: id })}
              onDownload={handleDownloadExpenseDetails}
            />
          </div>
        )}
        <Modal
          isOpen={openAddExpenseModal}
          onClose={() => setOpenAddExpenseModal(false)}
          title="Add Expense"
        >
          <AddExpenseForm onAddExpense={handleAddExpense} />
        </Modal>

        <Modal
          isOpen={openDeleteAlert.show}
          onClose={() => setOpenDeleteAlert({ show: false, data: null })}
          title="Delete Expense"
        >
          <DeleteAlert
            content="Are you sure you want to delete this expense?"
            onDelete={() => deleteExpense(openDeleteAlert.data)}
          />
        </Modal>
      </div>
    </DashboardLayout>
  );
};

export default Expense;
