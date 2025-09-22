import React, { useState } from "react";
import DashboardLayout from "../../components/layouts/DashboardLayout";
import IncomeOverview from "../../components/Income/IncomeOverview";
import axiosInstance from "../../utils/axiosInstance";
import { API_PATHS } from "../../utils/apiPaths";
import { useEffect } from "react";
import Modal from "../../components/Modal";
import AddIncomeForm from "../../components/Income/AddIncomeForm";
import toast from "react-hot-toast";
import IncomeList from "../../components/Income/IncomeList";
import DeleteAlert from "../../components/DeleteAlert";
import { useUserAuth } from "../../hooks/useUserAuth";

const Income = () => {
  useUserAuth();

  const [incomeData, setIncomeData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDeleteAlert, setOpenDeleteAlert] = useState({
    show: false,
    data: null,
  });
  const [openAddIncomeModal, setOpenAddIncomeModal] = useState(false);
  // Month/year selection
  const now = new Date();
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth() + 1); // 1-12
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());

  //Get all Income Details
  const fetchIncomeDetails = async () => {
    if (loading) return;
    setLoading(true);
    try {
      const response = await axiosInstance.get(
        `${API_PATHS.INCOME.GET_ALL_INCOME}?month=${selectedMonth}&year=${selectedYear}`
      );
      if (response.data) {
        setIncomeData(response.data);
      }
    } catch (error) {
      console.log("Something went wrong.Please try againn", error);
    } finally {
      setLoading(false);
    }
  };

  //Handle All Income
  const handleAddIncome = async (income) => {
    const { source, amount, date, icon } = income;

    //Validation checks
    if (!source.trim()) {
      toast.error("Source is required.");
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
      await axiosInstance.post(API_PATHS.INCOME.ADD_INCOME, {
        source,
        amount,
        date,
        icon,
      });
      setOpenAddIncomeModal(false);
      toast.success("Income added successfully.");
      fetchIncomeDetails();
    } catch (error) {
      console.error(
        "Error adding income",
        error.response?.data?.message || error.message
      );
    }
  };

  //Delete Income
  const deleteIncome = async (id) => {
    try {
      await axiosInstance.delete(API_PATHS.INCOME.DELETE_INCOME(id));
      setOpenDeleteAlert({ show: false, data: null });
      toast.success("Income deleted successfully.");
      fetchIncomeDetails();
    } catch (error) {
      console.error(
        "Error deleting income",
        error.response?.data?.message || error.message
      );
    }
  };

  //handle download income details
  const handleDownloadIncomeDetails = async () => {
    try {
      const response = await axiosInstance.get(
        `${API_PATHS.INCOME.DOWNLOAD_INCOME}`,
        {
          responseType: "blob",
        }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "income-details.xlsx");
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading income details", error);
      toast.error("Error downloading income details");
    }
  };

  useEffect(() => {
    fetchIncomeDetails();
    return () => {};
  }, [selectedMonth, selectedYear]);

  return (
    <DashboardLayout activeMenu="Income">
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
              <IncomeOverview
                transactions={incomeData}
                onAddIncome={setOpenAddIncomeModal}
              />
            </div>
            <IncomeList
              transactions={incomeData}
              onDelete={(id) => {
                setOpenDeleteAlert({ show: true, data: id });
              }}
              onDownload={handleDownloadIncomeDetails}
            />
          </div>
        )}

        <Modal
          isOpen={openAddIncomeModal}
          onClose={() => setOpenAddIncomeModal(false)}
          title="Add Income"
        >
          <AddIncomeForm onAddIncome={handleAddIncome} />
        </Modal>
        <Modal
          isOpen={openDeleteAlert.show}
          onClose={() => setOpenDeleteAlert({ show: false, data: null })}
          title="Delete Income"
        >
          <DeleteAlert
            content="Are you sure you want to delete this income?"
            onDelete={() => deleteIncome(openDeleteAlert.data)}
          />
        </Modal>
      </div>
    </DashboardLayout>
  );
};

export default Income;
