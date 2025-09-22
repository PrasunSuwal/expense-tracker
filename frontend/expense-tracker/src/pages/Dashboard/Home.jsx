import React, { useEffect, useState } from "react";
import DashboardLayout from "../../components/layouts/DashboardLayout";
import { useUserAuth } from "../../hooks/useUserAuth";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../../utils/axiosInstance";
import { API_PATHS } from "../../utils/apiPaths";
import InfoCard from "../../components/Cards/InfoCard";
import { LuHandCoins, LuWalletMinimal } from "react-icons/lu";
import { IoMdCard } from "react-icons/io";
import { addThousandsSeperator } from "../../utils/helper";
import RecentTransactions from "../../components/Dashboard/RecentTransactions";
import FinanceOverview from "../../components/Dashboard/FinanceOverview";
import ExpenseTransactions from "../../components/Dashboard/ExpenseTransactions";
import Last30DaysExpenses from "../../components/Dashboard/Last30DaysExpenses";
import RecentIncomeWithChart from "../../components/Dashboard/RecentIncomeWithChart";
import RecentIncome from "../../components/Dashboard/RecentIncome";

const Home = () => {
  useUserAuth();

  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const now = new Date();
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth() + 1); // 1-12
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());
  const [overall, setOverall] = useState(false); // default is current month

  const fetchDashboardData = async () => {
    if (loading) return;
    setLoading(true);
    try {
      let url = `${API_PATHS.DASHBOARD.GET_DATA}`;
      if (overall) {
        url += `?overall=true`;
      } else {
        url += `?month=${selectedMonth}&year=${selectedYear}`;
      }
      const response = await axiosInstance.get(url);
      if (response.data) {
        setDashboardData(response.data);
      }
    } catch (error) {
      console.log("Something went wrong. Please try again.", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    return () => {};
  }, [selectedMonth, selectedYear, overall]);

  // console.log("dashboardData", dashboardData);

  return (
    <DashboardLayout activeMenu="Dashboard">
      <div className="my-5 mx-auto">
        {/* Dropdown selector for month/year/overall, right aligned */}
        <div className="mb-4 flex items-center justify-end">
          <select
            value={overall ? "overall" : selectedMonth}
            onChange={(e) => {
              if (e.target.value === "overall") {
                setOverall(true);
              } else {
                setSelectedMonth(Number(e.target.value));
                setOverall(false);
              }
            }}
            className="px-3 py-2 rounded text-base focus:outline-none focus:ring-2 focus:ring-purple-300 bg-white shadow-sm"
            style={{ minWidth: 180, border: "none" }}
          >
            <option value="overall">Overall</option>
            {[...Array(12)].map((_, i) => (
              <option key={i + 1} value={i + 1}>
                {new Date(now.getFullYear(), i).toLocaleString("default", {
                  month: "long",
                })}{" "}
                {now.getFullYear()}
              </option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <InfoCard
            icon={<IoMdCard />}
            label="Total Balance"
            value={addThousandsSeperator(dashboardData?.totalBalance || 0)}
            color="bg-primary"
          />
          <InfoCard
            icon={<LuWalletMinimal />}
            label="Total Income"
            value={addThousandsSeperator(dashboardData?.totalIncome || 0)}
            color="bg-orange-500"
          />
          <InfoCard
            icon={<LuHandCoins />}
            label="Total Expense"
            value={addThousandsSeperator(dashboardData?.totalExpense || 0)}
            color="bg-red-500"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <RecentTransactions
            transactions={dashboardData?.recentTransactions}
            onSeeMore={() => {
              navigate("/expense");
            }}
          />
          <FinanceOverview
            totalBalance={dashboardData?.totalBalance || 0}
            totalIncome={dashboardData?.totalIncome || 0}
            totalExpense={dashboardData?.totalExpense || 0}
          />

          <ExpenseTransactions
            transactions={dashboardData?.last30DaysExpenses?.transactions || []}
            onSeeMore={() => navigate("/expense")}
          />
          <Last30DaysExpenses
            data={dashboardData?.last30DaysExpenses?.transactions || []}
          />

          <RecentIncomeWithChart
            data={
              dashboardData?.last30DaysIncome?.transactions?.slice(0, 4) || []
            }
            totalIncome={dashboardData?.totalIncome || 0}
          />
          <RecentIncome
            transactions={dashboardData?.last30DaysIncome?.transactions || []}
            onSeeMore={() => navigate("/income")}
          />
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Home;
