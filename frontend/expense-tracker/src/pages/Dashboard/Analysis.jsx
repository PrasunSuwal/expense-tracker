import React, { useEffect, useState, useCallback, useContext } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  LineChart,
  Line,
  CartesianGrid,
} from "recharts";
import { useUserAuth } from "../../hooks/useUserAuth";
import { UserContext } from "../../context/UserContext";
import DashboardLayout from "../../components/layouts/DashboardLayout";

const FORECAST_BASE_URL = "http://localhost:5005/api/forecast";

const Analysis = () => {
  useUserAuth();
  const { user } = useContext(UserContext);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [rows, setRows] = useState([{ category: "Salary", amount: "50000" }]);

  const fetchInsights = useCallback(async () => {
    if (!user?._id) {
      setError("User not authenticated");
      setLoading(false);
      return;
    }
    
    setLoading(true);
    setError("");
    try {
      const res = await axios.get(`${FORECAST_BASE_URL}/${user._id}`, { timeout: 15000 });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || "Failed to load insights");
    } finally {
      setLoading(false);
    }
  }, [user?._id]);

  const sendEstimates = useCallback(async (estimates) => {
    if (!user?._id) {
      setError("User not authenticated");
      return;
    }
    
    setLoading(true);
    setError("");
    try {
      const res = await axios.post(`${FORECAST_BASE_URL}/${user._id}`, { estimates }, { timeout: 15000 });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || "Failed to post estimates");
    } finally {
      setLoading(false);
    }
  }, [user?._id]);

  const addRow = () => setRows((r) => [...r, { category: "", amount: "" }]);
  const removeRow = (i) => setRows((r) => r.filter((_, idx) => idx !== i));
  const updateRow = (i, key, val) =>
    setRows((r) => r.map((row, idx) => (idx === i ? { ...row, [key]: val } : row)));

  const submitRows = () => {
    const estimates = {};
    rows.forEach(({ category, amount }) => {
      const name = String(category || "").trim();
      const num = Number(amount);
      if (name && Number.isFinite(num) && num > 0) {
        estimates[name] = num;
      }
    });
    if (Object.keys(estimates).length === 0) {
      setError("Please add at least one valid category with positive amount.");
      return;
    }
    sendEstimates(estimates);
  };

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);
  

  // Helpers for charts - Updated for monthly predictions
  const buildForecastBars = (forecast) => {
    const { next_month = "", categories = {} } = forecast || {};
    if (!next_month || !Object.keys(categories).length) return [];
    
    // Single month data point with all categories
    const row = { month: next_month };
    Object.entries(categories).forEach(([cat, amt]) => {
      row[cat] = amt || 0;
    });
    return [row]; // Return single month prediction
  };
  
  const categoryKeys = (forecast) => {
    if (!forecast?.categories) return [];
    return Object.keys(forecast.categories);
  };

  const noHistorical =
    data?.forecast_chart_data &&
    data.forecast_chart_data.categories &&
    Object.keys(data.forecast_chart_data.categories).length === 0;

  const forecastData = data ? buildForecastBars(data.forecast_chart_data) : [];
  const keys = data ? categoryKeys(data.forecast_chart_data) : [];

  return (
    <DashboardLayout activeMenu="Analysis">
      <div className="my-5 mx-auto">
        {/* Header Section */}
        <div className="mb-8 flex items-center justify-between bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Financial Analysis</h1>
            <p className="text-gray-600">AI-powered expense forecasts and personalized insights</p>
          </div>
          <button
            onClick={fetchInsights}
            className="inline-flex items-center px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors duration-200 shadow-sm"
            disabled={loading}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Refreshing...
              </>
            ) : (
              <>
                <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
                Refresh Analysis
              </>
            )}
          </button>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-8 w-8 bg-blue-200 rounded-full mb-4"></div>
              <div className="h-4 bg-gray-200 rounded w-48 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-32"></div>
            </div>
            <p className="text-gray-600 mt-4">Analyzing your financial data...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-6">
            <div className="flex items-center">
              <svg className="h-5 w-5 text-red-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z"></path>
              </svg>
              <div>
                <h3 className="font-medium text-red-900">Analysis Error</h3>
                <p className="text-sm text-red-800 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* No Historical Data - Estimates Input */}
        {!loading && data && noHistorical && (
          <div className="mb-8 rounded-lg border border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 p-8">
            <div className="text-center mb-6">
              <div className="mx-auto h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-blue-900 mb-2">Get Started with Forecasts</h3>
              <p className="text-blue-800 max-w-2xl mx-auto">
                No transaction history found. Enter your monthly income and expense estimates below to generate 
                personalized forecasts and insights. You can always update these values later.
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 shadow-sm">
              <h4 className="font-medium text-gray-900 mb-4">Monthly Estimates</h4>
              <div className="space-y-4">
                {rows.map((row, i) => (
                  <div key={i} className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                    <div className="md:col-span-5">
                      <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                      <input
                        className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                        placeholder="e.g., Food, Transport, Salary"
                        value={row.category}
                        onChange={(e) => updateRow(i, "category", e.target.value)}
                      />
                    </div>
                    <div className="md:col-span-5">
                      <label className="block text-sm font-medium text-gray-700 mb-1">Monthly Amount (Rs.)</label>
                      <input
                        className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                        placeholder="e.g., 12000"
                        value={row.amount}
                        onChange={(e) => updateRow(i, "amount", e.target.value)}
                        type="number"
                        min="0"
                      />
                    </div>
                    <div className="md:col-span-2">
                      <button
                        onClick={() => removeRow(i)}
                        className="w-full rounded-lg border border-red-200 px-4 py-3 text-sm text-red-700 hover:bg-red-50 transition-colors duration-200"
                        disabled={rows.length === 1}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={addRow}
                  className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 transition-colors duration-200"
                >
                  <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path>
                  </svg>
                  Add Category
                </button>
                <button
                  onClick={submitRows}
                  className="inline-flex items-center px-6 py-2 rounded-lg bg-blue-600 text-sm font-medium text-white hover:bg-blue-700 transition-colors duration-200"
                  disabled={loading}
                >
                  Generate Forecasts
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Charts Section */}
        {!loading && data && (
          <div className="mb-8">
            {/* Expense Forecast Chart */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Next Month Forecast</h3>
                  <p className="text-sm text-gray-600">{data?.forecast_chart_data?.next_month ? `Predicted spending for ${data.forecast_chart_data.next_month}` : 'AI-powered expense prediction'}</p>
                </div>
                <div className="flex items-center gap-4">
                  {data?.forecast_chart_data?.total && (
                    <div className="text-right">
                      <p className="text-xs text-gray-500 font-medium">Total Predicted</p>
                      <p className="text-xl font-bold text-blue-600">Rs.{data.forecast_chart_data.total.toLocaleString('en-IN')}</p>
                    </div>
                  )}

                  <div className="p-2 bg-blue-100 rounded-lg">
                    <svg className="h-5 w-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                    </svg>
                  </div>
                </div>
              </div>
              {forecastData.length ? (
                <div style={{ width: "100%", height: 350 }}>
                  <ResponsiveContainer>
                    <BarChart data={forecastData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '1px solid #e5e7eb', 
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }} 
                      />
                      <Legend />
                      {keys.map((k, i) => (
                        <Bar
                          key={k}
                          dataKey={k}
                          stackId="a"
                          fill={["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#f97316"][i % 6]}
                          radius={i === keys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="mx-auto h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                    <svg className="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                    </svg>
                  </div>
                  <p className="text-gray-500">No forecast data available</p>
                  <p className="text-sm text-gray-400 mt-1">Add transactions to see predictions</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Insights, Warnings, and Recommendations Section */}
        {!loading && data && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Insights */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-green-100 rounded-lg mr-4">
                  <svg className="h-5 w-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Financial Insights</h3>
                  <p className="text-sm text-gray-600">AI-generated recommendations</p>
                </div>
              </div>
              
              {/* Data Quality Message */}
              <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-start">
                  <svg className="h-4 w-4 text-blue-500 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                  <p className="text-xs text-blue-700 leading-relaxed">
                    <strong>Note:</strong> Predictions and confidence scores improve with more consistent tracking and historical data. Keep adding transactions regularly for better insights!
                  </p>
                </div>
              </div>
              
              {data.insights?.length ? (
                <div className="space-y-4">
                  {data.insights.map((insight, idx) => (
                    <div key={idx} className="flex items-start p-4 bg-green-50 rounded-lg border border-green-200">
                      <div className="flex-shrink-0 mt-0.5">
                        <div className="h-2 w-2 bg-green-400 rounded-full"></div>
                      </div>
                      <p className="ml-3 text-sm text-green-800 leading-relaxed">{insight}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="mx-auto h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                    <svg className="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                  </div>
                  <p className="text-gray-500">No insights available</p>
                  <p className="text-sm text-gray-400 mt-1">Add more transaction data for personalized insights</p>
                </div>
              )}
            </div>

            {/* Warnings */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-red-100 rounded-lg mr-4">
                  <svg className="h-5 w-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z"></path>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Alerts & Warnings</h3>
                  <p className="text-sm text-gray-600">Areas that need attention</p>
                </div>
              </div>
              
              {data.warnings?.length ? (
                <div className="space-y-4">
                  {data.warnings.map((warning, idx) => (
                    <div key={idx} className="flex items-start p-4 bg-red-50 rounded-lg border border-red-200">
                      <div className="flex-shrink-0 mt-0.5">
                        <svg className="h-4 w-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z"></path>
                        </svg>
                      </div>
                      <p className="ml-3 text-sm text-red-800 leading-relaxed">{warning}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="mx-auto h-12 w-12 rounded-full bg-green-100 flex items-center justify-center mb-4">
                    <svg className="h-6 w-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                  </div>
                  <p className="text-gray-600 font-medium">All Good!</p>
                  <p className="text-sm text-gray-400 mt-1">No warnings at this time.</p>
                </div>
              )}
            </div>

            {/* Recommendations */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-purple-100 rounded-lg mr-4">
                  <svg className="h-5 w-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Action Plan</h3>
                  <p className="text-sm text-gray-600">Personalized recommendations</p>
                </div>
              </div>
              
              {data.recommendations?.length ? (
                <div className="space-y-4">
                  {data.recommendations.map((rec, idx) => (
                    <div key={idx} className="flex items-start p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <div className="flex-shrink-0 mt-0.5">
                        <svg className="h-4 w-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path>
                        </svg>
                      </div>
                      <p className="ml-3 text-sm text-purple-800 leading-relaxed">{rec}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="mx-auto h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                    <svg className="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                    </svg>
                  </div>
                  <p className="text-gray-500">No recommendations available</p>
                  <p className="text-sm text-gray-400 mt-1">Track expenses to get personalized action items</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default Analysis;
