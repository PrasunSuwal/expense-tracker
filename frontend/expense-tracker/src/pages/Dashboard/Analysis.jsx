import React, { useEffect, useState, useCallback } from "react";
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

const INSIGHTS_URL = "http://localhost:5005/api/insights";

const Analysis = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [rows, setRows] = useState([{ category: "Salary", amount: "50000" }]);

  const fetchInsights = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.get(INSIGHTS_URL, { timeout: 15000 });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || "Failed to load insights");
    } finally {
      setLoading(false);
    }
  }, []);

  const sendEstimates = useCallback(async (estimates) => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.post(INSIGHTS_URL, { estimates }, { timeout: 15000 });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || "Failed to post estimates");
    } finally {
      setLoading(false);
    }
  }, []);

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

  // Helpers for charts
  const buildForecastBars = (forecast) => {
    const { labels = [], categories = {} } = forecast || {};
    return labels.map((label, idx) => {
      const row = { week: label };
      Object.entries(categories).forEach(([cat, vals]) => {
        row[cat] = Array.isArray(vals) ? vals[idx] ?? 0 : 0;
      });
      return row;
    });
  };
  const categoryKeys = (forecast) => Object.keys(forecast?.categories || {});

  const noHistorical =
    data?.forecast_chart_data &&
    data.forecast_chart_data.categories &&
    Object.keys(data.forecast_chart_data.categories).length === 0;

  const forecastData = data ? buildForecastBars(data.forecast_chart_data) : [];
  const keys = data ? categoryKeys(data.forecast_chart_data) : [];
  const cashflow = data?.cashflow_chart_data || {};
  const cashflowPoints = (cashflow.labels || []).map((l, i) => ({ week: l, value: cashflow.values?.[i] ?? 0 }));

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900">Analysis</h2>
          <p className="text-sm text-gray-500">Forecasts, cashflow and personalized insights</p>
        </div>
        <button
          onClick={fetchInsights}
          className="px-4 py-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm"
        >
          Refresh
        </button>
      </div>

      {loading && (
        <div className="rounded-md border border-gray-200 p-4 text-gray-700">Loading insights…</div>
      )}

      {error && (
        <div className="mb-4 rounded-md border border-red-200 bg-red-50 p-4 text-red-700">
          {error}
        </div>
      )}

      {!loading && data && noHistorical && (
        <div className="mb-6 rounded-lg border border-blue-200 bg-blue-50 p-5">
          <h3 className="font-medium text-blue-900 mb-1">No historical data</h3>
          <p className="text-sm text-blue-800 mb-4">
            Enter your monthly estimates to generate immediate forecasts. You can add more rows or adjust
            values any time.
          </p>

          <div className="space-y-2">
            {rows.map((row, i) => (
              <div key={i} className="grid grid-cols-12 gap-3">
                <input
                  className="col-span-6 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Category (e.g., Food)"
                  value={row.category}
                  onChange={(e) => updateRow(i, "category", e.target.value)}
                />
                <input
                  className="col-span-4 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Amount (e.g., 12000)"
                  value={row.amount}
                  onChange={(e) => updateRow(i, "amount", e.target.value)}
                  type="number"
                  min="0"
                />
                <div className="col-span-2 flex items-center">
                  <button
                    onClick={() => removeRow(i)}
                    className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 flex gap-3">
            <button
              onClick={addRow}
              className="rounded-md border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              Add row
            </button>
            <button
              onClick={submitRows}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              Submit estimates
            </button>
          </div>
        </div>
      )}

      {!loading && data && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <section className="rounded-lg border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">Forecast (next 4 weeks)</h3>
            {forecastData.length ? (
              <div style={{ width: "100%", height: 320 }}>
                <ResponsiveContainer>
                  <BarChart data={forecastData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {keys.map((k, i) => (
                      <Bar
                        key={k}
                        dataKey={k}
                        stackId="a"
                        fill={["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171"][i % 6]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-sm text-gray-500">No category forecasts yet.</div>
            )}
          </section>

          <section className="rounded-lg border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">Net Cashflow (next 4 weeks)</h3>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={cashflowPoints}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>
      )}

      {!loading && data && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <section className="rounded-lg border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">Insights</h3>
            {data.insights?.length ? (
              <ul className="list-disc pl-5 space-y-2 text-sm text-gray-800">
                {data.insights.map((t, idx) => (
                  <li key={idx}>{t}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-gray-500">No insights available.</div>
            )}
          </section>

          <section className="rounded-lg border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">Warnings</h3>
            {data.warnings?.length ? (
              <ul className="list-disc pl-5 space-y-2 text-sm text-gray-800">
                {data.warnings.map((t, idx) => (
                  <li key={idx}>{t}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-gray-500">No warnings.</div>
            )}
          </section>
        </div>
      )}
    </div>
  );
};

export default Analysis;
