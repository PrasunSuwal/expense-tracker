import React from "react";
import CARD from "../../assets/images/card.png";
import { LuTrendingUpDown } from "react-icons/lu";
import logo from "../../assets/images/logo.png";
import { useNavigate } from "react-router-dom";

const AuthLayout = ({ children }) => {
  const navigate = useNavigate();
  return (
    <div className="flex h-screen">
      {/* Left side: Login form */}
      <div className="w-full md:w-1/2 px-8 md:px-12 pt-8 pb-12 bg-white">
        <div
          className="flex items-center mb-6 cursor-pointer"
          onClick={() => navigate("/")}
        >
          <img src={logo} alt="Logo" className="rounded-full w-20 h-20 mr-3" />
          <h2 className="text-2xl font-bold text-purple-700">AutoCA</h2>
        </div>
        {children}
      </div>

      {/* Right side: Design + Stats */}
      <div className="hidden md:flex w-1/2 h-screen bg-violet-50 relative items-center justify-center overflow-hidden">
        {/* Decorative blobs */}
        <div className="absolute w-48 h-48 rounded-[40px] bg-purple-600 -top-7 -left-5" />
        <div className="absolute w-48 h-48 rounded-[40px] border-[20px] border-fuchsia-600 top-[30%] -right-10" />
        <div className="absolute w-48 h-48 rounded-[40px] bg-violet-500 -bottom-7 -left-5" />

        {/* Content (stat card and image) */}
        <div className="relative z-10 w-[80%] max-w-md flex flex-col gap-8">
          <StatsInfoCard
            icon={<LuTrendingUpDown />}
            label="Track Your Income & Expenses"
            value="430,000"
            color="bg-purple-600"
          />
          <img
            src={CARD}
            alt="Credit Card"
            className="w-full shadow-lg shadow-blue-400/15 rounded-xl"
          />
        </div>
      </div>
    </div>
  );
};

export default AuthLayout;

// Stats Card Component
const StatsInfoCard = ({ icon, label, value, color }) => {
  return (
    <div className="flex gap-6 bg-white p-4 rounded-xl shadow-md shadow-purple-400/10 border border-gray-200 z-10">
      <div
        className={`w-12 h-12 flex items-center justify-center text-[26px] text-white ${color} rounded-full drop-shadow-xl`}
      >
        {icon}
      </div>
      <div>
        <h6 className="text-xs text-gray-500 mb-1">{label}</h6>
        <span className="text-[20px] font-semibold">Rs {value}</span>
      </div>
    </div>
  );
};
