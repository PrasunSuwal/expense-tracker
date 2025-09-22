import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../../assets/images/logo.png";
import hero from "../../assets/images/hero.png";
import demo from "../../assets/images/demo.png";
import about from "../../assets/images/about.png";
import expenseTracking from "../../assets/images/expense-tracking.png";
import reporting from "../../assets/images/reporting.png";
import desktop from "../../assets/images/desktop.png";
import budget from "../../assets/images/budget.png";
import charts from "../../assets/images/charts.png";
import bills from "../../assets/images/bills.png";
import shikshya from "../../assets/images/shikshya.png";
import rabi from "../../assets/images/rabi.png";
import shreya from "../../assets/images/shreya.png";
import { useContext } from "react";
import { UserContext } from "../../context/UserContext";

const LandingPage = () => {
  const navigate = useNavigate();
  const [activeMenu, setActiveMenu] = useState("home");

  const handleGetStarted = () => navigate("/signup");
  const handleLogIn = () => navigate("/login");
  const handleSignUp = () => navigate("/signup");
  const { user } = useContext(UserContext);
  const profileImageUrl = user?.profileImageUrl;
  const isAuthenticated = !!localStorage.getItem("token");

  const handleMenuClick = (menuItem) => {
    setActiveMenu(menuItem);
  };

  return (
    <div className="bg-white text-gray-900 font-sans">
      {/* Navigation */}
      <header className="flex justify-between items-center px-4 py-3 border-b border-gray-200 sticky top-0 z-50 bg-white">
        <div className="flex items-center">
          <img src={logo} alt="Logo" className="rounded-full w-20 h-20 mr-3" />
          <h2 className="text-2xl font-bold text-purple-700">AutoCA</h2>
        </div>
        <nav>
          <ul className="flex gap-6 text-base font-semibold tracking-wide uppercase">
            <li>
              <a
                href="#home"
                className={`relative px-3 py-1 rounded-lg transition-all duration-300 hover:bg-purple-100 hover:text-purple-700 hover:shadow-md focus:bg-purple-200 focus:text-purple-900 ${
                  activeMenu === "home"
                    ? "bg-purple-200 text-purple-900 shadow-lg transform -translate-y-1"
                    : "hover:-translate-y-1"
                }`}
                onClick={() => handleMenuClick("home")}
              >
                Home
              </a>
            </li>
            <li>
              <a
                href="#features"
                className={`relative px-3 py-1 rounded-lg transition-all duration-300 hover:bg-purple-100 hover:text-purple-700 hover:shadow-md focus:bg-purple-200 focus:text-purple-900 ${
                  activeMenu === "features"
                    ? "bg-purple-200 text-purple-900 shadow-lg transform -translate-y-1"
                    : "hover:-translate-y-1"
                }`}
                onClick={() => handleMenuClick("features")}
              >
                Features
              </a>
            </li>
            <li>
              <a
                href="#about"
                className={`relative px-3 py-1 rounded-lg transition-all duration-300 hover:bg-purple-100 hover:text-purple-700 hover:shadow-md focus:bg-purple-200 focus:text-purple-900 ${
                  activeMenu === "about"
                    ? "bg-purple-200 text-purple-900 shadow-lg transform -translate-y-1"
                    : "hover:-translate-y-1"
                }`}
                onClick={() => handleMenuClick("about")}
              >
                About Us
              </a>
            </li>
            <li>
              <a
                href="#testimonials-section"
                className={`relative px-3 py-1 rounded-lg transition-all duration-300 hover:bg-purple-100 hover:text-purple-700 hover:shadow-md focus:bg-purple-200 focus:text-purple-900 ${
                  activeMenu === "review"
                    ? "bg-purple-200 text-purple-900 shadow-lg transform -translate-y-1"
                    : "hover:-translate-y-1"
                }`}
                onClick={() => handleMenuClick("review")}
              >
                Review
              </a>
            </li>
            <li>
              <a
                href="#contact"
                className={`relative px-3 py-1 rounded-lg transition-all duration-300 hover:bg-purple-100 hover:text-purple-700 hover:shadow-md focus:bg-purple-200 focus:text-purple-900 ${
                  activeMenu === "contact"
                    ? "bg-purple-200 text-purple-900 shadow-lg transform -translate-y-1"
                    : "hover:-translate-y-1"
                }`}
                onClick={() => handleMenuClick("contact")}
              >
                Contact
              </a>
            </li>
          </ul>
        </nav>
        <div className="flex gap-2">
          {isAuthenticated ? (
            <div className="flex items-center gap-2">
              <button
                className="bg-purple-700 text-white px-6 py-2 rounded-xl font-semibold text-lg flex items-center gap-2 hover:bg-purple-800 transition"
                onClick={() => navigate("/dashboard")}
              >
                Dashboard
                <span className="ml-1">→</span>
              </button>
              {profileImageUrl && (
                <img
                  src={profileImageUrl}
                  alt="Profile"
                  className="w-10 h-10 rounded-full ml-2"
                />
              )}
            </div>
          ) : (
            <>
              <button
                className="bg-purple-700 text-white px-4 py-1.5 rounded-lg font-semibold hover:bg-purple-800 transition"
                onClick={handleLogIn}
              >
                Log In
              </button>
              <button
                className="bg-purple-700 text-white px-4 py-1.5 rounded-lg font-semibold hover:bg-purple-800 transition"
                onClick={handleSignUp}
              >
                Sign Up
              </button>
            </>
          )}
        </div>
      </header>

      {/* Hero Section */}
      <section
        className="flex flex-col md:flex-row items-center justify-between px-10 py-16 md:py-24"
        id="home"
      >
        <div className="max-w-xl mb-10 md:mb-0">
          <h1 className="text-5xl font-extrabold mb-6">Welcome to AutoCA</h1>
          <p className="text-lg text-gray-600 mb-8">
            Navigate Your Finances with AutoCA.
          </p>
          <button
            className="bg-purple-700 text-white px-7 py-3 rounded-lg font-semibold text-lg hover:bg-purple-800 transition"
            onClick={handleGetStarted}
          >
            Get Started
          </button>
        </div>
        <div className="flex-shrink-0">
          <img
            src={hero}
            alt="Hero"
            className="w-[350px] md:w-[500px] lg:w-[620px] h-[300px] md:h-[400px] lg:h-[500px] object-contain"
          />
        </div>
      </section>

      {/* Demo Image Section */}
      <section className="py-8 px-10">
        <div className="flex justify-center">
          <img
            src={demo}
            alt="Finance Illustration"
            className="max-w-full h-auto rounded-xl shadow-md"
          />
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-gray-50 py-16 px-10 text-center" id="features">
        <h2 className="text-3xl font-bold text-purple-900 mb-2">
          Key Features
        </h2>
        <div className="w-20 h-1 bg-gray-900 mx-auto mb-10 rounded"></div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={expenseTracking}
              alt="Expense Tracking"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Expense Tracking
            </h3>
            <p className="text-gray-600">
              Keeping track of all personal or business expenses to understand
              where money is being spent.
            </p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={reporting}
              alt="Financial Reporting"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Financial Reporting
            </h3>
            <p className="text-gray-600">
              Generating summaries and statements to analyze financial
              performance over time.
            </p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={desktop}
              alt="Desktop Web Version"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Desktop Web Version
            </h3>
            <p className="text-gray-600">Manages businesses from Desktop</p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={budget}
              alt="Budgets Management"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Income/Budgets Management
            </h3>
            <p className="text-gray-600">
              Organizing and controlling income and expenses.
            </p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={charts}
              alt="Visual Representation"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Visual Representation
            </h3>
            <p className="text-gray-600">
              Using graphs and charts to visually display financial data for
              easier understanding and analysis.
            </p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow hover:shadow-lg flex flex-col items-center transition">
            <img
              className="w-24 h-24 object-contain mb-4"
              src={bills}
              alt="Upload Bill Images"
            />
            <h3 className="text-lg font-semibold text-purple-900 mb-2">
              Upload Bill Images
            </h3>
            <p className="text-gray-600">
              Paper bills and receipts can be uploaded and organized.
            </p>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-16 px-10 bg-white" id="about">
        <div className="flex flex-col md:flex-row items-center gap-12 max-w-6xl mx-auto">
          <div className="flex-1 flex justify-center mb-8 md:mb-0">
            <img
              src={about}
              alt="About Illustration"
              className="w-72 max-w-full rounded-full border-4 border-dashed border-gray-300 p-4"
            />
          </div>
          <div className="flex-1">
            <h2 className="text-3xl font-bold text-purple-700 mb-4">
              What is AutoCA?
            </h2>
            <div className="w-12 h-1 bg-purple-400 mb-6 rounded"></div>
            <p className="text-lg text-gray-700 mb-4">
              AutoCA is a comprehensive, full-stack accounting automation system
              designed to assist individuals and small businesses in managing
              their financial records. It simplifies traditional accounting
              workflows through intelligent AI modules, reducing dependency on
              financial assistants for routine tasks.
            </p>
            <p className="text-lg text-gray-700 mb-6">
              Let us guide you toward a more secure financial future!
            </p>
            <button
              className="bg-purple-700 text-white px-7 py-3 rounded-lg font-semibold text-lg hover:bg-purple-800 transition"
              onClick={handleGetStarted}
            >
              Get Started
            </button>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section
        className="bg-gray-100 py-16 px-10 text-center"
        id="testimonials-section"
      >
        <h2 className="text-2xl font-bold text-purple-900 mb-2">
          Hear from our trusted business owners
        </h2>
        <div className="w-16 h-1 bg-purple-900 mx-auto mb-10 rounded"></div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 justify-items-center">
          <div className="bg-white rounded-xl p-8 shadow flex flex-col items-center hover:shadow-lg transition">
            <img
              src={shikshya}
              alt="Shikshya Gurung"
              className="w-20 h-20 object-cover rounded-full mb-4"
            />
            <p className="italic text-gray-700 mb-4 min-h-[80px]">
              "I have been using AutoCA to record sales & purchases in my
              stationery and it is easier to know my profits and losses"
            </p>
            <h4 className="font-bold text-lg">Shikshya Gurung</h4>
            <p className="text-gray-500">Shikshya Stationery</p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow flex flex-col items-center hover:shadow-lg transition">
            <img
              src={rabi}
              alt="Rabi Thapa"
              className="w-20 h-20 object-cover rounded-full mb-4"
            />
            <p className="italic text-gray-700 mb-4 min-h-[80px]">
              "Daily income ra expenses track garna plus further expenses
              predict garna ekdam sajilo bhayeko cha."
            </p>
            <h4 className="font-bold text-lg">Rabi Thapa</h4>
            <p className="text-gray-500">RR Store</p>
          </div>
          <div className="bg-white rounded-xl p-8 shadow flex flex-col items-center hover:shadow-lg transition">
            <img
              src={shreya}
              alt="Shreya Bhusal"
              className="w-20 h-20 object-cover rounded-full mb-4"
            />
            <p className="italic text-gray-700 mb-4 min-h-[80px]">
              "It's been easier to track my expenses and incomes digitally
              rather than manually. I really like AutoCA"
            </p>
            <h4 className="font-bold text-lg">Shreya Bhusal</h4>
            <p className="text-gray-500">Shreya Confectionery</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        id="contact"
        className="bg-purple-900 text-purple-100 py-10 px-6 mt-10"
      >
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="mb-6 md:mb-0">
            <h3 className="text-4xl font-bold mb-2">AutoCA</h3>
            <p className="text-sm">© 2025 AutoCA. All rights reserved.</p>
          </div>
          <ul className="flex flex-col md:flex-row gap-10 text-base">
            <li>
              <span className="font-semibold">Pages</span>
              <ul className="ml-2 md:ml-0">
                <li>Home</li>
                <li>About Us</li>
                <li>Contact</li>
              </ul>
            </li>
            <li>
              <span className="font-semibold">Contact</span>
              <ul className="ml-2 md:ml-0">
                <li>01-5555667</li>
                <li>autoCA29@gmail.com</li>
                <li>Kathmandu, Nepal</li>
              </ul>
            </li>
          </ul>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
