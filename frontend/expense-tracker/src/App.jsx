import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/Auth/Login";
import SignUp from "./pages/Auth/SignUp";
import Home from "./pages/Dashboard/Home";
import Expense from "./pages/Dashboard/Expense";
import Income from "./pages/Dashboard/Income";
import Profile from "./pages/Dashboard/Profile";
import LandingPage from "./pages/Dashboard/LandingPage";
import UserProvider from "./context/UserContext";
import { Toaster } from "react-hot-toast";
// import { useUserAuth } from "./hooks/useUserAuth";
import ScrollToTop from "./components/Scroll/ScrollToTop";
import Analysis from "./pages/Dashboard/Analysis";

const App = () => {
  return (
    <Router>
      <UserProvider>
        <ScrollToTop />
        <AppContent />
      </UserProvider>
    </Router>
  );
};

const AppContent = () => {
  // ...existing code...
  return (
    <>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" exact element={<Login />} />
        <Route path="/signUp" exact element={<SignUp />} />
        <Route path="/dashboard" exact element={<Home />} />
        <Route path="/income" exact element={<Income />} />
        <Route path="/expense" exact element={<Expense />} />
        <Route path="/analysis" exact element={<Analysis />} />
        <Route path="/profile" exact element={<Profile />} />
      </Routes>
      <Toaster
        toastOptions={{
          clasName: "",
          style: {
            fontSize: "13px",
          },
        }}
      />
    </>
  );
};

export default App;
