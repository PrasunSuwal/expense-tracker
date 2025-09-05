const User = require("../models/User");
const jwt = require("jsonwebtoken");

//Generate JWT token

const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, { expiresIn: "7d" });
};

//Register User
exports.registerUser = async (req, res) => {
  const { fullName, email, password, profileImageUrl } = req.body;

  //validate
  if (!fullName || !email || !password) {
    return res.status(400).json({ message: "All fileds are required" });
  }
  try {
    //check if mail already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "Email already in use" });
    }

    //Create the user
    const user = await User.create({
      fullName,
      email,
      password,
      profileImageUrl,
    });

    res.status(201).json({
      id: user._id,
      user,
      token: generateToken(user._id),
    });
  } catch (err) {
    res
      .status(500)
      .json({ message: "Error regsitering user", error: err.message });
  }
};

//LogIn User
exports.loginUser = async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ message: "All fields are required" });
  }
  try {
    const user = await User.findOne({ email });
    if (!user || !(await user.comparePassword(password))) {
      return res.status(400).json({ message: "Invalid credentials" });
    }
    res.status(200).json({
      id: user._id,
      user,
      token: generateToken(user._id),
    });
  } catch (err) {
    res
      .status(500)
      .json({ message: "Error regsitering user", error: err.message });
  }
};

//Get uSER info
exports.getUserInfo = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-password");
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    res.status(200).json(user);
  } catch (err) {
    res
      .status(500)
      .json({ message: "Error regsitering user", error: err.message });
  }
};

//Update user profile
exports.updateUserProfile = async (req, res) => {
  try {
    const userId = req.user.id;
    const { fullName, profileImageUrl } = req.body;

    const updateData = {};
    if (fullName) updateData.fullName = fullName;
    if (profileImageUrl) updateData.profileImageUrl = profileImageUrl;

    const updatedUser = await User.findByIdAndUpdate(userId, updateData, {
      new: true,
    }).select("-password");

    if (!updatedUser) {
      return res.status(404).json({ message: "User not found" });
    }

    res.status(200).json(updatedUser);
  } catch (err) {
    res
      .status(500)
      .json({ message: "Error updating user profile", error: err.message });
  }
};
