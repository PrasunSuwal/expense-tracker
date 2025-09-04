const express = require("express");
const axios = require("axios");
const FormData = require("form-data");
const multer = require("multer");

const router = express.Router();
const upload = multer();

const OCR_URL = process.env.OCR_API_URL || "http://localhost:8000";

router.post("/ocr/process", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "file required" });
    const form = new FormData();
    form.append("file", req.file.buffer, { filename: req.file.originalname });

    const { data } = await axios.post(`${OCR_URL}/process`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
    });
    return res.json(data);
  } catch (err) {
    return res.status(500).json({ error: err?.response?.data?.detail || err.message });
  }
});

router.post("/ocr/feedback", async (req, res) => {
  try {
    const { data } = await axios.post(`${OCR_URL}/feedback`, req.body);
    return res.json(data);
  } catch (err) {
    return res.status(500).json({ error: err?.response?.data?.detail || err.message });
  }
});

module.exports = router;


