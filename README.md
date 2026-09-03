# Twinstagram — Cyberbullying Detection Using LSTM + CNN

> A Flask-based social-media application that detects cyberbullying in user-generated text and OCR-extracted image text, helping identify harmful content and track user reputation.

![Demo GIF or screenshot](docs/demo.gif)

**Live demo:** Not deployed yet

**Stack:**
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![LSTM](https://img.shields.io/badge/Model-LSTM-blue)
![CNN](https://img.shields.io/badge/Model-1D%20CNN-green)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)
![GloVe](https://img.shields.io/badge/Embeddings-GloVe-purple)

---

## What this demonstrates

- **End-to-end ML application** — Integrated a trained TensorFlow/Keras hybrid LSTM + 1D CNN text classifier into a Flask web application for cyberbullying detection.

- **Hybrid text classification** — Used LSTM to capture sequential word relationships and 1D CNN to identify local patterns in the same GloVe-based text representation.

- **Real-world input handling** — Processes both normal text posts and text extracted from uploaded images using Tesseract OCR before sending the content through the detection pipeline.

---

## Architecture

```mermaid
flowchart TD
    A[User] --> B[Flask Web Application]

    B --> C[Text Post / Comment]
    B --> D[Image Upload]

    D --> E[Tesseract OCR]
    E --> F[Extracted Text]

    C --> G[Text Preprocessing]
    F --> G

    G --> H[GloVe Word Embeddings]

    H --> I[LSTM Branch]
    H --> J[1D CNN Branch]

    I --> K[Concatenate]
    J --> K

    K --> L[Cyberbullying Classifier]
    L --> M[Prediction]

    M --> N[Display Result]
    M --> O[Update Reputation Score]

    O --> P[Account Flagging]
