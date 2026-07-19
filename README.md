# 🧬 DiagnosAI – AI-Powered Cancer Detection Web Application

<div align="center">

### *Early Breast Cancer Detection using Machine Learning*

DiagnosAI is a full-stack web application that predicts whether a breast tumor is **Benign** or **Malignant** using a Machine Learning model trained on clinical diagnostic parameters. The application provides an intuitive interface for healthcare professionals to analyze patient data while securely storing diagnosis records in MongoDB Atlas.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?style=for-the-badge&logo=mongodb)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)

</div>

---

# 📖 Overview

DiagnosAI is an AI-powered healthcare application designed to assist in breast cancer diagnosis using Machine Learning.

The application predicts whether a tumor is **Benign** or **Malignant** by analyzing **30 clinical diagnostic parameters** from the Breast Cancer Wisconsin Dataset.

In addition to prediction, the application stores patient information and diagnosis history in a MongoDB Atlas database, allowing users to review previous patient records through a dedicated interface.

---

# ✨ Features

- 🧬 Breast Cancer Prediction using Machine Learning
- 🧠 Logistic Regression Classification Model
- 🩺 Predict using 30 Medical Diagnostic Features
- 🖥️ Modern React Frontend
- ⚙️ Flask REST API Backend
- ☁️ MongoDB Atlas Cloud Database
- 📋 Store Patient Information
- 📜 View Previous Patient Records
- 🌐 Cross-Origin Support using Flask-CORS
- ⚡ Fast Prediction Response

---

# 🏗 Tech Stack

## Frontend

- React.js
- Tailwind CSS
- Fetch API

---

## Backend

- Flask
- Flask-CORS
- Python
- NumPy
- Scikit-Learn

---

## Database

- MongoDB Atlas

---

## Machine Learning

- Logistic Regression
- Joblib
- Breast Cancer Wisconsin Dataset

---

# 📂 Project Structure

```text
DiagnosAI/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── requirements.txt
│   └── ...
│
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DiagnosAI.git
```

---

## 2. Enter the project directory

```bash
cd DiagnosAI
```

---

# 🚀 Backend Setup

Move into the backend directory.

```bash
cd backend
```

---

## Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it.

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure MongoDB Atlas

Create a `.env` file inside the backend directory.

```env
MONGO_URI=your_mongodb_connection_string
```

---

## Start Flask Server

```bash
python app.py
```

Backend runs on

```
http://127.0.0.1:5000
```

---

# 💻 Frontend Setup

Open another terminal.

Move into the frontend folder.

```bash
cd frontend
```

Install dependencies.

```bash
npm install
```

Run React.

```bash
npm start
```

The frontend runs on

```
http://localhost:3000
```

---

# 🚀 How to Use

## Predict Cancer

- Open the application.
- Click **Begin Testing**.
- Enter patient information.
- Fill in the 30 diagnostic parameters.
- Click **Predict**.
- View prediction instantly.

---

## View Patient Records

Navigate to **Past Records**.

You can view:

- Patient Name
- Age
- Gender
- Prediction Result

---

# 🧠 Model Workflow

```text
Patient Information
          │
          ▼
30 Medical Features
          │
          ▼
React Frontend
          │
          ▼
 Flask REST API
          │
          ▼
Data Preprocessing
          │
          ▼
 Logistic Regression
          │
          ▼
Prediction
(Benign / Malignant)
          │
     ┌────┴─────┐
     ▼          ▼
Frontend    MongoDB Atlas
Display      Store Record
```

---

# 📊 Dataset

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

Dataset Characteristics:

- 569 patient samples
- 30 diagnostic features
- Binary classification

Classes:

- Benign
- Malignant

---

# 💻 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict tumor type |
| `/patients` | GET | Retrieve stored patient records |

---

# 🤝 Fork & Run the Project

## Step 1

Fork this repository using the **Fork** button on GitHub.

---

## Step 2

Clone your fork.

```bash
git clone https://github.com/<your-username>/DiagnosAI.git
```

---

## Step 3

Move into the project.

```bash
cd DiagnosAI
```

---

## Step 4

Setup Backend

```bash
cd backend
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run Flask.

```bash
python app.py
```

---

## Step 5

Setup Frontend

```bash
cd frontend
npm install
npm start
```

---

## Step 6

Configure MongoDB Atlas

Update your MongoDB connection string inside the backend configuration.

---

## Step 7

Open the application

```
http://localhost:3000
```

---

# 📸 Screenshots

Add screenshots here.

Example:

```
screenshots/
│
├── home.png
├── testing.png
├── prediction.png
├── records.png
└── database.png
```

---

# 📌 Current Limitations

- UI can be improved further for accessibility and responsiveness.
- Input validation can be strengthened.
- Authentication and user roles are not implemented.
- Single model architecture (future model comparison can be added).

---

# 🌱 Future Improvements

- 🔐 User Authentication
- 👨‍⚕️ Doctor Dashboard
- 📊 Prediction Confidence Visualization
- 📈 Analytics Dashboard
- 📁 Export Patient Reports (PDF)
- 📧 Email Report Generation
- ☁️ Cloud Deployment
- 🐳 Docker Support
- 📱 Mobile Responsive Design
- 🤖 Compare Multiple ML Algorithms
- 📉 Model Performance Dashboard

---

# 👨‍💻 Author

**Avani Kale**

Feel free to contribute, raise issues, or suggest improvements!

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, don't forget to star the repository!

Made with ❤️ using React, Flask, MongoDB & Machine Learning

</div>
1. UI can be improved for better usability and aesthetics
2. Input validation can be strengthened
3. Authentication not implemented (future scope)

🌱 Future Improvements
