🧬 DiagnosAI – AI-Powered Cancer Detection Web App

DiagnosAI is a full-stack web application that uses Machine Learning to predict whether a tumor is Benign or Malignant based on medical input parameters.
It provides an easy-to-use interface for testing new patients and maintaining past patient records.

🚀 Features
🔍 Cancer Prediction using a trained Machine Learning model
🧠 Uses 30 real medical parameters (Breast Cancer dataset)
🖥️ Interactive React Frontend
⚙️ Flask REST API Backend
🗄️ MongoDB Atlas for storing patient records
📜 View Past Patient Records (Name, Age, Gender, Prediction)
🌐 Cross-origin support using Flask-CORS

🏗️ Tech Stack
Frontend:
React.js
Tailwind CSS
Fetch API

Backend:
Python
Flask
Flask-CORS
NumPy
Scikit-Learn

Database:
MongoDB Atlas (Cloud)

Machine Learning:
Logistic Regression (trained & saved using joblib)
Breast Cancer Wisconsin Dataset

🧠 How It Works (Workflow)

1. User enters patient details and 30 diagnostic parameters on the Begin Testing page
2. Frontend sends data to Flask /predict API
3. ML model predicts Benign or Malignant
4. Result is:
   Returned to frontend
  Stored in MongoDB Atlas (diagnosis collection)
5. Past Records page fetches and displays stored patient data

📌 Current Limitations

1. UI can be improved for better usability and aesthetics
2. Input validation can be strengthened
3. Authentication not implemented (future scope)

🌱 Future Improvements
