import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import BeginTesting from "./components/BeginTesting";
import PastRecords from "./components/PastRecords";
import "./App.css";

function Home() {
  return (
    <div className="home-container">
      <div className="hero-section">
        <h1 className="main-title">Welcome to <span>DiagnosAI</span></h1>
        <p className="tagline">Your Partner in Smart Cancer Detection</p>
        <p className="intro-text">
          DiagnosAI is an innovative platform that leverages artificial intelligence to quickly 
          and accurately classify cancer as cancerous or non-cancerous. Our powerful diagnostic 
          engine uses advanced machine learning algorithms, trained on vast amounts of medical data, 
          to provide fast and reliable results for patients and healthcare providers alike.
        </p>

        <div className="button-group">
          <Link to="/begin" className="btn primary-btn">Begin Testing</Link>
          <Link to="/records" className="btn secondary-btn">Past Records</Link>
        </div>
      </div>

      <div className="features-section">
        <h2 className="section-title">Why Choose DiagnosAI?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>🧠 Trusted AI Expertise</h3>
            <p>Harnesses the latest advances in artificial intelligence for precise cancer classification.</p>
          </div>
          <div className="feature-card">
            <h3>⚡ Fast & Accessible</h3>
            <p>Offers near-instant results, helping users and clinicians make informed decisions more quickly.</p>
          </div>
          <div className="feature-card">
            <h3>📊 Data-Driven Accuracy</h3>
            <p>Trained on large, diverse datasets for robust and reliable performance.</p>
          </div>
          <div className="feature-card">
            <h3>🔒 Privacy-Focused & User-Friendly</h3>
            <p>Keeps your data secure while delivering results in a clear, easy-to-use interface.</p>
          </div>
        </div>

        <p className="closing-text">
          DiagnosAI empowers better cancer detection, supporting early intervention and personalized care. 
          Let our technology help you on your healthcare journey. Try DiagnosAI today!
        </p>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/begin" element={<BeginTesting />} />
        <Route path="/records" element={<PastRecords />} />
      </Routes>
    </Router>
  );
}

export default App;
