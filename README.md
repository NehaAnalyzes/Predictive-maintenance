# 🔧 Predictive Maintenance System

> End-to-end machine failure prediction system using IoT sensor data — built with Random Forest, deployed via FastAPI + Streamlit.


---

## 🔴 Live Demo

| Service | URL |
|---|---|
| FastAPI (REST API + Swagger docs) | `https://your-api.onrender.com/docs` |
| Streamlit Dashboard | `https://your-app.streamlit.app` |

> Replace above with your live URLs after deployment

---

## 📌 Problem Statement

Unplanned machine failures cost manufacturers millions in downtime. This project builds a real-time predictive maintenance system that takes live sensor readings and predicts whether a machine is likely to fail — before it does.

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository

- 10,000 rows × 14 columns
- 5 sensor features: air temperature, process temperature, rotational speed, torque, tool wear
- Target: binary machine failure (0 = normal, 1 = failure)
- Class imbalance: 96.6% normal / 3.4% failure → handled with SMOTE

---

## 🏗️ Project Architecture

```
Sensor Inputs (6 features)
        ↓
  Feature Engineering
  (power, temp_diff, strain)
        ↓
  StandardScaler
        ↓
  Random Forest Classifier
  (threshold = 0.65)
        ↓
  FastAPI  →  POST /predict
        ↓
  Streamlit Dashboard
```

---

## 📁 Project Structure

```
Predictive-maintenance/
├── api/
│   └── main.py              ← FastAPI server
├── dashboard/
│   ├── app.py               ← Streamlit dashboard
│   └── index.html           ← Standalone HTML dashboard
├── phase2_and_3.py          ← EDA + preprocessing
├── phase4_modeling.py       ← Model training + threshold tuning
├── model.pkl                ← Trained Random Forest model
├── scaler.pkl               ← Fitted StandardScaler
├── threshold.pkl            ← Tuned decision threshold (0.65)
├── Dockerfile               ← Container for FastAPI
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

Three features engineered from raw sensor data:

| Feature | Formula | Why |
|---|---|---|
| `power` | torque × (rpm × 2π/60) | Correlates strongly with PWF failures |
| `temp_diff` | process_temp − air_temp | Low diff triggers heat dissipation failures |
| `strain` | tool_wear × torque | Combined mechanical stress indicator |

---

## 🤖 Model Results

| Model | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.30 | 0.18 | 0.88 | 0.94 |
| Random Forest | 0.70 | 0.59 | 0.84 | 0.98 |
| Gradient Boosting | 0.65 | 0.52 | 0.88 | 0.98 |
| **Random Forest (tuned)** | **0.81** | **0.83** | **0.79** | **0.98** |

**Key insight:** Default threshold (0.50) gave F1=0.70. Threshold tuning to 0.65 improved F1 to **0.81** — a 16% gain with no retraining.

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/NehaAnalyzes/Predictive-maintenance.git
cd Predictive-maintenance
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** (or skip — pkl files are included)
```bash
python phase2_and_3.py
python phase4_modeling.py
```

**4. Start the API**
```bash
python -m uvicorn api.main:app --reload
```
API runs at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

**5. Start the dashboard** (new terminal)
```bash
python -m streamlit run dashboard/app.py
```
Dashboard runs at `http://localhost:8501`

---

## 🐳 Run with Docker

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

---

## 🔌 API Usage

**POST** `/predict`

```json
{
  "Type": 0,
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

**Response**
```json
{
  "prediction": 0,
  "result": "✅ NORMAL OPERATION",
  "failure_probability": 0.03,
  "threshold_used": 0.65
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| ML | scikit-learn, imbalanced-learn (SMOTE) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Serialization | joblib |
| Container | Docker |
| Deployment | Render (API), Streamlit Cloud (dashboard) |

---

## 👩‍💻 Author

**Neha** — [GitHub](https://github.com/NehaAnalyzes)

---

## 📄 License

MIT License — free to use and modify.
