## What is VIT Mess Predictor?

A single-file Flask app that predicts how long you'll wait in the mess queue at any mess from your campus ( here, VIT Bhopal) before you even leave your hostel room. It uses a Gradient Boosting ML model trained on historical caterer data, with a live feedback loop so the model improves over time.

## Features

| Feature | Details |
|---|---|
| **ML Prediction** | `GradientBoostingRegressor` (200 estimators) trained on caterer × hour × dish data |
| **Hourly Trend Chart** | See wait time patterns across the day for any mess |
| **Popular vs Quiet** | Side-by-side chart comparing popular vs quiet dish wait times |
| **Confidence Band** | 95% confidence interval (best case / worst case) |
| **Live Feedback Loop** | Submit your actual wait → model retrains instantly |
| **Zero Setup Frontend** | Flask serves the entire HTML/CSS/JS UI — no separate frontend needed |
| **4 Caterers Covered** | Mayuri · Safal · JB · AB Catering |

---

##  Quick Start

```bash
# 1. Clone or download mess.py

# 2. Install dependencies (use the SAME Python that runs the script!)
python3.13 -m pip install flask scikit-learn numpy

# 3. Run
python mess.py

# 4. Open browser
# → http://127.0.0.1:5000
```

> **Windows users:** If you get `ModuleNotFoundError: flask`, use `python3.13 -m pip install flask` instead of plain `pip install flask`. See [this common issue](#-common-issues).

---

## How the ML Works

```
Input:  caterer + hour (PM) + food item + is_popular flag
   ↓
LabelEncoder → numerical encoding
   ↓
GradientBoostingRegressor (n=200, lr=0.08, depth=4)
   ↓
Output: predicted wait time in minutes
   ↓
Confidence band: ±20% lower / +30% upper
```

Fallback to **rule-based regression** if ML isn't ready or encounters an unseen label.

---

## Project Structure

```
mess.py              ← entire app (backend + ML + frontend)
vit_campus_ai.db     ← SQLite DB (auto-created on first run)
```

Single-file architecture — no build step, no config, no env files needed.

---

##API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/messes` | List all available caterers |
| `GET` | `/api/menu?caterer=X` | Get food items + hours for a caterer |
| `POST` | `/api/predict` | Get ML wait prediction + charts data |
| `POST` | `/api/feedback` | Submit actual wait → retrain model |

---

## Common Issues

**`ModuleNotFoundError: No module named 'flask'` after installing**

Windows has multiple Python installs. `pip` and `python` may point to different versions.

```powershell
# Fix: always use -m pip with the exact python that runs your script
python3.13 -m pip install flask scikit-learn numpy
```

---

## Tech Stack

- **Backend** — Python · Flask · SQLite
- **ML** — scikit-learn `GradientBoostingRegressor` · `LabelEncoder` · NumPy
- **Frontend** — Vanilla HTML/CSS/JS · Chart.js (CDN)
- **DB** — SQLite (auto-seeded with 48 sample records across 4 caterers)
