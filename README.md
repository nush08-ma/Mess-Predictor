<div align="center">

<!-- Wave Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=VIT%20Mess%20Predictor&fontSize=48&fontColor=fff&animation=twinkling&fontAlignY=35&desc=AI-powered%20mess%20wait%20time%20prediction%20for%20VIT%20Bhopal&descAlignY=58&descSize=18" />

<!-- Typing SVG -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F97316&center=true&vCenter=true&width=600&lines=Predict+your+mess+wait+time+%F0%9F%8D%BD%EF%B8%8F;Powered+by+Gradient+Boosting+ML+%F0%9F%A4%96;Flask+%2B+SQLite+%2B+scikit-learn;VIT+Bhopal+%E2%80%A2+Hostel+Life+Optimized+%E2%9C%85;Submit+feedback+%E2%86%92+model+retrains+live!" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualizations-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

</div>

---

## 🧑‍💻 About Me

```python
developer = {
    "name"        : "Anushka",
    "university"  : "VIT Bhopal University",
    "batch"       : "B.Tech CSE · 2025–2029",
    "based_in"    : "VIT Bhopal Campus 🇮🇳",
    "interests"   : ["Machine Learning", "Flask & Backend Dev", "Solving Real Problems"],
    "currently"   : "Building AI tools that actually solve hostel problems 🍽️",
    "motto"       : "Why wait in line when you can predict it? ✨",
}
```

---

## 🍽️ What is VIT Mess Predictor?

A **single-file Flask app** that predicts how long you'll wait in the mess queue at VIT Bhopal — before you even leave your hostel room. It uses a **Gradient Boosting ML model** trained on historical caterer data, with a live feedback loop so the model improves over time.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **ML Prediction** | `GradientBoostingRegressor` (200 estimators) trained on caterer × hour × dish data |
| 📊 **Hourly Trend Chart** | See wait time patterns across the day for any mess |
| 🥬 **Popular vs Quiet** | Side-by-side chart comparing popular vs quiet dish wait times |
| 🎯 **Confidence Band** | 95% confidence interval (best case / worst case) |
| 🔁 **Live Feedback Loop** | Submit your actual wait → model retrains instantly |
| 🏠 **Zero Setup Frontend** | Flask serves the entire HTML/CSS/JS UI — no separate frontend needed |
| 🍛 **4 Caterers Covered** | Mayuri · Safal · JB · AB Catering |

---

## 🚀 Quick Start

```bash
# 1. Clone or download mess.py

# 2. Install dependencies (use the SAME Python that runs the script!)
python3.13 -m pip install flask scikit-learn numpy

# 3. Run
python mess.py

# 4. Open browser
# → http://127.0.0.1:5000
```

> ⚠️ **Windows users:** If you get `ModuleNotFoundError: flask`, use `python3.13 -m pip install flask` instead of plain `pip install flask`. See [this common issue](#-common-issues).

---

## 🧠 How the ML Works

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

## 📁 Project Structure

```
mess.py              ← entire app (backend + ML + frontend)
vit_campus_ai.db     ← SQLite DB (auto-created on first run)
```

Single-file architecture — no build step, no config, no env files needed.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/messes` | List all available caterers |
| `GET` | `/api/menu?caterer=X` | Get food items + hours for a caterer |
| `POST` | `/api/predict` | Get ML wait prediction + charts data |
| `POST` | `/api/feedback` | Submit actual wait → retrain model |

---

## ⚠️ Common Issues

**`ModuleNotFoundError: No module named 'flask'` after installing**

Windows has multiple Python installs. `pip` and `python` may point to different versions.

```powershell
# Fix: always use -m pip with the exact python that runs your script
python3.13 -m pip install flask scikit-learn numpy
```

---

## 🛠️ Tech Stack

- **Backend** — Python · Flask · SQLite
- **ML** — scikit-learn `GradientBoostingRegressor` · `LabelEncoder` · NumPy
- **Frontend** — Vanilla HTML/CSS/JS · Chart.js (CDN)
- **DB** — SQLite (auto-seeded with 48 sample records across 4 caterers)

---

## 💬 Dev Quote

<div align="center">

![Quote](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,30&height=160&section=footer&text=Thanks%20for%20visiting!&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=65&desc=Made%20with%20%F0%9F%8D%9B%20and%20late-night%20hunger%20pangs%20%C2%B7%20Anushka%20%C2%B7%20VIT%20Bhopal&descAlignY=85&descSize=14" />

</div>
