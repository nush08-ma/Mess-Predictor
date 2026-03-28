## What is VIT Mess Predictor?

A single-file Flask app that predicts how long you'll wait in the mess queue at any mess from your campus ( here, VIT Bhopal) before you even leave your hostel room. It uses a Gradient Boosting ML model trained on historical caterer data, with a live feedback loop so the model improves over time.

## Features

**ML Prediction** : `GradientBoostingRegressor` (200 estimators) trained on caterer × hour × dish data.

**Hourly Trend Chart**: See wait time patterns across the day for any mess.

**Popular vs Quiet** : Side-by-side chart comparing popular vs quiet dish wait times.

**Confidence Band** : 95% confidence interval (best case / worst case).

**Live Feedback Loop** :Submit your actual wait so model retrains instantly.

**Zero Setup Frontend** : Flask serves the entire HTML/CSS/JS UI — no separate frontend needed.

**4 Caterers Covered** : Mayuri , Safal , JB , AB Catering.


##  Quick Start


### 1. Clone or download mess.py

### 2. Install dependencies 
Install flask,scikit-learn,numpy

### 3. Run
python mess.py

### 4. Open browser
http://127.0.0.1:5000


## How the ML Works


1 .Input:  caterer + hour (PM) + food item + is_popular flag

2 .LabelEncoder → numerical encoding

3 .GradientBoostingRegressor (n=200, lr=0.08, depth=4)

4 .Output: predicted wait time in minutes

5 .Confidence band: ±20% lower / +30% upper


Fallback to rule-based regression if ML isn't ready or encounters an unseen label.


## Project Structure


mess.py              : entire app (backend + ML + frontend)

vit_campus_ai.db     : SQLite DB (auto-created on first run)


Single-file architecture — no build step, no config, no env files needed.

## Common Issues

**`ModuleNotFoundError: No module named 'flask'` after installing**

Windows has multiple Python installs. `pip` and `python` may point to different versions.


### Fix: always use -m pip with the exact python that runs your script
python3.13 -m pip install flask scikit-learn numpy


## Tech Stack

**Backend** — Python , Flask , SQLite

**ML** — scikit-learn `GradientBoostingRegressor` , `LabelEncoder` , NumPy

**Frontend** — Vanilla HTML/CSS/JS , Chart.js (CDN)

**DB** — SQLite (auto-seeded with 48 sample records across 4 caterers)
