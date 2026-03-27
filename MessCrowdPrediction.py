"""
VIT Bhopal Smart Mess Predictor
Flask + SQLite + ML (scikit-learn Gradient Boosting)
Single-file app — backend serves the HTML frontend directly.
"""

import sqlite3, json, os
from flask import Flask, jsonify, request, Response
import numpy as np

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB = "vit_campus_ai.db"

SAMPLE_DATA = [
    # (caterer, hour_pm, food_item, is_popular, wait_time)
    # ── Mayuri ──────────────────────────────────────────
    ('Mayuri', 1,  'Veg Biryani',          1, 35), ('Mayuri', 1,  'Dal Tadka',           0, 15),
    ('Mayuri', 2,  'Veg Biryani',          1, 42), ('Mayuri', 2,  'Dal Tadka',           0, 18),
    ('Mayuri', 7,  'Paneer Butter Masala', 1, 38), ('Mayuri', 7,  'Aloo Gobi',           0, 10),
    ('Mayuri', 8,  'Paneer Butter Masala', 1, 40), ('Mayuri', 8,  'Aloo Gobi',           0, 12),
    ('Mayuri', 12, 'Veg Biryani',          1, 50), ('Mayuri', 12, 'Dal Tadka',           0, 20),
    ('Mayuri', 13, 'Paneer Butter Masala', 1, 45), ('Mayuri', 13, 'Aloo Gobi',           0, 14),
    # ── Safal ───────────────────────────────────────────
    ('Safal',  1,  'Rajma Chawal',         1, 28), ('Safal',  1,  'Khichdi',             0,  8),
    ('Safal',  2,  'Rajma Chawal',         1, 32), ('Safal',  2,  'Khichdi',             0, 10),
    ('Safal',  7,  'Pav Bhaji',            1, 26), ('Safal',  7,  'Mix Veg',             0,  8),
    ('Safal',  8,  'Pav Bhaji',            1, 30), ('Safal',  8,  'Mix Veg',             0, 10),
    ('Safal',  12, 'Rajma Chawal',         1, 38), ('Safal',  12, 'Khichdi',             0, 14),
    ('Safal',  13, 'Pav Bhaji',            1, 35), ('Safal',  13, 'Mix Veg',             0, 12),
    # ── JB ──────────────────────────────────────────────
    ('JB',     1,  'Veg Manchurian',       1, 25), ('JB',     1,  'Lemon Rice',          0,  5),
    ('JB',     2,  'Veg Manchurian',       1, 30), ('JB',     2,  'Lemon Rice',          0,  7),
    ('JB',     7,  'Matar Paneer',         1, 32), ('JB',     7,  'Sev Tamatar',         0, 12),
    ('JB',     8,  'Matar Paneer',         1, 35), ('JB',     8,  'Sev Tamatar',         0, 15),
    ('JB',     12, 'Veg Manchurian',       1, 40), ('JB',     12, 'Lemon Rice',          0, 10),
    ('JB',     13, 'Matar Paneer',         1, 38), ('JB',     13, 'Sev Tamatar',         0, 14),
    # ── AB Catering ─────────────────────────────────────
    ('AB Catering', 1,  'Masala Dosa',     1, 30), ('AB Catering', 1,  'Curd Rice',      0,  5),
    ('AB Catering', 2,  'Masala Dosa',     1, 35), ('AB Catering', 2,  'Curd Rice',      0,  7),
    ('AB Catering', 7,  'Egg Curry',       1, 22), ('AB Catering', 7,  'Idli Sambar',    0,  8),
    ('AB Catering', 8,  'Egg Curry',       1, 25), ('AB Catering', 8,  'Idli Sambar',    0, 10),
    ('AB Catering', 12, 'Masala Dosa',     1, 45), ('AB Catering', 12, 'Curd Rice',      0, 10),
    ('AB Catering', 13, 'Egg Curry',       1, 30), ('AB Catering', 13, 'Idli Sambar',    0, 12),
]


def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mess_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  caterer TEXT, hour_pm INTEGER, food_item TEXT,
                  is_popular INTEGER, wait_time INTEGER)''')
    c.execute("SELECT COUNT(*) FROM mess_logs")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO mess_logs(caterer,hour_pm,food_item,is_popular,wait_time) VALUES(?,?,?,?,?)",
                      SAMPLE_DATA)
        conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  ML MODEL  (trained once at startup, retrained after feedback)
# ─────────────────────────────────────────────────────────────────────────────
le_caterer   = LabelEncoder()
le_food      = LabelEncoder()
model        = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
model_ready  = False


def load_training_data():
    conn = get_conn()
    rows = conn.execute("SELECT caterer, hour_pm, food_item, is_popular, wait_time FROM mess_logs").fetchall()
    conn.close()
    return rows


def train_model():
    global model_ready
    rows = load_training_data()
    caterers  = [r["caterer"]   for r in rows]
    foods     = [r["food_item"] for r in rows]
    hours     = [r["hour_pm"]   for r in rows]
    popular   = [r["is_popular"] for r in rows]
    targets   = [r["wait_time"] for r in rows]

    le_caterer.fit(caterers)
    le_food.fit(foods)

    X = np.column_stack([
        le_caterer.transform(caterers),
        hours,
        le_food.transform(foods),
        popular,
    ])
    y = np.array(targets, dtype=float)
    model.fit(X, y)
    model_ready = True


def ml_predict(caterer, hour, food_item, is_popular):
    if not model_ready:
        return None
    try:
        cat_enc  = le_caterer.transform([caterer])[0]
        food_enc = le_food.transform([food_item])[0]
        X = np.array([[cat_enc, hour, food_enc, is_popular]])
        pred = float(model.predict(X)[0])
        return max(2.0, round(pred, 1))
    except Exception:
        return None


def rule_based_predict(conn, caterer, hour, food_item):
    """Fallback weighted-regression logic (original approach)."""
    c = conn.cursor()
    c.execute("SELECT wait_time FROM mess_logs WHERE caterer=? AND hour_pm=?", (caterer, hour))
    hist = [r[0] for r in c.fetchall()]
    base = sum(hist) / len(hist) if hist else 20
    c.execute("SELECT is_popular FROM mess_logs WHERE food_item=?", (food_item,))
    row = c.fetchone()
    pop = row[0] if row else 0
    pred = base + (15 if pop else -5)
    return max(2.0, round(pred, 1)), round(base, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/api/messes")
def api_messes():
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT caterer FROM mess_logs").fetchall()
    conn.close()
    return jsonify([r["caterer"] for r in rows])


@app.route("/api/menu")
def api_menu():
    caterer = request.args.get("caterer", "")
    conn = get_conn()
    items = conn.execute("SELECT DISTINCT food_item FROM mess_logs WHERE caterer=?", (caterer,)).fetchall()
    hours = conn.execute("SELECT DISTINCT hour_pm   FROM mess_logs WHERE caterer=?", (caterer,)).fetchall()
    conn.close()
    return jsonify({
        "items": [r["food_item"] for r in items],
        "hours": sorted([r["hour_pm"]  for r in hours]),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body     = request.get_json(force=True)
    caterer  = body.get("caterer", "")
    hour     = int(body.get("hour", 1))
    food     = body.get("food", "")

    conn = get_conn()
    row = conn.execute("SELECT is_popular FROM mess_logs WHERE food_item=? LIMIT 1", (food,)).fetchone()
    is_pop = row["is_popular"] if row else 0

    ml_pred = ml_predict(caterer, hour, food, is_pop)

    _, base_avg = rule_based_predict(conn, caterer, hour, food)

    # Hourly trend across all hours for this caterer+food
    trend_rows = conn.execute(
        "SELECT hour_pm, AVG(wait_time) as avg_w FROM mess_logs WHERE caterer=? AND food_item=? GROUP BY hour_pm ORDER BY hour_pm",
        (caterer, food)
    ).fetchall()
    trend = [{"hour": r["hour_pm"], "avg": round(r["avg_w"], 1)} for r in trend_rows]

    # Popularity vs non-popular comparison for this mess
    pop_rows = conn.execute(
        "SELECT is_popular, AVG(wait_time) as avg_w FROM mess_logs WHERE caterer=? AND hour_pm=? GROUP BY is_popular",
        (caterer, hour)
    ).fetchall()
    pop_data = {r["is_popular"]: round(r["avg_w"], 1) for r in pop_rows}

    conn.close()

    final = ml_pred if ml_pred is not None else (base_avg + (15 if is_pop else -5))
    final = max(2.0, round(final, 1))

    # Confidence band ±15%
    lo = round(final * 0.85, 1)
    hi = round(final * 1.15, 1)

    return jsonify({
        "prediction":   final,
        "base_avg":     base_avg,
        "ml_used":      ml_pred is not None,
        "is_popular":   bool(is_pop),
        "confidence_lo": lo,
        "confidence_hi": hi,
        "trend":        trend,
        "pop_compare":  pop_data,
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Accept real wait-time feedback and retrain."""
    body    = request.get_json(force=True)
    caterer = body["caterer"]
    hour    = int(body["hour"])
    food    = body["food"]
    actual  = int(body["actual_wait"])

    conn = get_conn()
    row  = conn.execute("SELECT is_popular FROM mess_logs WHERE food_item=? LIMIT 1", (food,)).fetchone()
    pop  = row["is_popular"] if row else 0
    conn.execute(
        "INSERT INTO mess_logs(caterer,hour_pm,food_item,is_popular,wait_time) VALUES(?,?,?,?,?)",
        (caterer, hour, food, pop, actual)
    )
    conn.commit()
    conn.close()
    train_model()   # retrain with new data
    return jsonify({"status": "ok", "message": "Model retrained with your feedback!"})


@app.route("/api/stats")
def api_stats():
    caterer = request.args.get("caterer", "")
    conn = get_conn()
    rows = conn.execute(
        "SELECT hour_pm, AVG(wait_time) as avg_w, COUNT(*) as cnt FROM mess_logs WHERE caterer=? GROUP BY hour_pm ORDER BY hour_pm",
        (caterer,)
    ).fetchall()
    conn.close()
    return jsonify([{"hour": r["hour_pm"], "avg": round(r["avg_w"],1), "count": r["cnt"]} for r in rows])


# ── Serve the embedded HTML frontend ─────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VIT Mess AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#0b0c10;--surface:#13151c;--card:#1a1d28;--border:#252836;
    --accent:#f97316;--accent2:#fb923c;--green:#22c55e;--red:#ef4444;
    --text:#f1f5f9;--muted:#64748b;--dim:#94a3b8;
    --grad:linear-gradient(135deg,#f97316,#fb923c,#fbbf24);
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}

  /* ── noise texture overlay ── */
  body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");opacity:.4;pointer-events:none;z-index:0}

  /* ── glow blobs ── */
  .blob{position:fixed;border-radius:50%;filter:blur(120px);pointer-events:none;z-index:0;opacity:.15}
  .blob1{width:600px;height:600px;background:#f97316;top:-200px;left:-200px}
  .blob2{width:500px;height:500px;background:#7c3aed;bottom:-150px;right:-150px}

  .wrap{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:2rem 1.5rem}

  /* ── header ── */
  header{text-align:center;padding:3rem 0 2rem}
  .badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:var(--accent);font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;padding:.35rem .9rem;border-radius:999px;margin-bottom:1.5rem}
  .badge span{width:6px;height:6px;background:var(--accent);border-radius:50%;animation:pulse 1.5s ease-in-out infinite}
  h1{font-family:'Syne',sans-serif;font-size:clamp(2.2rem,5vw,3.8rem);font-weight:800;line-height:1.05;letter-spacing:-.03em}
  h1 em{font-style:normal;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .sub{color:var(--muted);margin-top:.75rem;font-size:1rem;font-weight:300}

  /* ── cards ── */
  .card{background:var(--card);border:1px solid var(--border);border-radius:1rem;padding:1.75rem}
  .card+.card{margin-top:1.25rem}
  .card-label{font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem}

  /* ── step grid ── */
  .step-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  @media(max-width:600px){.step-grid{grid-template-columns:1fr}}

  /* ── select / buttons ── */
  select,button{font-family:'DM Sans',sans-serif}
  .sel-wrap{position:relative}
  .sel-wrap svg{position:absolute;right:.9rem;top:50%;transform:translateY(-50%);pointer-events:none;color:var(--muted)}
  select{width:100%;appearance:none;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:.75rem 2.5rem .75rem 1rem;border-radius:.6rem;font-size:.95rem;cursor:pointer;transition:.2s}
  select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(249,115,22,.15)}

  .btn{display:flex;align-items:center;justify-content:center;gap:.5rem;width:100%;background:var(--grad);color:#000;font-weight:700;font-size:1rem;border:none;border-radius:.7rem;padding:.85rem;cursor:pointer;transition:.2s;margin-top:1.25rem;font-family:'Syne',sans-serif;letter-spacing:.02em}
  .btn:hover{filter:brightness(1.1);transform:translateY(-1px)}
  .btn:active{transform:translateY(0)}
  .btn:disabled{opacity:.4;cursor:not-allowed;filter:none;transform:none}

  /* ── result panel ── */
  #result{display:none}
  .result-top{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap}
  .wait-num{font-family:'Syne',sans-serif;font-size:4rem;font-weight:800;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}
  .wait-label{font-size:.8rem;color:var(--muted);margin-top:.2rem}
  .pill{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:999px;font-size:.75rem;font-weight:600}
  .pill.pop{background:rgba(249,115,22,.15);color:var(--accent);border:1px solid rgba(249,115,22,.25)}
  .pill.quiet{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.25)}
  .pill.ml{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.25)}
  .confidence{margin-top:.5rem;font-size:.85rem;color:var(--dim)}

  /* ── mini stats bar ── */
  .stats-row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1rem}
  .stat-box{background:var(--surface);border:1px solid var(--border);border-radius:.6rem;padding:1rem;text-align:center}
  .stat-val{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700}
  .stat-key{font-size:.7rem;color:var(--muted);margin-top:.2rem;text-transform:uppercase;letter-spacing:.08em}

  /* ── canvas charts ── */
  .chart-wrap{margin-top:1rem;position:relative;height:180px}
  canvas{border-radius:.5rem}

  /* ── feedback ── */
  .fb-row{display:flex;gap:.75rem;margin-top:1rem;align-items:center}
  .fb-row input{flex:1;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:.65rem 1rem;border-radius:.6rem;font-size:.95rem;font-family:'DM Sans',sans-serif}
  .fb-row input:focus{outline:none;border-color:var(--accent)}
  .fb-btn{background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.3);color:var(--accent);padding:.65rem 1.25rem;border-radius:.6rem;font-weight:600;cursor:pointer;white-space:nowrap;font-family:'DM Sans',sans-serif;transition:.2s}
  .fb-btn:hover{background:rgba(249,115,22,.25)}
  .fb-msg{font-size:.8rem;color:var(--green);margin-top:.5rem;min-height:1.2em}

  /* ── toast ── */
  #toast{position:fixed;bottom:1.5rem;left:50%;transform:translateX(-50%) translateY(4rem);background:var(--card);border:1px solid var(--border);color:var(--text);padding:.75rem 1.5rem;border-radius:.7rem;font-size:.9rem;transition:.35s;z-index:999;opacity:0}
  #toast.show{transform:translateX(-50%) translateY(0);opacity:1}

  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
  .fade-up{animation:fadeUp .45s ease both}
</style>
</head>
<body>
<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div id="toast"></div>

<div class="wrap">
  <header>
    <div class="badge"><span></span>Live AI Predictor</div>
    <h1>VIT Mess<br/><em>Wait-Time AI</em></h1>
    <p class="sub">Gradient Boosting ML · SQLite · Real-time Feedback Loop</p>
  </header>

  <!-- SELECTOR CARD -->
  <div class="card fade-up">
    <div class="card-label">Configure your prediction</div>
    <div class="step-grid">
      <div>
        <div class="card-label" style="margin-bottom:.5rem">1 · Mess</div>
        <div class="sel-wrap">
          <select id="selMess"><option value="">— Select mess —</option></select>
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
        </div>
      </div>
      <div>
        <div class="card-label" style="margin-bottom:.5rem">2 · Hour (PM)</div>
        <div class="sel-wrap">
          <select id="selHour" disabled><option value="">— Select mess first —</option></select>
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
        </div>
      </div>
      <div style="grid-column:1/-1">
        <div class="card-label" style="margin-bottom:.5rem">3 · Dish</div>
        <div class="sel-wrap">
          <select id="selFood" disabled><option value="">— Select mess first —</option></select>
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
        </div>
      </div>
    </div>
    <button class="btn" id="btnPredict" disabled>
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35M11 8v3l2 2"/></svg>
      Predict Wait Time
    </button>
  </div>

  <!-- RESULT CARD -->
  <div class="card fade-up" id="result">
    <div class="card-label">AI Prediction Result</div>
    <div class="result-top">
      <div>
        <div class="wait-num" id="rMin">—</div>
        <div class="wait-label">minutes estimated wait</div>
        <div style="margin-top:.6rem;display:flex;gap:.5rem;flex-wrap:wrap" id="pills"></div>
        <div class="confidence" id="confBand"></div>
      </div>
      <div class="stat-box" style="min-width:130px;text-align:center">
        <div class="stat-val" id="rBase">—</div>
        <div class="stat-key">Base avg (min)</div>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-val" id="rLo" style="color:var(--green)">—</div>
        <div class="stat-key">Best case</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" id="rHi" style="color:var(--red)">—</div>
        <div class="stat-key">Worst case</div>
      </div>
    </div>

    <!-- Hourly trend chart -->
    <div class="card-label" style="margin-top:1.5rem">Hourly Wait Trend · <span id="chartMess"></span></div>
    <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>

    <!-- Popular vs quiet chart -->
    <div class="card-label" style="margin-top:1.5rem">Popular vs Quiet Dishes at This Hour</div>
    <div class="chart-wrap" style="height:140px"><canvas id="chartPop"></canvas></div>

    <!-- Feedback -->
    <div class="card-label" style="margin-top:1.5rem">Help the AI Learn · Submit Actual Wait</div>
    <div class="fb-row">
      <input type="number" id="fbInput" placeholder="e.g. 22" min="1" max="120"/>
      <button class="fb-btn" id="btnFb">Submit</button>
    </div>
    <div class="fb-msg" id="fbMsg"></div>
  </div>
</div>

<script>
const BASE = '';
let trendChart, popChart;
let lastSel = {mess:'',hour:0,food:''};

// ── helpers ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function toast(msg){
  const t=$('toast'); t.textContent=msg; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 2800);
}

function mkChart(id, labels, data, color='#f97316', type='bar'){
  const ctx=document.getElementById(id).getContext('2d');
  const existing = type==='bar'? trendChart : popChart;
  if(type==='bar'&&trendChart){trendChart.destroy();}
  if(type==='barP'&&popChart){popChart.destroy();}
  const cfg={
    type:'bar',
    data:{labels,datasets:[{data,backgroundColor:color,borderRadius:6,borderSkipped:false}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.parsed.y} min`}}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:11}}},
        y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:11}},beginAtZero:true}
      }
    }
  };
  const ch=new Chart(ctx,cfg);
  if(type==='bar') trendChart=ch; else popChart=ch;
}

// ── load messes ──────────────────────────────────────────────────────────────
async function loadMesses(){
  const r=await fetch('/api/messes');
  const data=await r.json();
  const sel=$('selMess');
  data.forEach(m=>{const o=document.createElement('option');o.value=o.textContent=m;sel.appendChild(o);});
}

// ── mess changed ─────────────────────────────────────────────────────────────
$('selMess').addEventListener('change', async function(){
  const mess=this.value;
  if(!mess) return;
  const r=await fetch(`/api/menu?caterer=${encodeURIComponent(mess)}`);
  const d=await r.json();

  const sh=$('selHour');
  sh.innerHTML='';
  d.hours.forEach(h=>{const o=document.createElement('option');o.value=h;o.textContent=`${h} PM`;sh.appendChild(o);});
  sh.disabled=false;

  const sf=$('selFood');
  sf.innerHTML='';
  d.items.forEach(f=>{const o=document.createElement('option');o.value=o.textContent=f;sf.appendChild(o);});
  sf.disabled=false;

  $('btnPredict').disabled=false;
});

// ── predict ──────────────────────────────────────────────────────────────────
$('btnPredict').addEventListener('click', async()=>{
  const mess=$('selMess').value;
  const hour=parseInt($('selHour').value);
  const food=$('selFood').value;
  if(!mess||!food) return;

  lastSel={mess,hour,food};
  $('btnPredict').disabled=true;
  $('btnPredict').textContent='Predicting…';

  const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({caterer:mess,hour,food})});
  const d=await r.json();

  $('btnPredict').disabled=false;
  $('btnPredict').innerHTML=`<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35M11 8v3l2 2"/></svg> Predict Wait Time`;

  // populate
  $('rMin').textContent=d.prediction;
  $('rBase').textContent=d.base_avg;
  $('rLo').textContent=d.confidence_lo;
  $('rHi').textContent=d.confidence_hi;
  $('confBand').textContent=`95% confidence: ${d.confidence_lo}–${d.confidence_hi} min`;
  $('chartMess').textContent=mess;
  $('fbMsg').textContent='';

  // pills
  const pills=$('pills');
  pills.innerHTML='';
  if(d.is_popular) pills.innerHTML+=`<span class="pill pop">🔥 Popular dish</span>`;
  else pills.innerHTML+=`<span class="pill quiet">✅ Quiet choice</span>`;
  if(d.ml_used) pills.innerHTML+=`<span class="pill ml">🤖 ML Model</span>`;

  // trend chart
  if(d.trend.length){
    mkChart('chartTrend',d.trend.map(t=>`${t.hour} PM`),d.trend.map(t=>t.avg),'rgba(249,115,22,.75)','bar');
  }

  // pop chart
  const popLabels=['🥬 Quiet Dish','🌶️ Popular Dish'];
  const popVals=[d.pop_compare[0]||0, d.pop_compare[1]||0];
  mkChart('chartPop',popLabels,popVals,['rgba(34,197,94,.7)','rgba(249,115,22,.75)'],'barP');

  $('result').style.display='block';
  $('result').scrollIntoView({behavior:'smooth',block:'nearest'});
});

// ── feedback ─────────────────────────────────────────────────────────────────
$('btnFb').addEventListener('click', async()=>{
  const val=parseInt($('fbInput').value);
  if(!val||val<1){toast('Enter a valid wait time (1–120 min)');return;}
  const r=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({caterer:lastSel.mess,hour:lastSel.hour,food:lastSel.food,actual_wait:val})});
  const d=await r.json();
  $('fbMsg').textContent='✅ '+d.message;
  $('fbInput').value='';
  toast('Model updated with your data!');
});

// ── Chart.js via CDN ─────────────────────────────────────────────────────────
const s=document.createElement('script');
s.src='https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js';
document.head.appendChild(s);

loadMesses();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_db()
    train_model()
    print("\n🍽️  VIT Mess AI → http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)