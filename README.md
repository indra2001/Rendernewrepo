# 🚀 **BlinkChart Developer Guide**

### Version: v1.0 (Clean Starter Edition)

**Author:** LH IdeaCraft Inc.
**Maintainer:** Vinay Jay (Founder & Technical Architect)

---

## 🧭 Overview

**BlinkChart** (aka MagicChart) is a SaaS application that transforms Excel or CSV data into beautiful, AI-suggested charts.
This version includes a fully functional **FastAPI backend** and **React + Vite frontend**.

It’s designed for:

* Students & teachers to visualize data easily
* Small businesses to generate instant insights
* Dev teams to extend into a full SaaS (Free / Pro tiers)

---

## 🏗️ Folder Structure

```
blinkchart/
├── frontend/                  # React + Vite UI
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── components/
│       │   ├── Chart.jsx
│       │   └── Uploader.jsx
│       └── pages/
│           ├── Upload.jsx
│           ├── Gallery.jsx
│           └── About.jsx
│
├── service/                   # FastAPI backend
│   ├── fastapi_app.py
│   └── requirements.txt
│
├── Dockerfile.frontend
├── Dockerfile.backend
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Backend — FastAPI

```bash
cd service
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn fastapi_app:app --reload --host 127.0.0.1 --port 8080
```

**Verify**

```
http://127.0.0.1:8080/healthz
→ {"status": "ok"}
```

---

### 2️⃣ Frontend — React + Vite

```bash
cd frontend
npm install
npm run dev
```

**Access UI:**
👉 [http://localhost:5173](http://localhost:5173)

---

### 3️⃣ Docker (Optional, Full Stack)

```bash
docker compose up --build
```

**Services:**

* Frontend → [http://localhost:5173](http://localhost:5173)
* Backend → [http://localhost:8080/healthz](http://localhost:8080/healthz)

---

## 🧩 Core Features

| Module        | Function                                            |
| ------------- | --------------------------------------------------- |
| `/api/upload` | Accepts CSV/XLSX files and infers chart suggestions |
| `/api/render` | Renders selected chart (bar/line/pie/scatter)       |
| `/healthz`    | Simple health check endpoint                        |
| Frontend      | Upload UI, chart controls, Plotly rendering         |

---

## 🧠 Development Flow

1. **Upload a dataset** (CSV/XLSX) from the UI.
2. **Backend processes** file → detects columns → suggests chart types.
3. **User selects** chart type and columns.
4. **Frontend renders** using Plotly.js.
5. (Pro tier only: Export charts as PNG/PDF via Kaleido in next phase.)

---

## 🔮 Roadmap

| Phase | Milestone            | Description                      |
| ----- | -------------------- | -------------------------------- |
| 1     | ✅ Core MVP           | Working upload + chart rendering |
| 2     | 🧱 Templates Gallery | Preloaded test datasets          |
| 3     | 🔐 Auth System       | Free vs Pro users                |
| 4     | 📤 Export & Share    | PNG/PDF exports, share links     |
| 5     | ☁️ SaaS Deployment   | Fly.io / Railway / AWS rollout   |

---

## 👩‍💻 Developer Notes

* **Tech stack:** React 18, Vite, Plotly.js, FastAPI, Python 3.11+
* **Frontend port:** `5173`
* **Backend port:** `8080`
* **CORS:** Enabled for all origins (for local testing)
* **Data limit:** 10,000 rows per upload (configurable)
* **Security:** No file storage — data stays in memory for privacy

---

## 🧰 Troubleshooting

| Issue               | Fix                                                       |
| ------------------- | --------------------------------------------------------- |
| `npm not found`     | Install Node.js LTS from [nodejs.org](https://nodejs.org) |
| Backend won’t start | Ensure Python 3.11+ and `pip install -r requirements.txt` |
| Chart not showing   | Check console → ensure `/api/render` returns valid JSON   |
| CORS error          | Update `allow_origins=["*"]` in `fastapi_app.py`          |

---

## 👥 Team Guidelines

* Always commit via feature branches (`feature/ui`, `feature/api`)
* Test locally before merging to main
* Use consistent naming conventions (`snake_case` for backend, `camelCase` for frontend)
* All UI strings must be user-friendly and error-safe
* Add screenshots for UI PRs

---

## 🧡 Credits

**Founder & Architect:** Vinay Jay
**Studio:** LH IdeaCraft Inc. (Texas, USA)
**Mission:** To craft human-centered innovations that strengthen communities and create lasting value for generations to come.

---

Would you like me to generate this as a **Word and Markdown** file pair (so you can share with your offshore devs via Teams or ClickUp)?
