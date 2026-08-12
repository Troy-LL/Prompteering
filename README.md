# Prompteering

A local prompt analyzer. Paste a prompt, get NLP stats, API pricing, and intent classification.

Live: https://prompteering.netlify.app/


## ✨ Features

- **Word Weight Map**: Color-coded word importance scores using NLTK POS tagging + keyword dictionaries.
- **Token Calculator**: Exact GPT token counts (`tiktoken`) plus calibrated estimates for Claude and Gemini.
- **Intent Classifier**: Predicts response type with confidence bars using a custom regex pattern engine.
- **Response Roadmap**: Dynamic flowchart of predicted processing steps based on prompt intent.
- **Cost Analysis**: Instant API pricing breakdown across major models (GPT-4o, Claude 3.5+, Gemini Flash).

---

## 🛠️ Tech Stack

- **Backend**: Zero-framework Python API (`http.server`).
- **NLTK**: Natural Language Toolkit for POS tagging and statistical analysis.
- **Tiktoken**: Exact token count matching for GPT models.
- **Frontend**: Vanilla HTML5/CSS3/JavaScript (No excessive frameworks).
- **Deployment**: Configured for **Netlify Functions** (Serverless).

---

## 🚀 Quick Start

### Option 1: Automatic Launcher (Windows)
Double-click `start.bat` in the root directory. This will automate the virtual environment setup and start the server.

### Option 2: Manual Installation
```bash
# 1. Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the server
python server.py

# → Open http://localhost:8000 in your browser
```

---

## 📂 Project Structure

```text
├── server.py              # Zero-framework backend API
├── requirements.txt       # Dependencies: `tiktoken` + `nltk`
├── start.bat              # One-click Windows launcher
├── templates/             # HTML Templates
│   └── index.html         # Main UI Interface
├── static/                # Static Assets
│   ├── css/style.css      # Dark theme UI Styling
│   └── js/app.js          # Client-side Logic
└── netlify/               # Netlify Serverless Functions
```

---

## ☁️ Deployment

### To Netlify
1. Connect your repository to Netlify.
2. The `netlify.toml` file automatically handles routing for the `/api/*` endpoints.
3. No build command is required. Set the publish directory to `.` (Root).

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

**Developed with ❤️ for the AI Community.**
