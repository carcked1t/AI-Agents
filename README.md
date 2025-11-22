# AI AGENT 1: YouTube Transcript to Content for Social Media

A lightweight Streamlit web application that extracts YouTube transcripts and generates AI-powered summaries using modern LLM APIs.

🔗 **Live App:** *(https://ai-agents-youtube-transcript.streamlit.app/)*

---

## 🚀 Features

* Extracts transcripts from any public YouTube video with a transcript.
* Generates AI-powered summaries and content for one or more social media platforms.
* Simple, clean UI built with Streamlit.
* Modular logic handled in `youtubetranscripter.py`.
* Does **not** store any user data.

---

## 📁 Project Structure

```
.
├── app.py                     # Main Streamlit app
├── youtubetranscripter.py     # Transcript + AI summary logic
└── requirements.txt           # Dependencies
```

---

## 🛠️ Local Installation & Setup

### 1️. Clone the repository

```bash
https://github.com/carcked1t/AI-Agents.git
cd AI-agents
```

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

### 3️. Add your API keys

Create a `.env` file (not included in GitHub):

```
AI_API_KEY=your_key
```

### 4. Run the app

```bash
streamlit run app.py
```

---
