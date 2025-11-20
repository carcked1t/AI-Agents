# AI AGENT 1: YouTube Transcript to Content for Social Media

# YouTube Transcript AI Agent

This project is a Streamlit-based web application that allows users to input a query and a YouTube video ID to automatically fetch, process, and generate social media content across multiple platforms using AI.

## 🚀 Features

* **Query Input**: Users can submit any question or content requirement.
* **YouTube Video Support**: Accepts a YouTube video ID and retrieves the transcript.
* **AI-Generated Content**: Produces social media content for one or multiple platforms based on user input.
* **Streamlit UI**: Simple and interactive front-end interface.

## 📂 Project Structure

```
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <project-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

## 🔑 Environment Variables

Make sure to add your API keys in a `.env` file:

```
OPENAI_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **OpenAI API / Groq API**
* **YouTube Transcript API**

