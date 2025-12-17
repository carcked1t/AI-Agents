# AI AGENT 1: YouTube Transcript to Content for Social Media

A lightweight Streamlit web application that extracts YouTube transcripts and generates AI-powered summaries using modern LLM APIs.

🔗 **Live App:** *[(https://ai-agents-youtube-transcript.streamlit.app/)](https://ai-agents-yt-transcript.streamlit.app/)*

##  What This Project Does

This app allows you to:

* Input a **YouTube video URL or ID**
* Provide a **user query** or content goal
* Select one or more **social media platforms** (LinkedIn, Twitter/X, Instagram, etc.)
* **Generate concise, engaging content** tailored to each platform using a large language model

The workflow:

1. Fetch YouTube captions (English, manual or auto-generated)
2. If available, **process transcript**
3. Generate social media content using the LLM



##  Features

* Automatic **YouTube transcript fetching**
* Handles **multiple English transcripts**
* **Truncates large transcripts** to avoid context overflow
* **Platform-specific content generation**
* **Caching** for transcripts to improve performance in Streamlit
* **Clean, reusable code structure** (`youtubetranscripter.py`)


##  How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI / Groq API key in `.env`:

```
AI_API_KEY=your_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

5. Enter a **YouTube video URL**, provide your **query**, select **platforms**, and click **Generate**.



##  Known Issues & Limitations

1. **YouTube captions are unreliable**

   * Some videos may not provide English captions
   * Captions may fail due to age restrictions, region restrictions, or YouTube experiments

2. **Generated vs manual transcripts**

   * Sometimes auto-generated captions are inaccurate
   * Currently, the app only uses available English transcripts

3. **Long videos**

   * Transcripts exceeding ~12,000 characters are truncated to prevent LLM context overflow

4. **Fallback not implemented yet**

   * Videos without captions will show “Transcript not available”
   * A future enhancement could integrate audio transcription (e.g., Whisper)



##  How This App is Different

Unlike generic ChatPDF-style apps:

* Focused specifically on **YouTube video transcripts**
* Generates **platform-specific content** rather than just summarizing
* Handles multiple transcript types (manual/generated)
* Provides a **lightweight, cache-friendly, reproducible Streamlit UI**



##  Issues Faced

* **Parsing different YouTube URL formats** (watch, shorts, embed, youtu.be)
* **Handling multiple transcript languages and generated captions**
* **Truncating long transcripts** to avoid model context overflow
* **Graceful error handling** for unavailable transcripts or API failures



##  Future Improvements

* Whisper / audio fallback for videos without captions
* More sophisticated **platform-specific tone** (formal LinkedIn vs casual Instagram)
* Integration with **batch processing** for multiple videos
* Interactive **preview and editing** of generated content



## 📂 File Structure

```
├── app.py                    # Streamlit app
├── youtubetranscripter.py    # Transcript fetching & content generation
├── requirements.txt          # Python dependencies
├── .env                      # API keys
└── README.md
```

