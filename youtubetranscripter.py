import os
import logging
from typing import List, Optional
from dataclasses import dataclass
import re
from urllib.parse import urlparse, parse_qs


from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()
API_KEY = os.getenv("AI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ AI_API_KEY not found in environment variables")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 12000  # prevents context overflow


# Utils
def truncate_text(text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def extract_response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return str(response)


# Transcript
def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try ALL English transcripts (manual + generated)
        for transcript in transcript_list:
            if transcript.language_code.startswith("en"):
                try:
                    fetched = transcript.fetch()
                    text = " ".join(item["text"] for item in fetched)
                    if text.strip():
                        return truncate_text(text)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch transcript "
                        f"(lang={transcript.language_code}, "
                        f"generated={transcript.is_generated}): {e}"
                    )
                    continue  # 🔥 THIS IS THE FIX

        # No usable English transcript
        logger.warning("No valid English transcript could be fetched.")
        return ""

    except Exception as e:
        logger.error(f"Transcript list failed for {video_id}: {e}")
        return ""

# Content Generation
def generate_social_media_content(
    transcript: str,
    platform: str,
    user_query: str
) -> str:
    if not transcript.strip():
        return "❌ Transcript is empty."

    prompt = f"""
You are a professional social media content writer.

Platform: {platform}

User request:
{user_query}

Video transcript:
{transcript}

Generate concise, engaging, platform-appropriate content.
"""

    try:
        response = client.responses.create(
            model="llama-3.3-70b-versatile",
            input=prompt,
            max_output_tokens=700,
        )
        return extract_response_text(response)

    except Exception as e:
        logger.exception("LLM generation failed")
        return f"❌ Content generation failed: {str(e)}"


def extract_video_id(youtube_input: str) -> str | None:

    youtube_input = youtube_input.strip()

    # Case 1: Already a video ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", youtube_input):
        return youtube_input

    parsed = urlparse(youtube_input)

    # youtu.be/VIDEO_ID
    if parsed.netloc in {"youtu.be"}:
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=VIDEO_ID
    if parsed.netloc.endswith("youtube.com"):
        query = parse_qs(parsed.query)

        if "v" in query:
            return query["v"][0]

        # /embed/VIDEO_ID or /shorts/VIDEO_ID
        path_parts = parsed.path.split("/")
        if "embed" in path_parts or "shorts" in path_parts:
            return path_parts[-1]

    return None
