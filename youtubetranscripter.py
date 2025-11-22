import asyncio
import os
import logging
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    InvalidVideoId,
)
import openai
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass


load_dotenv()
ai_api_key = os.getenv("AI_API_KEY")
client = OpenAI(
    api_key=ai_api_key,
    base_url="https://api.groq.com/openai/v1"
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Post:
    platform: str
    content: str


def _extract_response_text(response) -> str:
    """Best-effort extractor for different OpenAI response shapes.

    Tries common attributes (output_text, output) and falls back to str(response).
    """
    # Preferred attribute used by some SDK versions
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Some SDKs return an 'output' list/dict structure
    output = getattr(response, "output", None)
    if output:
        try:
            parts: List[str] = []
            for item in output:
                # item can be dict-like or object
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                parts.append(c["text"])
                            elif isinstance(c, str):
                                parts.append(c)
                    elif isinstance(content, str):
                        parts.append(content)
                else:
                    parts.append(str(item))
            if parts:
                return " ".join(parts)
        except Exception:
            # fall through to final fallback
            pass

    # Final fallback
    return str(response)


def generate_social_media_content(video_transcript: str, social_media_platform: str) -> str:
    logger.info("Generating content for %s", social_media_platform)
    prompt = (
        f"Here is a new video transcript:\n{video_transcript}\n\n"
        f"Generate engaging content suitable for {social_media_platform} based on this transcript."
    )

    try:
        response = client.responses.create(
            model="llama-3.3-70b-versatile",
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=2500,
        )
        return _extract_response_text(response)
    except openai.RateLimitError as e:
        logger.error("OpenAI rate limit / quota error: %s", e)
        return "[OpenAI request failed: rate limit or insufficient quota]"
    except Exception as e:
        logger.exception("Unexpected error calling OpenAI: %s", e)
        return f"[OpenAI request failed: {e}]"


def get_transcript(video_id: str, languages: Optional[List[str]] = None) -> str:
    if languages is None:
        languages = ["en"]

    try:
        # Get transcript list object
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Attempt to find the requested language transcript
        transcript_obj = transcript_list.find_transcript(languages)

        # Fetch full transcript
        fetched = transcript_obj.fetch()

        # Join all text segments
        transcript_text = " ".join(item.get("text", "") for item in fetched)

        return transcript_text.strip()

    except (CouldNotRetrieveTranscript, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, InvalidVideoId) as e:
        logger.warning("Transcript not available for %s: %s", video_id, e)
        return "Transcript not available."

    except Exception as e:
        logger.error("Unexpected error while fetching transcript: %s", e)
        return "Transcript not available."


async def main():
    video_id = "Zxs7Rf2rWxc"
    transcript = get_transcript(video_id, languages=["en"])  # sync call
    msg = f"Create a linkedin post based on this video transcript: {transcript}"
    post = generate_social_media_content(transcript, "LinkedIn")
    logger.info("generated post:\n%s", post)


if __name__ == "__main__":
    asyncio.run(main())

