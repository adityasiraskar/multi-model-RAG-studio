from youtube_transcript_api import YouTubeTranscriptApi
import requests
import threading
from config import app_config
import re
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_FILE = os.path.join(PROJECT_ROOT, "transcript_cache.json")

video_context = {}

# Load from disk if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        video_context = json.load(f)

# Save cache
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(video_context, f)

# Extract video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Background transcript fetching
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        full_text = " ".join(
            entry.text if hasattr(entry, "text") else entry["text"]
            for entry in transcript
        )
        video_context[video_id] = full_text
        save_cache()
    except Exception as e:
        video_context[video_id] = f"Error fetching transcript: {e}"
        save_cache()

# Handle URL submission
def handle_url_submit(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL."
    if video_id not in video_context:
        threading.Thread(target=fetch_transcript, args=(video_id,)).start()
        return f"Transcript is being fetched in the background for video ID: {video_id}..."
    return "Transcript already cached. Ready for QA."

# Send a prompt to Ollama
def query_ollama(prompt, model="mistral"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    ollama_base = app_config.OLLAMA_API_BASE.rstrip("/")
    response = requests.post(f"{ollama_base}/chat", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]["content"]

# Answer a question
def answer_question(question, url, model):
    video_id = extract_video_id(url)
    if video_id not in video_context:
        return "Transcript not ready yet."
    context = video_context[video_id]
    if context.startswith("Error"):
        return context
    prompt = f"""Answer the following question based on this YouTube video transcript:\n
    Transcript:\n{context[:4000]}\n
    Question: {question}
    """
    try:
        return query_ollama(prompt, model)
    except Exception as e:
        return f"Error from Ollama: {e}"

# Summarize transcript
def summarize_transcript(url, model):
    video_id = extract_video_id(url)
    if video_id not in video_context:
        return "Transcript not ready."
    context = video_context[video_id]
    if context.startswith("Error"):
        return context
    prompt = f"""Summarize the following YouTube video transcript:\n\n{context[:4000]}"""
    try:
        return query_ollama(prompt, model)
    except Exception as e:
        return f"Error from Ollama: {e}"


def answer_or_summarize_youtube(message, url=None, model="phi3:mini"):
    """Route a pasted-video request synchronously: answer a question or summarize."""
    candidate_url = url or extract_url(message)
    if not candidate_url:
        return "Please include a valid YouTube URL."

    video_id = extract_video_id(candidate_url)
    if not video_id:
        return "Please include a valid YouTube URL containing a video ID."

    if video_id not in video_context:
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id)
            text = " ".join(
                entry.text if hasattr(entry, "text") else entry["text"]
                for entry in transcript
            )
            video_context[video_id] = text
            save_cache()
        except Exception as exc:
            return f"Error fetching YouTube transcript: {exc}"

    context = video_context[video_id]
    if context.startswith("Error"):
        return context

    question = re.sub(r'https?://[^\s]+', '', message, flags=re.I).strip()
    if not question or question.lower() in {"youtube", "video", "watch", "summary", "summarize", "summarise"}:
        prompt = f"Summarize the following YouTube video transcript:\n\n{context[:4000]}"
    else:
        prompt = f"Answer using only the following YouTube transcript:\n\nTranscript:\n{context[:4000]}\n\nQuestion: {question}"

    try:
        return query_ollama(prompt, model)
    except Exception as exc:
        return f"Error from Ollama: {exc}"


def extract_url(message):
    match = re.search(r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s]+', str(message), flags=re.I)
    return match.group(0).rstrip(".,)]}") if match else None

