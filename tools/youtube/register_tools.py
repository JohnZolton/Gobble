import tempfile
from pathlib import Path
from pydantic import Field
from tools.transcription.parakeet import transcribe_audio
from .download_youtube import download_audio
import tempfile as _tempfile
import os
import re

def _sanitize_for_filename(s: str) -> str:
    # keep it simple: replace whitespace with underscore and remove unsafe chars
    s = s.strip()
    s = re.sub(r'\\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_\\-\\.\\(\\)]', '', s)
    if not s:
        return "unknown"
    return s

def get_youtube_transcript(url: Field(description="the url of the youtube video to transcribe")):
    """Download a video's audio to a temp file, transcribe it, save transcript to knowledge/,
    and return the transcript path.

    download_audio(url, output_path) is expected to return either:
      - None on failure
      - a dict with keys {"path": str, "title": str, "channel": str}
      - or a plain path string (legacy)

    This function creates a temporary file for the download, passes that path to download_audio,
    uses returned metadata (title/channel) when available, transcribes with Parakeet, writes the
    transcript to knowledge/youtube/<channel>/<title>-transcript.txt and returns that path.
    """
    # create a temp file name to pass to download_audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        temp_path = Path(tf.name)
    # ensure the tempfile is closed (NamedTemporaryFile on some platforms needs close)
    try:
        result = download_audio(url, temp_path)
    except Exception as e:
        # ensure cleanup on unexpected error
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return {"error": "download_exception", "message": str(e)}

    if not result:
        # download failed; clean up temp and return error
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return {"error": "download_failed", "url": url}

    # Normalize result into metadata fields
    if isinstance(result, dict):
        audio_path = Path(result.get("path"))
        title = result.get("title") or audio_path.stem
        channel = result.get("channel") or "unknown_channel"
    else:
        audio_path = Path(result)
        title = audio_path.stem
        channel = "unknown_channel"

    # Transcribe
    transcript_text = transcribe_audio(str(audio_path))
    if transcript_text is None:
        # cleanup downloaded file if it was the temp we created
        try:
            tmpdir = _tempfile.gettempdir()
            if str(audio_path).startswith(tmpdir) and audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        return {"error": "transcription_failed", "path": str(audio_path)}

    # Sanitize channel/title for filesystem
    channel_safe = _sanitize_for_filename(channel)
    title_safe = _sanitize_for_filename(title)

    # Save transcript into knowledge/youtube/<channel>/<title>-transcript.txt
    knowledge_root = Path("knowledge") / "youtube"
    out_dir = knowledge_root / channel_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{title_safe}-transcript.txt"
    try:
        out_path.write_text(transcript_text, encoding="utf-8")
    except Exception as e:
        return {"error": "write_failed", "message": str(e), "path": str(out_path)}

    # Remove temporary audio file if it's inside the system temp dir (to avoid deleting user files)
    try:
        tmpdir = _tempfile.gettempdir()
        if str(audio_path).startswith(tmpdir) and audio_path.exists():
            audio_path.unlink()
    except Exception:
        # non-fatal
        pass

    return str(out_path)


def register_youtube_tools(mcp):
    """Register youtube.get_transcript on the provided MCP server instance."""
    mcp.tool(
        name="youtube.get_transcript",
        description="Download and transcribe a YouTube video's audio."
    )(get_youtube_transcript)
