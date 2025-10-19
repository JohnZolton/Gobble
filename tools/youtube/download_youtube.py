#!/usr/bin/env python3
"""
Download and transcribe a podcast from a direct URL.

This script downloads an audio file from a provided URL and transcribes it using Whisper.
"""

import os
import sys
import argparse
import tempfile
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Import our custom modules
from tools.transcription.parakeet import transcribe_audio


def download_audio(url, output_path):
    """
    Download audio from a URL (supports YouTube/other urls via yt_dlp when available)
    and return a metadata dict: {"path": str(path), "title": str, "channel": str}
    Falls back to a simple requests-based download when yt_dlp is not available
    or when it fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try using yt_dlp for YouTube URLs to get metadata (title, uploader/channel)
    try:
        import yt_dlp
        ydl_opts = {
            "format": "bestaudio/best",
            # write extracted audio via ffmpeg postprocessor to mp3
            "outtmpl": str(output_path.with_suffix(".%(ext)s")),
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Extract metadata
        title = info.get("title") if isinstance(info, dict) else None
        channel = info.get("uploader") if isinstance(info, dict) else None

        # Determine actual file path (yt_dlp with the postprocessor creates .mp3)
        candidate = output_path.with_suffix(".mp3")
        if candidate.exists():
            return {"path": str(candidate), "title": title or candidate.stem, "channel": channel or "unknown_channel"}

        # If the expected file doesn't exist, try to infer from info
        # yt_dlp may provide a filename in info.get('_filename') or 'requested_downloads'
        possible_paths = []
        if isinstance(info, dict):
            if "_filename" in info:
                possible_paths.append(Path(info["_filename"]))
            if "requested_downloads" in info and isinstance(info["requested_downloads"], list):
                for d in info["requested_downloads"]:
                    if "filepath" in d:
                        possible_paths.append(Path(d["filepath"]))

        for p in possible_paths:
            if p.exists():
                return {"path": str(p), "title": title or p.stem, "channel": channel or "unknown_channel"}

        # As a last resort return the templated mp3 path even if it doesn't exist yet
        return {"path": str(candidate), "title": title or output_path.stem, "channel": channel or "unknown_channel"}

    except Exception as e:
        # yt_dlp not available or failed; fall back to simple requests download
        print(f"yt_dlp not used or failed ({e}), falling back to requests for URL: {url}")

    # Fallback: simple streaming download (keeps previous behavior)
    try:
        print(f"Downloading audio from URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        with open(output_path, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        print(f"Download complete: {output_path}")
        # best-effort metadata from filename
        return {"path": str(output_path), "title": output_path.stem, "channel": "unknown_channel"}
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download audio from a URL (YouTube or direct audio)."
    )
    parser.add_argument("url", help="URL of the YouTube video or audio file")
    parser.add_argument(
        "--out",
        "-o",
        help="Output path (file). If omitted, a temp file will be used",
        default=None,
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        out = tmp.name
        tmp.close()

    result = download_audio(args.url, out)
    if result is None:
        print("Download failed")
        sys.exit(1)

    # result may be dict with metadata
    if isinstance(result, dict):
        print("Downloaded:", result.get("path"))
        print("Title:", result.get("title"))
        print("Channel:", result.get("channel"))
    else:
        print("Downloaded:", result)

    sys.exit(0)
