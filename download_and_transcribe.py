#!/usr/bin/env python3
"""
Download and transcribe a podcast from a direct URL.

This script downloads an audio file from a provided URL and transcribes it using Whisper.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Import our custom modules
from transcribe import transcribe_audio, save_transcription, TranscriptionResult

def download_audio(url, output_path):
    """
    Download audio from a URL to a local file.
    
    Args:
        url: URL of the audio file
        output_path: Path to save the downloaded file
        
    Returns:
        Path to the downloaded file or None if download failed
    """
    print(f"Downloading audio from URL: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        output_path = Path(output_path)
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        print(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python download_and_transcribe.py <audio_url> [output_directory]")
        sys.exit(1)
    
    # Get the URL from command line arguments
    url = sys.argv[1]
    
    # Get the output directory (default: output)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    output_dir = Path(output_dir)
    
    # Create output directories
    audio_dir = output_dir / "audio"
    transcripts_dir = output_dir / "transcripts"
    
    for directory in [audio_dir, transcripts_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Generate a filename based on the current timestamp
    timestamp = int(time.time())
    audio_path = audio_dir / f"podcast_{timestamp}.mp4"
    
    # Download the audio
    downloaded_file = download_audio(url, audio_path)
    
    if not downloaded_file:
        print("Download failed. Exiting.")
        sys.exit(1)
    
    # Transcribe the audio
    print(f"Transcribing audio: {downloaded_file}")
    result = transcribe_audio(str(downloaded_file))
    
    # Save the transcription
    transcript_txt_path = transcripts_dir / f"podcast_{timestamp}.txt"
    save_transcription(result, str(transcript_txt_path))
    print(f"Transcription saved to {transcript_txt_path}")
    
    # Also save raw JSON for later processing
    transcript_json_path = transcripts_dir / f"podcast_{timestamp}.json"
    import json
    with open(transcript_json_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"Raw transcription data saved to {transcript_json_path}")
    
    print("\nProcess complete!")
    print(f"Audio file: {downloaded_file}")
    print(f"Transcript: {transcript_txt_path}")
    print(f"JSON data: {transcript_json_path}")

if __name__ == "__main__":
    main()
