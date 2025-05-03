import whisper
import time
from pathlib import Path
import json
from typing import List, Dict
from pydantic import BaseModel

class Segment(BaseModel):
    text: str
    start: float
    end: float

class TranscriptionResult(BaseModel):
    segments: List[Segment]

def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """
    Transcribe audio file using Whisper tiny model and return segments with timestamps.
    """
    print("Loading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    print(f"Transcribing {audio_path}...")
    start_time = time.time()
    
    # Transcribe with word-level timestamps
    result = model.transcribe(
        audio_path,
        verbose=True,
        word_timestamps=True
    )
    
    # Convert to our data model
    segments = [
        Segment(
            text=segment["text"],
            start=segment["start"],
            end=segment["end"]
        )
        for segment in result["segments"]
    ]
    
    print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
    return TranscriptionResult(segments=segments)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def save_transcription(result: TranscriptionResult, output_path: str):
    """Save transcription with formatted timestamps to a file"""
    with open(output_path, 'w') as f:
        for segment in result.segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"[{start} --> {end}] {segment.text}\n")

if __name__ == "__main__":
    audio_file = "tftc_608.mp3"
    
    # Ensure audio file exists
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Transcribe audio
    result = transcribe_audio(audio_file)
    
    # Save timestamped transcription
    output_file = "transcription.txt"
    save_transcription(result, output_file)
    print(f"Transcription saved to {output_file}")
    
    # Also save raw JSON for later processing
    with open("transcription.json", 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    print("Raw transcription data saved to transcription.json")