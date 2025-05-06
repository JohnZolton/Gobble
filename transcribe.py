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

def transcribe_audio(audio_path: str, model_name: str = "tiny") -> TranscriptionResult:
    """
    Transcribe audio file using Whisper model and return segments with timestamps.
    
    Args:
        audio_path: Path to the audio file
        model_name: Name of the Whisper model to use (default: tiny)
        
    Returns:
        TranscriptionResult object containing the transcription
    """
    import torch
    
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper {model_name} model on {device}...")
    
    # Load the model with the specified device
    model = whisper.load_model(model_name, device=device)
    
    print(f"Transcribing {audio_path} using {device}...")
    start_time = time.time()
    
    # Transcribe with word-level timestamps
    result = model.transcribe(
        audio_path,
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio file using Whisper")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file prefix (without extension)", default="transcription")
    args = parser.parse_args()
    
    audio_file = args.audio_file
    output_prefix = args.output
    
    # Ensure audio file exists
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    print(f"Transcribing audio file: {audio_file}")
    
    # Transcribe audio
    result = transcribe_audio(audio_file)
    
    # Save timestamped transcription
    output_file = f"{output_prefix}.txt"
    save_transcription(result, output_file)
    print(f"Transcription saved to {output_file}")
    
    # Also save raw JSON for later processing
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"Raw transcription data saved to {json_file}")
