import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import math
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

def transcribe_audio(audio_path: str, segment_length_sec: int = 60) -> TranscriptionResult:
    """
    Transcribe audio file using Parakeet TDT model and return segments with timestamps.
    
    Args:
        audio_path: Path to the audio file
        segment_length_sec: Length of each audio segment in seconds (default: 60)
        
    Returns:
        TranscriptionResult object containing the transcription
    """
    # Check if the model is already loaded, otherwise load it
    if not hasattr(transcribe_audio, "asr_model"):
        print("Loading Parakeet TDT model...")
        # Ensure the model is moved to the GPU if available
        transcribe_audio.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").cuda()
        print("Model loaded and moved to GPU.")

    asr_model = transcribe_audio.asr_model
    original_audio_path = audio_path
    temp_files = []  # To keep track of temporary segment files

    try:
        # Ensure the file exists
        if not os.path.exists(original_audio_path):
            raise FileNotFoundError(f"Audio file not found at {original_audio_path}")

        print(f"Processing audio file: {os.path.basename(original_audio_path)}")
        start_time = time.time()

        # Load the audio file using pydub
        audio = AudioSegment.from_file(original_audio_path)

        # Check and convert to mono if necessary
        if audio.channels > 1:
            print(f"Audio has {audio.channels} channels. Converting to mono.")
            audio = audio.set_channels(1)

        # Check sample rate and resample to 16kHz if necessary (standard for ASR models)
        if audio.frame_rate != 16000:
            print(f"Audio has sample rate {audio.frame_rate} Hz. Resampling to 16000 Hz.")
            audio = audio.set_frame_rate(16000)

        # --- Split audio into segments ---
        segment_length_ms = segment_length_sec * 1000
        total_length_ms = len(audio)
        num_segments = math.ceil(total_length_ms / segment_length_ms)

        print(f"Total audio length: {total_length_ms / 1000:.2f} seconds")
        print(f"Splitting into {num_segments} segments of up to {segment_length_sec} seconds.")

        all_segments = []

        for i in range(num_segments):
            start_time_ms = i * segment_length_ms
            end_time_ms = min((i + 1) * segment_length_ms, total_length_ms)
            segment = audio[start_time_ms:end_time_ms]

            # Create a temporary WAV file for the segment
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.wav', delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                temp_files.append(temp_wav_file_path)  # Add to list for cleanup

            # Export the segment to the temporary file
            segment.export(temp_wav_file_path, format='wav')

            # Transcribe the current segment
            print(f"Transcribing segment {i+1}/{num_segments} ({start_time_ms/1000:.2f}s - {end_time_ms/1000:.2f}s)...")
            segment_transcription = asr_model.transcribe([temp_wav_file_path], timestamps=True)

            if segment_transcription and len(segment_transcription) > 0:
                # Get segment-level timestamps
                segment_timestamps = segment_transcription[0].timestamp.get('segment', [])
                
                for timestamp in segment_timestamps:
                    # Adjust timestamps to account for the segment offset
                    adjusted_start = timestamp['start'] + (start_time_ms / 1000)
                    adjusted_end = timestamp['end'] + (start_time_ms / 1000)
                    
                    all_segments.append(Segment(
                        text=timestamp['segment'],
                        start=adjusted_start,
                        end=adjusted_end
                    ))
            else:
                print(f"[Transcription Failed for Segment {i+1}]")

        print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        return TranscriptionResult(segments=all_segments)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Clean up all temporary files
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("Temporary files cleaned up.")

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
    
    parser = argparse.ArgumentParser(description="Transcribe audio file using Parakeet TDT model")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file prefix (without extension)", default="transcription")
    parser.add_argument("-s", "--segment-length", type=int, default=60,
                        help="Length of audio segments in seconds (default: 60)")
    args = parser.parse_args()
    
    audio_file = args.audio_file
    output_prefix = args.output
    segment_length = args.segment_length
    
    # Ensure audio file exists
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    print(f"Transcribing audio file: {audio_file}")
    
    # Transcribe audio
    result = transcribe_audio(audio_file, segment_length)
    
    # Save timestamped transcription
    output_file = f"{output_prefix}.txt"
    save_transcription(result, output_file)
    print(f"Transcription saved to {output_file}")
    
    # Also save raw JSON for later processing
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"Raw transcription data saved to {json_file}")
