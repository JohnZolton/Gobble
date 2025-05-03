import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
google_key = os.getenv('GEMINI_API_KEY')

class Advertisement(BaseModel):
    name: str = Field(description="The sponsor/company name")
    start: str = Field(description="Start timestamp of the ad (HH:MM:SS)")
    end: str = Field(description="End timestamp of the ad (HH:MM:SS)")

class AdvertisementList(BaseModel):
    advertisements: List[Advertisement] = Field(description="List of advertisements found in the transcript")

def load_transcription_from_txt(txt_path: str) -> str:
    """
    Load the transcription from a TXT file
    
    Args:
        txt_path: Path to the transcript TXT file
        
    Returns:
        The transcript text with timestamps
    """
    with open(txt_path, 'r') as f:
        return f.read()

def load_transcription_from_json(json_path: str) -> Dict:
    """
    Load the transcription from JSON file
    
    Args:
        json_path: Path to the transcript JSON file
        
    Returns:
        Dictionary containing the transcript data
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def find_sponsored_segments(transcript_input, is_json: bool = False) -> List[Advertisement]:
    """
    Use Gemini to analyze transcription and find sponsored segments
    
    Args:
        transcript_input: Either a dictionary (from JSON) or a string (from TXT)
        is_json: Whether the input is from a JSON file
        
    Returns:
        List of Advertisement objects
    """
    # Format transcription for analysis if it's from JSON
    if is_json:
        formatted_text = "\n".join([
            f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text']}"
            for segment in transcript_input['segments']
        ])
    else:
        # If it's already formatted text from a TXT file, use it directly
        formatted_text = transcript_input
    
    prompt = """Analyze this podcast transcription and identify all sponsored segments and advertisements.
    
    Return a list of all advertisements found in the transcript. For each advertisement, include:
    - The sponsor/company name
    - Start timestamp of the ad (HH:MM:SS)
    - End timestamp of the ad (HH:MM:SS)
    
    If no advertisements are found, return an empty list.
    """

    client = genai.Client(api_key=google_key)
    
    try:
        # Use structured output with Pydantic model
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17", 
            contents=[prompt, formatted_text],
            config={
                'response_mime_type': 'application/json',
                'response_schema': AdvertisementList,
            }
        )
        
        # Parse the response using the Pydantic model
        ad_list = response.parsed
        return ad_list.advertisements
    except Exception as e:
        print(f"Error with structured output: {e}")
        print("Falling back to regular JSON parsing...")
        
        # Fallback to regular generation without structured output
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17", 
            contents=[prompt, formatted_text]
        )
        
        # Parse the response into Advertisement objects
        ads = []
        try:
            result = json.loads(response.text)
            if isinstance(result, list):
                for ad in result:
                    ads.append(Advertisement(**ad))
            elif isinstance(result, dict) and "advertisements" in result:
                for ad in result["advertisements"]:
                    ads.append(Advertisement(**ad))
        except Exception as e:
            print(f"Error parsing response: {e}")
            print("Raw response:", response.text)
            return []
    
    return ads

def find_ads_in_file(file_path: str) -> List[Advertisement]:
    """
    Find advertisements in a transcript file (either TXT or JSON)
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        List of Advertisement objects
    """
    path = Path(file_path)
    
    if path.suffix.lower() == '.txt':
        # Load from TXT file
        transcript = load_transcription_from_txt(file_path)
        return find_sponsored_segments(transcript, is_json=False)
    elif path.suffix.lower() == '.json':
        # Load from JSON file
        transcript = load_transcription_from_json(file_path)
        return find_sponsored_segments(transcript, is_json=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find advertisements in podcast transcripts")
    parser.add_argument("transcript_file", help="Path to the transcript file (TXT or JSON)")
    
    args = parser.parse_args()
    
    # Find sponsored segments
    print(f"Analyzing transcript file: {args.transcript_file}")
    print("Looking for sponsored segments...")
    start_time = time.time()
    ads = find_ads_in_file(args.transcript_file)
    
    # Print results
    print(f"\nFound {len(ads)} sponsored segments:")
    for ad in ads:
        print(f"\n{ad.name}:")
        print(f"  Start: {ad.start}")
        print(f"  End: {ad.end}")
    
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
