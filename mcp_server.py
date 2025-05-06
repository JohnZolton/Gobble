"""
Gobble: MCP server for Chat-with-podcast integration.
"""
import os
import json
import logging
import tempfile
from typing import Optional, Sequence, Dict, Any, Tuple, List
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

from fastmcp import FastMCP
import mcp.types as types
import asyncio
from pydantic import Field, BaseModel

# Import our custom modules
from transcribe import transcribe_audio, save_transcription, TranscriptionResult
from main import PodcastProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gobble-server")

# Increase timeout to 5 minutes (300 seconds)
mcp = FastMCP("Podcast MCP Server", host="0.0.0.0", port=8000, timeout=300)

# won't work easily bc its all frontend js not url based, ignore for now, user will have to upload a link
# okay we can use google and then filter fountain results to get fountain.fm/{"show" or "episode"}/{id}
@mcp.tool()
async def search_fountain(query: str):
    """Search for a specific podcast on fountain.fm"""
    pass


@mcp.tool()
async def transcribe_episode_fountain(episode_url: str):
    """Downloads and transcribes a specific podcast episode from a given fountain.fm URL
    
    Args:
        episode_url: URL of the Fountain.fm episode to transcribe
        
    Returns:
        Dictionary containing the transcription results with segments and timestamps
    """
    logger.info(f"Transcribing Fountain.fm episode: {episode_url}")
    
    try:
        # Import necessary modules
        import requests
        import re
        import threading
        from transcribe import transcribe_audio, save_transcription
        
        # Extract MP3 URL and podcast info from fountain.fm
        logger.info("Extracting episode info from fountain.fm")
        
        # Get the webpage content
        response = requests.get(episode_url)
        response.raise_for_status()
        html_content = response.text
        
        # Extract the MP3 URL using regex
        mp3_pattern = r'https://[^"]*\.mp3[^"]*'
        mp3_matches = re.findall(mp3_pattern, html_content)
        
        if not mp3_matches:
            logger.error("No MP3 URL found in the webpage")
            return {"error": "Could not extract MP3 URL from the fountain.fm page"}
        
        # Get the first MP3 URL and clean it up (replace escaped characters)
        mp3_url = mp3_matches[0].replace('\\u0026', '&').rstrip('\\')
        logger.info(f"Found MP3 URL: {mp3_url}")
        
        # Extract the title from the HTML
        title_pattern = r'<title>(.*?)</title>'
        title_match = re.search(title_pattern, html_content)
        full_title = title_match.group(1) if title_match else "Unknown Episode"
        
        # Extract show name from the title (usually before the first • or - character)
        show_name_match = re.search(r'^(.*?)(?:\s[•\-]\s|\s\|\s)', full_title)
        show_name = show_name_match.group(1).strip() if show_name_match else "Unknown Show"
        
        # Clean up the show name for use as a directory name
        safe_show_name = re.sub(r'[^\w\-\.]', '_', show_name)
        logger.info(f"Show name: {safe_show_name}")
        
        # Extract episode title (after the show name)
        episode_title_match = re.search(r'(?:\s[•\-]\s|\s\|\s)(.*?)(?:\s[•\-]\s|\s\|\s|$)', full_title)
        episode_title = episode_title_match.group(1).strip() if episode_title_match else full_title
        logger.info(f"Episode title: {episode_title}")
        
        # Create a safe filename from the episode title
        safe_title = re.sub(r'[^\w\-\.]', '_', episode_title)
        
        # Create output directories
        output_dir = Path("output")
        audio_dir = output_dir / "audio" / safe_show_name
        transcripts_dir = output_dir / "transcripts" / safe_show_name
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        audio_path = audio_dir / f"{safe_title}.mp3"
        transcript_txt_path = transcripts_dir / f"{safe_title}.txt"
        transcript_json_path = transcripts_dir / f"{safe_title}.json"
        
        # Check if already transcribed
        if transcript_txt_path.exists() and transcript_json_path.exists():
            logger.info(f"Episode already transcribed: {transcript_txt_path}")
            # Load existing transcription
            with open(transcript_json_path, 'r') as f:
                transcription_data = json.load(f)
                
            # Add metadata to the result
            transcription_data["metadata"] = {
                "show_name": safe_show_name,
                "episode_title": episode_title,
                "url": episode_url,
                "mp3_url": mp3_url,
                "transcript_path": str(transcript_txt_path),
                "json_path": str(transcript_json_path),
                "audio_path": str(audio_path),
                "status": "completed"
            }
            
            return transcription_data
        
        # Define a function to process the episode in the background
        def process_episode():
            try:
                # Download the episode if it doesn't exist
                if not audio_path.exists():
                    logger.info(f"Downloading episode to {audio_path}")
                    response = requests.get(mp3_url, stream=True)
                    response.raise_for_status()
                    
                    with open(audio_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                else:
                    logger.info(f"Episode already downloaded: {audio_path}")
                
                # Transcribe the audio
                logger.info(f"Transcribing audio file: {audio_path}")
                result = transcribe_audio(str(audio_path), model_name="tiny.en")
                
                # Save the transcription
                save_transcription(result, str(transcript_txt_path))
                logger.info(f"Transcription saved to: {transcript_txt_path}")
                
                # Save raw JSON for later processing
                with open(transcript_json_path, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2)
                logger.info(f"Raw transcription data saved to: {transcript_json_path}")
                
            except Exception as e:
                logger.error(f"Error processing episode in background: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Start the background processing thread
        thread = threading.Thread(target=process_episode)
        thread.daemon = True
        thread.start()
        
        # Return a quick response with status information
        return {
            "status": "processing",
            "message": "Episode is being downloaded and transcribed in the background",
            "metadata": {
                "show_name": safe_show_name,
                "episode_title": episode_title,
                "url": episode_url,
                "mp3_url": mp3_url,
                "expected_transcript_path": str(transcript_txt_path),
                "expected_json_path": str(transcript_json_path),
                "expected_audio_path": str(audio_path)
            }
        }
            
    except Exception as e:
        logger.error(f"Error transcribing episode: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Transcription failed: {str(e)}"}
    
@mcp.tool()
async def create_vector_db_from_series_fountain(show_id: str, episode_limit: int):
    """Downloads, transcribes, and stores entire series of a fountain.fm show URL into a vector db"""
    pass

@mcp.tool()
async def search_youtube(query: str):
    """"Search youtube for a specific channel/episode etc"""
    pass

@mcp.tool()
async def transcribe_youtube(query: str):
    """"transcribe a youtube video"""
    pass


# MCP resources for accessing transcripts
@mcp.resource("gobble://transcripts/{filename*}")
async def get_transcript(filename: str) -> str:
    """Retrieve the content of a transcript file.
    
    Args:
        filename: Path to the transcript file within the output/transcripts directory
        
    Returns:
        Content of the transcript file as text
    """
    logger.info(f"Resource request for transcript: {filename}")
    
    # Construct the full path to the transcript file
    transcript_path = Path("output/transcripts") / filename
    
    # Check if the file exists
    if not transcript_path.exists():
        return f"Transcript file not found: {filename}"
    
    # Read and return the content of the file
    try:
        with open(transcript_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading transcript file: {str(e)}")
        return f"Error reading transcript file: {str(e)}"

@mcp.resource("gobble://transcripts")
async def list_transcripts() -> dict:
    """List all available transcript files.
    
    Returns:
        Dictionary containing lists of available transcripts organized by show
    """
    logger.info("Resource request for transcript listing")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("output/transcripts")
    
    # Check if the directory exists
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # Dictionary to store the results
    results = {}
    
    # Walk through the directory and collect all .txt files
    for show_dir in transcripts_dir.iterdir():
        if show_dir.is_dir():
            show_name = show_dir.name
            results[show_name] = []
            
            for transcript_file in show_dir.glob("*.txt"):
                results[show_name].append(transcript_file.name)
    
    return results

# Search resource
@mcp.resource("gobble://search/{query}")
async def get_podcast_results(query: str, max_results: int = 10) -> str:
    """Retrieve search results for a specific query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        JSON string containing the search results
    """
    logger.info(f"Resource request for search results with query: {query}, max_results: {max_results}")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(mcp.run_sse_async())
