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
import re
import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter



client = chromadb.PersistentClient(path="chroma")
transcript_collection = client.get_or_create_collection(name="my_collection")

# Path to store processed files metadata
PROCESSED_FILES_PATH = Path("chroma/processed_files.json")

def load_processed_files():
    """Load record of previously processed files"""
    if PROCESSED_FILES_PATH.exists():
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading processed files record, starting fresh")
            return {}
    return {}

def save_processed_files(processed):
    """Save record of processed files"""
    # Create directory if it doesn't exist
    PROCESSED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed, f, indent=2)

def get_file_metadata(path: Path) -> dict:
    """Get metadata for a file to track changes"""
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "num_chunks": 0  # Will be updated after processing
    }


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
# @mcp.tool()
# async def search_fountain(query: str):
#     """Search for a specific podcast on fountain.fm"""
#     pass


def initialize_knowledge_base(force_reprocess=False):
    """Initialize Chroma knowledge base with new or modified transcripts
    
    Args:
        force_reprocess: If True, reprocess all transcripts regardless of whether they've changed
    """
    logger.info("Initializing Chroma knowledge base...")
    
    # Load record of previously processed files
    processed_files = load_processed_files()
    
    transcripts_dir = Path("output/transcripts")
    if not transcripts_dir.exists():
        logger.warning(f"Transcripts directory not found: {transcripts_dir}")
        return
    
    transcript_files = list(transcripts_dir.rglob("*.txt"))
    logger.info(f"Found {len(transcript_files)} transcript files")
    
    # Track files to process
    files_to_process = []
    for transcript_path in transcript_files:
        file_key = str(transcript_path)
        current_metadata = get_file_metadata(transcript_path)
        
        # Skip if file hasn't changed and we're not forcing reprocessing
        if not force_reprocess and file_key in processed_files:
            old_metadata = processed_files[file_key]
            if (old_metadata["size"] == current_metadata["size"] and 
                old_metadata["mtime"] == current_metadata["mtime"]):
                logger.info(f"Skipping unchanged transcript: {transcript_path}")
                continue
        
        files_to_process.append((transcript_path, current_metadata))
    
    logger.info(f"Processing {len(files_to_process)} new or modified transcripts")
    
    # Process files that need updating
    total_chunks = 0
    for transcript_path, current_metadata in files_to_process:
        try:
            # Extract metadata from path
            source = transcript_path.parent.parent.name  # youtube or fountain
            channel = transcript_path.parent.name
            title = transcript_path.stem
            file_key = str(transcript_path)
            
            logger.info(f"Processing transcript: {source}/{channel}/{title}")
            
            # Read transcript content
            with open(transcript_path, 'r') as f:
                content = f.read()
            
            # Clean transcript content (remove timestamps)
            cleaned_lines = [
                line[25:].strip() 
                for line in content.split('\n')
                if line.strip()
            ]
            cleaned_text = ' '.join(cleaned_lines)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len
            )
            chunks = text_splitter.split_text(cleaned_text)
            
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Remove existing chunks for this transcript if any
            try:
                # Get existing IDs for this transcript
                existing_ids = transcript_collection.get(
                    where={"path": str(transcript_path)}
                )["ids"]
                
                if existing_ids:
                    logger.info(f"Removing {len(existing_ids)} existing chunks for {title}")
                    transcript_collection.delete(ids=existing_ids)
            except Exception as e:
                logger.warning(f"Error checking for existing chunks: {str(e)}")
            
            # Add chunks to collection with metadata
            for i, chunk in enumerate(chunks):
                transcript_collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "source": source,
                        "channel": channel,
                        "title": title,
                        "chunk": i,
                        "path": str(transcript_path)
                    }],
                    ids=[f"{source}-{channel}-{title}-chunk-{i}"]
                )
            
            # Update metadata with number of chunks
            current_metadata["num_chunks"] = len(chunks)
            processed_files[file_key] = current_metadata
            total_chunks += len(chunks)
            
            logger.info(f"Successfully added {len(chunks)} chunks to collection for {title}")
            
            # Save processed files record periodically
            save_processed_files(processed_files)
            
        except Exception as e:
            logger.error(f"Error processing transcript {transcript_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Final save of processed files record
    save_processed_files(processed_files)
    
    logger.info(f"Knowledge base initialization complete!")
    logger.info(f"Processed {len(files_to_process)} transcripts")
    logger.info(f"Total chunks added to collection: {total_chunks}")
            


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
        
        # Helper function for sanitizing filenames
        def sanitize_filename(name):
            # Replace problematic characters with underscores
            name = re.sub(r'[<>:"/\\|?*\u0000-\u001F]', '_', name)
            name = re.sub(r'[\s,]+', '_', name)  # Replace spaces and commas
            name = re.sub(r'[^\x00-\x7F]+', '_', name)  # Replace non-ASCII
            name = re.sub(r'_{2,}', '_', name)  # Replace multiple underscores
            name = name.strip('._-')  # Trim special chars from ends
            return name if name else "unnamed"
        
        # Clean up names for use in paths
        safe_show_name = sanitize_filename(show_name)
        logger.info(f"Show name: {safe_show_name}")
        
        # Extract episode title (after the show name)
        episode_title_match = re.search(r'(?:\s[•\-]\s|\s\|\s)(.*?)(?:\s[•\-]\s|\s\|\s|$)', full_title)
        episode_title = episode_title_match.group(1).strip() if episode_title_match else full_title
        logger.info(f"Episode title: {episode_title}")
        
        # Create a safe filename from the episode title
        safe_title = sanitize_filename(episode_title)
        
        # Create output directories
        output_dir = Path("output")
        temp_dir = output_dir / "temp"
        audio_dir = output_dir / "audio" / "fountain" / safe_show_name
        transcripts_dir = output_dir / "transcripts" / "fountain" / safe_show_name
        
        for directory in [temp_dir, audio_dir, transcripts_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        temp_audio_path = temp_dir / f"{safe_title}.mp3"
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
                "status": "already_transcribed"
            }
            
            return {
                "status": "already_transcribed",
                "message": f"Episode was already transcribed and available at: {transcript_txt_path}",
                "data": transcription_data,
                "metadata": transcription_data["metadata"]
            }
        
        # Define a function to process the episode in the background
        def process_episode():
            try:
                try:
                    # Download to temp directory first
                    if not audio_path.exists():
                        logger.info(f"Downloading episode to temp location: {temp_audio_path}")
                        response = requests.get(mp3_url, stream=True)
                        response.raise_for_status()
                        
                        with open(temp_audio_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        
                        # Move to final location
                        temp_audio_path.rename(audio_path)
                        logger.info(f"Moved audio file to: {audio_path}")
                    else:
                        logger.info(f"Episode already downloaded: {audio_path}")
                    
                    # Transcribe the audio
                    logger.info(f"Transcribing audio file: {audio_path}")
                    result = transcribe_audio(str(audio_path))
                    
                    # Save the transcription
                    save_transcription(result, str(transcript_txt_path))
                    logger.info(f"Transcription saved to: {transcript_txt_path}")
                    
                    # Save raw JSON for later processing
                    with open(transcript_json_path, 'w') as f:
                        json.dump(result.model_dump(), f, indent=2)
                    logger.info(f"Raw transcription data saved to: {transcript_json_path}")
                    
                    # Add the new transcript to the vector store
                    logger.info(f"Adding new transcript to vector store: {transcript_txt_path}")
                    initialize_knowledge_base(force_reprocess=False)
                    
                finally:
                    # Clean up temp file if it exists
                    if temp_audio_path.exists():
                        temp_audio_path.unlink()
                
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
    
# @mcp.tool()
# async def create_vector_db_from_series_fountain(show_id: str, episode_limit: int):
#     """Downloads, transcribes, and stores entire series of a fountain.fm show URL into a vector db"""
#     pass

@mcp.tool()
async def search_youtube(query: str):
    """Search youtube for a specific channel/episode etc"""
    logger.info(f"Searching YouTube for: {query}")
    
    try:
        # Import necessary modules
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Initialize the driver
        driver = webdriver.Chrome(options=options)
        
        try:
            # Construct search URL
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            logger.info(f"Fetching search URL: {search_url}")
            
            # Load the page
            driver.get(search_url)
            
            # Wait for video results to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#contents ytd-video-renderer")))
            
            # Extract video information
            videos = []
            video_elements = driver.find_elements(By.CSS_SELECTOR, "div#contents ytd-video-renderer")
            
            for element in video_elements[:10]:  # Limit to first 10 results
                try:
                    # Extract video details
                    title_element = element.find_element(By.CSS_SELECTOR, "#video-title")
                    title = title_element.text.strip()
                    video_url = title_element.get_attribute("href")
                    
                    # Extract channel name
                    try:
                        # Try multiple selectors to find the channel name
                        selectors = [
                            "ytd-channel-name #text-container yt-formatted-string a",
                            "ytd-channel-name #text a",
                            "#channel-info ytd-channel-name a",
                            "#channel-name a"
                        ]
                        channel = ""
                        channel_url = ""
                        for selector in selectors:
                            try:
                                channel_element = element.find_element(By.CSS_SELECTOR, selector)
                                channel = channel_element.text.strip()
                                channel_url = channel_element.get_attribute("href")
                                if channel:  # If we found a non-empty channel name, break
                                    break
                            except:
                                continue
                        
                        if not channel:  # If no channel name found, try one last method
                            channel_element = element.find_element(By.CSS_SELECTOR, "ytd-channel-name")
                            channel = channel_element.text.strip()
                            
                    except Exception as e:
                        logger.error(f"Error extracting channel info: {str(e)}")
                        channel = ""
                        channel_url = ""
                    
                    # Extract video metadata (views, date)
                    metadata_element = element.find_element(By.CSS_SELECTOR, "#metadata-line")
                    metadata_text = metadata_element.text
                    
                    # Parse views and date
                    views_match = re.search(r"([\d,]+)\s+views", metadata_text)
                    views = views_match.group(1) if views_match else "Unknown"
                    
                    date_match = re.search(r"(?:[\d,]+\s+views\s+)?(.*?)$", metadata_text)
                    date = date_match.group(1) if date_match else "Unknown"
                    
                    # Extract description if available
                    try:
                        description = element.find_element(By.CSS_SELECTOR, "#description-text").text.strip()
                    except:
                        description = ""
                    
                    videos.append({
                        "title": title,
                        "url": video_url,
                        "channel": channel,
                        "channel_url": channel_url,
                        "views": views,
                        "date": date,
                        "description": description
                    })
                    
                except Exception as e:
                    logger.error(f"Error extracting video info: {str(e)}")
                    continue
            
            return {
                "status": "success",
                "results": videos,
                "query": query
            }
            
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"YouTube search failed: {str(e)}"}

@mcp.tool()
async def transcribe_youtube(urls: List[str] = Field(description="A list of URLs to transcribe")):
    """Download and transcribe YouTube videos
    
    Args:
        urls: List of YouTube video URLs to transcribe
        
    Returns:
        List of video information including expected paths to transcription files
    """
    
    try:
        # Import necessary modules
        import yt_dlp
        from pathlib import Path
        import tempfile
        from transcribe import transcribe_audio, save_transcription
        import threading
        
        # Create output directories
        output_dir = Path("output")
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': {
                'default': str(temp_dir / '%(title)s.%(ext)s')
            },
            'quiet': True,
            'no_warnings': True
        }
        
        # Helper function to sanitize filenames
        def sanitize_filename(name):
            # Replace problematic characters with underscores
            name = re.sub(r'[<>:"/\\|?*\u0000-\u001F]', '_', name)
            name = re.sub(r'[\s,]+', '_', name)  # Replace spaces and commas
            name = re.sub(r'[^\x00-\x7F]+', '_', name)  # Replace non-ASCII
            name = re.sub(r'_{2,}', '_', name)  # Replace multiple underscores
            name = name.strip('._-')  # Trim special chars from ends
            return name if name else "unnamed"
        
        # List to store video information
        video_info_list = []
        
        # Extract video information for each URL
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for url in urls:
                try:
                    # Get video info without downloading
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get('title', 'Unknown Title')
                    channel_name = info.get('channel', 'Unknown Channel')
                    
                    # Clean up names for use in paths
                    safe_title = sanitize_filename(video_title)
                    safe_channel = sanitize_filename(channel_name)
                    
                    # Create output directories
                    audio_dir = output_dir / "audio" / "youtube" / safe_channel
                    transcripts_dir = output_dir / "transcripts" / "youtube" / safe_channel
                    
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    transcripts_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Define output paths
                    audio_path = audio_dir / f"{safe_title}.mp3"
                    transcript_txt_path = transcripts_dir / f"{safe_title}.txt"
                    transcript_json_path = transcripts_dir / f"{safe_title}.json"
                    
                    # Add to video info list
                    video_info_list.append({
                        "url": url,
                        "title": video_title,
                        "channel": channel_name,
                        "safe_title": safe_title,
                        "safe_channel": safe_channel,
                        "audio_path": audio_path,
                        "transcript_txt_path": transcript_txt_path,
                        "transcript_json_path": transcript_json_path
                    })
                    
                except Exception as e:
                    logger.error(f"Error extracting info for URL {url}: {str(e)}")
                    continue
        
        # Define function to process videos in background
        def process_videos():
            for video_info in video_info_list:
                try:
                    url = video_info["url"]
                    audio_path = video_info["audio_path"]
                    transcript_txt_path = video_info["transcript_txt_path"]
                    transcript_json_path = video_info["transcript_json_path"]
                    
                    # Check if already transcribed
                    if transcript_txt_path.exists() and transcript_json_path.exists():
                        logger.info(f"Video already transcribed: {transcript_txt_path}")
                        continue
                    
                    # Download audio using yt-dlp
                    logger.info(f"Downloading audio from YouTube video: {url}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    
                    # Find the downloaded file and move it
                    downloaded_file = next(temp_dir.glob("*.mp3"))
                    downloaded_file.rename(audio_path)
                    logger.info(f"Moved audio file to: {audio_path}")
                    
                    # Transcribe the audio
                    logger.info(f"Transcribing audio file: {audio_path}")
                    result = transcribe_audio(str(audio_path))
                    
                    # Save the transcription
                    save_transcription(result, str(transcript_txt_path))
                    logger.info(f"Transcription saved to: {transcript_txt_path}")
                    
                    # Save raw JSON for later processing
                    with open(transcript_json_path, 'w') as f:
                        json.dump(result.model_dump(), f, indent=2)
                    logger.info(f"Raw transcription data saved to: {transcript_json_path}")
                    
                    # Add the new transcript to the vector store
                    logger.info(f"Adding new transcript to vector store: {transcript_txt_path}")
                    initialize_knowledge_base(force_reprocess=False)
                    
                except Exception as e:
                    logger.error(f"Error processing video {url}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        # Start background processing
        thread = threading.Thread(target=process_videos)
        thread.daemon = True
        thread.start()
        
        # Return expected paths immediately
        return {
            "status": "processing",
            "message": f"Processing {len(urls)} videos in background",
            "videos": [
                {
                    "url": info["url"],
                    "channel": info["channel"],
                    "title": info["title"],
                    "expected_audio_path": str(info["audio_path"]),
                    "expected_transcript_txt_path": str(info["transcript_txt_path"]),
                    "expected_transcript_json_path": str(info["transcript_json_path"])
                }
                for info in video_info_list
            ]
        }
            
    except Exception as e:
        logger.error(f"Error transcribing videos: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Transcription failed: {str(e)}"}

# MCP resources for accessing transcripts
@mcp.tool()
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

@mcp.tool()
async def list_shows() -> dict:
    """List all available show names with transcripts.
    
    Returns:
        Dictionary containing a list of available show names
    """
    logger.info("Resource request for show listing")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("output/transcripts")
    
    # Check if the directory exists
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # List to store the show names
    shows = []
    
    # Walk through the directory and collect show names (directories)
    for show_dir in transcripts_dir.iterdir():
        if show_dir.is_dir():
            shows.append(show_dir.name)
    
    return {"shows": sorted(shows)}

@mcp.tool()
async def list_episodes(show_name: str) -> dict:
    """List all available episodes for a specific show.
    
    Args:
        show_name: Name of the show to list episodes for
        
    Returns:
        Dictionary containing a list of available episodes for the specified show
    """
    logger.info(f"Resource request for episodes listing for show: {show_name}")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("output/transcripts")
    
    # Check if the directory exists
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # Check if the specific show directory exists
    show_dir = transcripts_dir / show_name
    if not show_dir.exists():
        return {"error": f"Show directory not found: {show_name}"}
    
    # List to store the episode files
    episodes = []
    
    # Recursively find all .txt files in this show directory
    for transcript_file in show_dir.rglob("*.txt"):
        # Get the relative path from the show directory
        rel_path = transcript_file.relative_to(show_dir)
        episodes.append(str(rel_path))
    
    return {
        "show_name": show_name,
        "episodes": sorted(episodes)
    }

# Search resource
@mcp.tool()
async def search_transcripts(
    query: str, 
    max_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search transcript vector store for a specific query with advanced filtering options.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        where: Optional dictionary for metadata filtering with operators:
               - $eq, $ne: Equal/not equal (string, int, float)
               - $gt, $gte, $lt, $lte: Greater/less than (int, float)
               - $in, $nin: In/not in list (string, int, float)
               - $and, $or: Logical operators for combining filters
               Example: {"source": "youtube", "channel": {"$in": ["Channel1", "Channel2"]}}
        where_document: Optional dictionary for document content filtering:
               - $contains: Text the document must contain
               - $not_contains: Text the document must not contain
               Example: {"$contains": "bitcoin", "$not_contains": "ethereum"}
        ids: Optional list of specific document IDs to search within
        
    Returns:
        Dictionary containing the search results with metadata
    """
    logger.info(f"Searching transcripts with query: {query}, max_results: {max_results}")
    logger.info(f"Filters - where: {where}, where_document: {where_document}, ids: {ids}")
    
    include = ["documents", "metadatas", "distances"]
    
    try:
        # Execute the query with all parameters
        results = transcript_collection.query(
            query_texts=[query],
            n_results=max_results,
            where=where,
            where_document=where_document,
            include=include,
            ids=ids
        )
        
        # Format the results in a more user-friendly structure
        formatted_results = {
            "query": query,
            "total_results": len(results["ids"][0]) if results["ids"] else 0,
            "results": []
        }
        
        # Process each result
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                result_item = {
                    "id": results["ids"][0][i]
                }
                
                # Add included fields
                if "documents" in include and "documents" in results:
                    result_item["document"] = results["documents"][0][i]
                
                if "metadatas" in include and "metadatas" in results:
                    result_item["metadata"] = results["metadatas"][0][i]
                
                if "distances" in include and "distances" in results:
                    result_item["distance"] = results["distances"][0][i]
                
                formatted_results["results"].append(result_item)
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching transcripts: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "total_results": 0,
            "results": []
        }
    
@mcp.tool()
async def add_podcast_feed(podcast_rss_url: str = Field(description="The RSS feed URL of the podcast to monitor")):
    """Adds a podcast RSS feed to the monitor list for automatic processing
    
    Args:
        podcast_rss_url: The RSS feed URL of the podcast
        
    Returns:
        Dictionary containing the status and feed information
    """
    logger.info(f"Adding podcast feed to RSS monitor: {podcast_rss_url}")
    
    try:
        import json
        from pathlib import Path
        
        # Create rss directory if it doesn't exist
        rss_dir = Path("rss")
        rss_dir.mkdir(exist_ok=True)
        
        # Path to the podcasts JSON file
        podcasts_json_path = rss_dir / "podcasts.json"
        
        # Load existing feeds or create empty list
        if podcasts_json_path.exists() and podcasts_json_path.stat().st_size > 0:
            try:
                with open(podcasts_json_path, 'r') as f:
                    podcast_feeds = json.load(f)
                if not isinstance(podcast_feeds, list):
                    podcast_feeds = []
            except json.JSONDecodeError:
                logger.error(f"Error reading {podcasts_json_path}, starting with empty list")
                podcast_feeds = []
        else:
            podcast_feeds = []
        
        # Check if this feed is already in our list
        if podcast_rss_url in podcast_feeds:
            return {
                "status": "already_exists",
                "message": f"Podcast feed '{podcast_rss_url}' is already being monitored",
                "feed_url": podcast_rss_url
            }
        
        # Add the new feed URL to our list
        podcast_feeds.append(podcast_rss_url)
        
        # Save the updated list
        with open(podcasts_json_path, 'w') as f:
            json.dump(podcast_feeds, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Added podcast feed '{podcast_rss_url}' to RSS monitor",
            "feed_url": podcast_rss_url,
            "total_feeds": len(podcast_feeds)
        }
            
    except Exception as e:
        logger.error(f"Error adding podcast feed to RSS monitor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Failed to add podcast feed: {str(e)}"}

@mcp.tool()
async def add_substack_feed(substack_rss_url: str = Field(description="The RSS feed URL of the Substack to monitor")):
    """Adds a Substack RSS feed to the monitor list for automatic processing
    
    Args:
        substack_rss_url: The RSS feed URL of the Substack
        
    Returns:
        Dictionary containing the status and feed information
    """
    logger.info(f"Adding Substack feed to RSS monitor: {substack_rss_url}")
    
    try:
        import json
        from pathlib import Path
        
        # Create rss directory if it doesn't exist
        rss_dir = Path("rss")
        rss_dir.mkdir(exist_ok=True)
        
        # Path to the substack JSON file
        substack_json_path = rss_dir / "substack.json"
        
        # Load existing feeds or create empty list
        if substack_json_path.exists() and substack_json_path.stat().st_size > 0:
            try:
                with open(substack_json_path, 'r') as f:
                    substack_feeds = json.load(f)
                if not isinstance(substack_feeds, list):
                    substack_feeds = []
            except json.JSONDecodeError:
                logger.error(f"Error reading {substack_json_path}, starting with empty list")
                substack_feeds = []
        else:
            substack_feeds = []
        
        # Check if this feed is already in our list
        if substack_rss_url in substack_feeds:
            return {
                "status": "already_exists",
                "message": f"Substack feed '{substack_rss_url}' is already being monitored",
                "feed_url": substack_rss_url
            }
        
        # Add the new feed URL to our list
        substack_feeds.append(substack_rss_url)
        
        # Save the updated list
        with open(substack_json_path, 'w') as f:
            json.dump(substack_feeds, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Added Substack feed '{substack_rss_url}' to RSS monitor",
            "feed_url": substack_rss_url,
            "total_feeds": len(substack_feeds)
        }
            
    except Exception as e:
        logger.error(f"Error adding Substack feed to RSS monitor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Failed to add Substack feed: {str(e)}"}

@mcp.tool()
async def run_rss_monitor():
    """Manually trigger the RSS monitoring cycle to check for new content
    
    Returns:
        Dictionary containing the status and results of the monitoring cycle
    """
    logger.info("Manually triggering RSS monitoring cycle")
    
    try:
        from enhanced_rss_monitor import RSSMonitor
        
        # Create and run the monitor
        monitor = RSSMonitor()
        monitor.run_monitoring_cycle()
        
        return {
            "status": "success",
            "message": "RSS monitoring cycle completed successfully"
        }
            
    except Exception as e:
        logger.error(f"Error running RSS monitor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"RSS monitoring failed: {str(e)}"}

@mcp.tool()
async def list_rss_feeds():
    """List all RSS feeds currently being monitored
    
    Returns:
        Dictionary containing all monitored feeds organized by type
    """
    logger.info("Listing all RSS feeds")
    
    try:
        import json
        from pathlib import Path
        
        rss_dir = Path("rss")
        result = {
            "youtube_channels": [],
            "podcast_feeds": [],
            "substack_feeds": []
        }
        
        # Load YouTube channels
        youtube_json_path = rss_dir / "youtube.json"
        if youtube_json_path.exists() and youtube_json_path.stat().st_size > 0:
            try:
                with open(youtube_json_path, 'r') as f:
                    result["youtube_channels"] = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error reading {youtube_json_path}")
        
        # Load podcast feeds
        podcasts_json_path = rss_dir / "podcasts.json"
        if podcasts_json_path.exists() and podcasts_json_path.stat().st_size > 0:
            try:
                with open(podcasts_json_path, 'r') as f:
                    result["podcast_feeds"] = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error reading {podcasts_json_path}")
        
        # Load Substack feeds
        substack_json_path = rss_dir / "substack.json"
        if substack_json_path.exists() and substack_json_path.stat().st_size > 0:
            try:
                with open(substack_json_path, 'r') as f:
                    result["substack_feeds"] = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error reading {substack_json_path}")
        
        return result
            
    except Exception as e:
        logger.error(f"Error listing RSS feeds: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Failed to list RSS feeds: {str(e)}"}

@mcp.tool()
async def monitor_yt_rss_feed(youtube_channel_url: str = Field(description="The URL of the YouTube channel to monitor")):
    """Adds a YouTube channel to the RSS feed monitor to automatically process new videos
    
    Args:
        youtube_channel_url: The URL of the YouTube channel homepage or channel ID
        
    Returns:
        Dictionary containing the status and channel information
    """
    logger.info(f"Adding YouTube channel to RSS monitor: {youtube_channel_url}")
    
    try:
        # Import necessary modules
        import re
        import json
        import requests
        from pathlib import Path
        
        # Create rss directory if it doesn't exist
        rss_dir = Path("rss")
        rss_dir.mkdir(exist_ok=True)
        
        # Path to the YouTube channels JSON file
        youtube_json_path = rss_dir / "youtube.json"
        
        # Load existing channels or create empty list
        if youtube_json_path.exists() and youtube_json_path.stat().st_size > 0:
            try:
                with open(youtube_json_path, 'r') as f:
                    channel_ids = json.load(f)
                if not isinstance(channel_ids, list):
                    channel_ids = []
            except json.JSONDecodeError:
                logger.error(f"Error reading {youtube_json_path}, starting with empty list")
                channel_ids = []
        else:
            channel_ids = []
        
        # Extract channel ID from URL
        channel_id = None
        
        # Check if input is already a channel ID
        if re.match(r'^[A-Za-z0-9_-]{24}$', youtube_channel_url):
            channel_id = youtube_channel_url
        else:
            # Process as URL
            # Handle different YouTube URL formats
            if "youtube.com/channel/" in youtube_channel_url:
                # Direct channel URL format: youtube.com/channel/CHANNEL_ID
                channel_id_match = re.search(r'youtube\.com/channel/([A-Za-z0-9_-]+)', youtube_channel_url)
                if channel_id_match:
                    channel_id = channel_id_match.group(1)
            elif "youtube.com/@" in youtube_channel_url:
                # Handle @username format: youtube.com/@username
                username_match = re.search(r'youtube\.com/@([A-Za-z0-9_-]+)', youtube_channel_url)
                if username_match:
                    username = username_match.group(1)
                    
                    # Fetch the channel page to extract the channel ID
                    response = requests.get(f"https://www.youtube.com/@{username}")
                    if response.status_code == 200:
                        # Look for channel ID in the HTML
                        html_content = response.text
                        
                        # Try different patterns to find channel ID
                        patterns = [
                            r'"channelId":"([A-Za-z0-9_-]+)"',
                            r'channel/([A-Za-z0-9_-]+)',
                            r'externalId":"([A-Za-z0-9_-]+)"'
                        ]
                        
                        for pattern in patterns:
                            channel_match = re.search(pattern, html_content)
                            if channel_match:
                                channel_id = channel_match.group(1)
                                break
            elif "youtube.com/c/" in youtube_channel_url:
                # Handle /c/ format: youtube.com/c/customname
                custom_match = re.search(r'youtube\.com/c/([A-Za-z0-9_-]+)', youtube_channel_url)
                if custom_match:
                    custom_name = custom_match.group(1)
                    
                    # Fetch the channel page to extract the channel ID
                    response = requests.get(f"https://www.youtube.com/c/{custom_name}")
                    if response.status_code == 200:
                        # Look for channel ID in the HTML
                        html_content = response.text
                        
                        # Try different patterns to find channel ID
                        patterns = [
                            r'"channelId":"([A-Za-z0-9_-]+)"',
                            r'channel/([A-Za-z0-9_-]+)',
                            r'externalId":"([A-Za-z0-9_-]+)"'
                        ]
                        
                        for pattern in patterns:
                            channel_match = re.search(pattern, html_content)
                            if channel_match:
                                channel_id = channel_match.group(1)
                                break
            else:
                # Try to handle as a generic YouTube URL
                # Fetch the page and look for channel ID
                response = requests.get(youtube_channel_url)
                if response.status_code == 200:
                    html_content = response.text
                    
                    # Try different patterns to find channel ID
                    patterns = [
                        r'"channelId":"([A-Za-z0-9_-]+)"',
                        r'channel/([A-Za-z0-9_-]+)',
                        r'externalId":"([A-Za-z0-9_-]+)"'
                    ]
                    
                    for pattern in patterns:
                        channel_match = re.search(pattern, html_content)
                        if channel_match:
                            channel_id = channel_match.group(1)
                            break
        
        # If we couldn't extract a channel ID, return an error
        if not channel_id:
            return {
                "status": "error",
                "message": "Could not extract channel ID from the provided URL",
                "url": youtube_channel_url
            }
        
        # Check if this channel is already in our list
        if channel_id in channel_ids:
            return {
                "status": "already_exists",
                "message": f"Channel ID '{channel_id}' is already being monitored",
                "channel_id": channel_id
            }
        
        # Add the new channel ID to our list
        channel_ids.append(channel_id)
        
        # Save the updated list
        with open(youtube_json_path, 'w') as f:
            json.dump(channel_ids, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Added channel ID '{channel_id}' to RSS monitor",
            "channel_id": channel_id,
            "total_channels": len(channel_ids)
        }
            
    except Exception as e:
        logger.error(f"Error adding YouTube channel to RSS monitor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Failed to add channel: {str(e)}"}    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_knowledge_base()
    asyncio.run(mcp.run_sse_async())
