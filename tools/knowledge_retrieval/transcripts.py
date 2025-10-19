import os
import json
import logging
import tempfile
from typing import Optional, Sequence, Dict, Any, Tuple, List
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from pathlib import Path
import re

from tools.knowledge_retrieval.vector_store import search_topics

logger = logging.getLogger(__name__)

async def get_transcript(filename: str) -> str:
    """Retrieve the content of a transcript file.
    
    Args:
        filename: Path to the transcript file within the directory {showname}/{transcript_title}
        
    Returns:
        Content of the transcript file as text
    """
    logger.info(f"Resource request for transcript: {filename}")
    
    # Construct the full path to the transcript file
    transcript_path = Path("knowledge/youtube") / filename
    
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

async def list_shows() -> dict:
    """List all available show names with transcripts.
    
    Returns:
        Dictionary containing a list of available show names
    """
    logger.info("Resource request for show listing")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("knowledge/youtube")
    
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

async def list_episodes(show_name: str) -> dict:
    """List all available episodes for a specific show.
    
    Args:
        show_name: Name of the show to list episodes for
        
    Returns:
        Dictionary containing a list of available episodes for the specified show
    """
    logger.info(f"Resource request for episodes listing for show: {show_name}")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("knowledge/youtube")
    
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

async def search_episodes(
    query: str,
    show_name: Optional[str] = None,
    max_results: int = 10,
    context_chars: int = 200
) -> Dict[str, Any]:
    """Search for episodes containing specific text using OpenSearch (text-based search).
    
    This performs a case-insensitive text search across all transcript files to find
    episodes that contain the search query. Useful for finding specific episodes by
    keywords, phrases, or topics mentioned in the transcript.
    
    Args:
        query: Text to search for in transcripts
        show_name: Optional show name to limit search to a specific show
        max_results: Maximum number of matching episodes to return (default: 10)
        context_chars: Number of characters of context to show around each match (default: 200)
        
    Returns:
        Dictionary containing matching episodes with context snippets
    """
    logger.info(f"Text search for episodes with query: {query}, show: {show_name}")
    
    transcripts_dir = Path("knowledge/youtube")
    
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # Determine search path
    if show_name:
        search_path = transcripts_dir / show_name
        if not search_path.exists():
            return {"error": f"Show directory not found: {show_name}"}
    else:
        search_path = transcripts_dir
    
    results = []
    query_lower = query.lower()
    
    # Search through all transcript files
    for transcript_file in search_path.rglob("*.txt"):
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Case-insensitive search
            content_lower = content.lower()
            if query_lower not in content_lower:
                continue
            
            # Get relative path from knowledge/youtube
            rel_path = transcript_file.relative_to(transcripts_dir)
            show = rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown"
            episode = str(rel_path.relative_to(show))
            
            # Find all matches and extract context
            matches = []
            start = 0
            while True:
                pos = content_lower.find(query_lower, start)
                if pos == -1:
                    break
                
                # Extract context around the match
                context_start = max(0, pos - context_chars)
                context_end = min(len(content), pos + len(query) + context_chars)
                context = content[context_start:context_end]
                
                # Add ellipsis if truncated
                if context_start > 0:
                    context = "..." + context
                if context_end < len(content):
                    context = context + "..."
                
                matches.append({
                    "position": pos,
                    "context": context.strip()
                })
                
                start = pos + 1
                
                # Limit matches per file to avoid overwhelming results
                if len(matches) >= 3:
                    break
            
            results.append({
                "show": show,
                "episode": episode,
                "file_path": str(rel_path),
                "match_count": len(matches),
                "matches": matches[:3]  # Return up to 3 context snippets
            })
            
            # Stop if we've reached max_results
            if len(results) >= max_results:
                break
                
        except Exception as e:
            logger.error(f"Error reading {transcript_file}: {str(e)}")
            continue
    
    return {
        "query": query,
        "show_filter": show_name,
        "total_episodes_found": len(results),
        "results": results
    }


    
    
def register_knowledgebase_tools(mcp):
    mcp.tool()(get_transcript)
    mcp.tool()(list_shows)
    mcp.tool()(list_episodes)
    mcp.tool()(search_episodes)
    mcp.tool()(search_topics)
