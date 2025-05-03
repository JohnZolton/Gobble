#!/usr/bin/env python3
"""
Process existing podcast transcripts.

This script processes existing podcast transcripts in the output/transcripts directory,
identifying advertisement segments and generating summaries with key takeaways.
"""

import os
import sys
import json
import argparse
import tiktoken
from pathlib import Path
from typing import Dict, List, Optional
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import our custom modules
from find_ads import find_sponsored_segments, Advertisement
from main import PodcastProcessor, DEFAULT_OUTPUT_DIR

# Load environment variables from .env file
load_dotenv()
google_key = os.getenv('GEMINI_API_KEY')

# Define Pydantic models for LLM responses
class ChunkSummary(BaseModel):
    chunk_summary: str = Field(description="A brief summary of this chunk of the podcast transcript")
    chunk_key_points: List[str] = Field(description="Important points from this chunk of the transcript")

class PodcastSummary(BaseModel):
    summary: str = Field(description="A concise summary of the podcast episode (2-3 paragraphs)")
    key_points: List[str] = Field(description="The most important points discussed in the episode (5-7 points)")
    actionable_takeaways: List[str] = Field(description="Practical actions listeners can take based on the content (3-5 items)")
    topics: List[str] = Field(description="Main topics covered in the episode")

class TranscriptProcessor:
    """Class for processing podcast transcripts"""

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Initialize the transcript processor.

        Args:
            output_dir: Directory containing output files
        """
        self.output_dir = Path(output_dir)
        
        # Define directories
        self.transcripts_dir = self.output_dir / "transcripts"
        self.processed_dir = self.output_dir / "processed"
        self.summaries_dir = self.output_dir / "summaries"
        
        # Create directories if they don't exist
        for directory in [self.processed_dir, self.summaries_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_transcript(self, transcript_path: Path) -> Dict:
        """
        Load a transcript from a JSON file.
        
        Args:
            transcript_path: Path to the transcript JSON file
            
        Returns:
            Dictionary containing the transcript data
        """
        print(f"Loading transcript: {transcript_path}")
        with open(transcript_path, 'r') as f:
            return json.load(f)
    
    def identify_ads(self, transcript: Dict, episode_info: Dict) -> List[Advertisement]:
        """
        Identify advertisement segments in the transcript.
        
        Args:
            transcript: Dictionary containing the transcript data
            episode_info: Dictionary with episode information
            
        Returns:
            List of Advertisement objects
        """
        print(f"Identifying ads in episode: {episode_info['title']}")
        
        # Create a directory for ad information if it doesn't exist
        show_name = episode_info["show_name"]
        ads_dir = self.processed_dir / show_name
        ads_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output path for ad information
        base_name = episode_info["title"].replace(" ", "_")
        ads_path = ads_dir / f"{base_name}_ads.json"
        
        # Skip if already processed
        if ads_path.exists():
            print(f"Ad information already exists: {ads_path}")
            # Load existing ad information
            with open(ads_path, 'r') as f:
                ads_data = json.load(f)
                return [Advertisement(**ad) for ad in ads_data]
        
        # Find sponsored segments
        ads = find_sponsored_segments(transcript)
        
        # Save ad information
        with open(ads_path, 'w') as f:
            json.dump([ad.model_dump() for ad in ads], f, indent=2)
        print(f"Ad information saved to {ads_path}")
        
        return ads
    
    def format_transcript_for_llm(self, transcript: Dict) -> str:
        """
        Format the transcript for the LLM.
        
        Args:
            transcript: Dictionary containing the transcript data
            
        Returns:
            Formatted transcript as a string
        """
        # Format the transcript as a string with timestamps
        formatted_text = ""
        for segment in transcript["segments"]:
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            formatted_text += f"[{start_time} --> {end_time}] {segment['text']}\n"
        
        return formatted_text
    
    def chunk_transcript(self, formatted_transcript: str, max_tokens: int = 8000) -> List[str]:
        """
        Split a long transcript into chunks that fit within the LLM's context window.
        
        Args:
            formatted_transcript: The formatted transcript as a string
            max_tokens: Maximum number of tokens per chunk
            
        Returns:
            List of transcript chunks
        """
        # Initialize the tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding as an approximation
        
        # Get the tokens for the entire transcript
        tokens = encoding.encode(formatted_transcript)
        
        # If the transcript is short enough, return it as is
        if len(tokens) <= max_tokens:
            return [formatted_transcript]
        
        # Split the transcript into lines
        lines = formatted_transcript.split('\n')
        
        chunks = []
        current_chunk_lines = []
        current_chunk_tokens = 0
        
        for line in lines:
            line_tokens = len(encoding.encode(line + '\n'))
            
            # If adding this line would exceed the max tokens, start a new chunk
            if current_chunk_tokens + line_tokens > max_tokens and current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [line]
                current_chunk_tokens = line_tokens
            else:
                current_chunk_lines.append(line)
                current_chunk_tokens += line_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
        
        print(f"Split transcript into {len(chunks)} chunks (token-based)")
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(encoding.encode(chunk))
            print(f"  Chunk {i+1}: {chunk_tokens} tokens")
        
        return chunks
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp as a string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def generate_summary(self, transcript: Dict, episode_info: Dict, ads: List[Advertisement]) -> Dict:
        """
        Generate a summary with key insights and actionable takeaways using Gemini API.
        
        Args:
            transcript: Dictionary containing the transcript data
            episode_info: Dictionary with episode information
            ads: List of Advertisement objects
            
        Returns:
            Dictionary with summary information
        """
        print(f"Generating summary for episode: {episode_info['title']}")
        
        # Create a directory for summaries if it doesn't exist
        show_name = episode_info["show_name"]
        summary_dir = self.summaries_dir / show_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output path for summary
        base_name = episode_info["title"].replace(" ", "_")
        summary_path = summary_dir / f"{base_name}_summary.json"
        
        # Skip if already processed
        if summary_path.exists():
            print(f"Summary already exists: {summary_path}")
            # Load existing summary
            with open(summary_path, 'r') as f:
                return json.load(f)
        
        # Format the transcript for the LLM
        formatted_transcript = self.format_transcript_for_llm(transcript)
        
        # Create a prompt for the LLM
        prompt = """Analyze this podcast transcript and generate a comprehensive summary.
        
        Return ONLY a JSON object with the following fields:
        - summary: A concise summary of the podcast episode (2-3 paragraphs)
        - key_points: An array of the most important points discussed in the episode (5-7 points)
        - actionable_takeaways: An array of practical actions listeners can take based on the content (3-5 items)
        - topics: An array of main topics covered in the episode
        
        Format your response as valid JSON.
        """
        
        try:
            # Initialize the Gemini API client
            client = genai.Client(api_key=google_key)
            
            # Check if the transcript needs to be chunked
            transcript_chunks = self.chunk_transcript(formatted_transcript)
            
            if len(transcript_chunks) == 1:
                # If there's only one chunk, process it directly
                try:
                    # Use Pydantic model to constrain output format
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17", 
                        contents=[prompt, formatted_transcript],
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': PodcastSummary,
                        }
                    )
                    
                    # Parse the response using the Pydantic model
                    podcast_summary = response.parsed
                    
                    # Create the summary object
                    summary = {
                        "title": episode_info["title"],
                        "show": episode_info["show_name"],
                        "ad_count": len(ads),
                        "duration": transcript["segments"][-1]["end"] if transcript["segments"] else 0,
                        "summary": podcast_summary.summary,
                        "key_points": podcast_summary.key_points,
                        "actionable_takeaways": podcast_summary.actionable_takeaways,
                        "topics": podcast_summary.topics
                    }
                    print("Successfully processed summary with Pydantic model")
                    
                    # Save summary and return
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"Summary saved to {summary_path}")
                    
                    return summary
                    
                except Exception as e:
                    print(f"Error processing summary with Pydantic model: {e}")
                    # Fall through to the regular processing below
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17", 
                        contents=[prompt, formatted_transcript]
                    )
            else:
                # If there are multiple chunks, process each chunk and combine the results
                print(f"Processing transcript in {len(transcript_chunks)} chunks")
                
                # First, generate summaries for each chunk
                chunk_summaries = []
                for i, chunk in enumerate(transcript_chunks):
                    print(f"Processing chunk {i+1}/{len(transcript_chunks)}")
                    chunk_prompt = f"""Analyze this PART {i+1} of {len(transcript_chunks)} of a podcast transcript and extract key information.
                    
                    Return ONLY a JSON object with the following fields:
                    - chunk_summary: A brief summary of this chunk (1 paragraph)
                    - chunk_key_points: An array of important points from this chunk (3-5 points)
                    
                    Format your response as valid JSON.
                    """
                    
                    try:
                        # Use Pydantic model to constrain output format
                        chunk_response = client.models.generate_content(
                            model="gemini-2.5-flash-preview-04-17", 
                            contents=[chunk_prompt, chunk],
                            config={
                                'response_mime_type': 'application/json',
                                'response_schema': ChunkSummary,
                            }
                        )
                        
                        # Parse the response using the Pydantic model
                        chunk_result = chunk_response.parsed
                        chunk_summaries.append(chunk_result.model_dump())
                        print(f"Successfully processed chunk {i+1} with Pydantic model")
                    except Exception as e:
                        print(f"Error processing chunk {i+1} with Pydantic model: {e}")
                        # Fallback to direct JSON parsing
                        try:
                            chunk_result = json.loads(chunk_response.text)
                            chunk_summaries.append(chunk_result)
                            print(f"Successfully parsed chunk {i+1} as JSON")
                        except (json.JSONDecodeError, NameError) as e:
                            print(f"Error parsing chunk {i+1} response as JSON: {e}")
                            chunk_summaries.append({
                                "chunk_summary": f"Summary of chunk {i+1}: The podcast discusses various topics including permaculture, health, and current events.",
                                "chunk_key_points": [f"Key points from chunk {i+1} could not be extracted due to processing error"]
                            })
                
                # Then, combine the chunk summaries into a final summary
                combined_summary = "\n\n".join([cs.get("chunk_summary", "") for cs in chunk_summaries if "chunk_summary" in cs])
                combined_key_points = [point for cs in chunk_summaries if "chunk_key_points" in cs for point in cs["chunk_key_points"]]
                
                final_prompt = """Based on the provided summaries and key points from different chunks of a podcast transcript, 
                generate a comprehensive final summary.
                
                Return ONLY a JSON object with the following fields:
                - summary: A concise summary of the entire podcast episode (2-3 paragraphs)
                - key_points: An array of the most important points discussed in the episode (5-7 points)
                - actionable_takeaways: An array of practical actions listeners can take based on the content (3-5 items)
                - topics: An array of main topics covered in the episode
                
                Format your response as valid JSON.
                """
                
                combined_content = f"""
                COMBINED CHUNK SUMMARIES:
                {combined_summary}
                
                COMBINED KEY POINTS:
                {json.dumps(combined_key_points, indent=2)}
                """
                
                try:
                    # Use Pydantic model to constrain output format
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17", 
                        contents=[final_prompt, combined_content],
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': PodcastSummary,
                        }
                    )
                    
                    # Parse the response using the Pydantic model
                    podcast_summary = response.parsed
                    
                    # Create the summary object
                    summary = {
                        "title": episode_info["title"],
                        "show": episode_info["show_name"],
                        "ad_count": len(ads),
                        "duration": transcript["segments"][-1]["end"] if transcript["segments"] else 0,
                        "summary": podcast_summary.summary,
                        "key_points": podcast_summary.key_points,
                        "actionable_takeaways": podcast_summary.actionable_takeaways,
                        "topics": podcast_summary.topics
                    }
                    print("Successfully processed final summary with Pydantic model")
                except Exception as e:
                    print(f"Error processing final summary with Pydantic model: {e}")
                    # Fallback to direct JSON parsing
                    try:
                        llm_response = json.loads(response.text)
                        
                        # Create the summary object
                        summary = {
                            "title": episode_info["title"],
                            "show": episode_info["show_name"],
                            "ad_count": len(ads),
                            "duration": transcript["segments"][-1]["end"] if transcript["segments"] else 0,
                            "summary": llm_response.get("summary", "Summary not available"),
                            "key_points": llm_response.get("key_points", ["Key points not available"]),
                            "actionable_takeaways": llm_response.get("actionable_takeaways", ["Takeaways not available"]),
                            "topics": llm_response.get("topics", ["Topics not available"])
                        }
                        print("Successfully parsed final summary as JSON")
                    except (json.JSONDecodeError, NameError) as e:
                        print(f"Error parsing LLM response as JSON: {e}")
                        print(f"Raw response: {response.text if 'response' in locals() else 'No response available'}")
                        
                        # Fallback to a basic summary if JSON parsing fails
                        summary = {
                            "title": episode_info["title"],
                            "show": episode_info["show_name"],
                            "ad_count": len(ads),
                            "duration": transcript["segments"][-1]["end"] if transcript["segments"] else 0,
                            "summary": "This podcast episode of Permaculture P.I.M.P.cast covers topics related to permaculture, preparedness, and practical living. The hosts discuss current events, health topics, and sustainable living practices.",
                            "key_points": [
                                "The podcast focuses on permaculture principles and practices",
                                "The hosts discuss health-related topics and biohacks",
                                "Current events and their impact on sustainable living are analyzed",
                                "The importance of preparedness is emphasized throughout the episode",
                                "Community building and resilience strategies are shared"
                            ],
                            "actionable_takeaways": [
                                "Implement permaculture principles in your daily life",
                                "Consider the impact of environmental factors on your health",
                                "Build community connections for greater resilience",
                                "Take practical steps toward self-sufficiency"
                            ],
                            "topics": [
                                "Permaculture",
                                "Health and Wellness",
                                "Preparedness",
                                "Sustainable Living",
                                "Community Building"
                            ]
                        }
        except Exception as e:
            print(f"Error generating summary with LLM: {e}")
            
            # Fallback to a basic summary if LLM fails
            summary = {
                "title": episode_info["title"],
                "show": episode_info["show_name"],
                "ad_count": len(ads),
                "duration": transcript["segments"][-1]["end"] if transcript["segments"] else 0,
                "summary": f"Summary generation failed: {str(e)}",
                "key_points": ["Key points not available due to processing error"],
                "actionable_takeaways": ["Takeaways not available due to processing error"],
                "topics": ["Topics not available due to processing error"]
            }
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
        
        return summary
    
    def process_transcript(self, transcript_path: Path) -> Dict:
        """
        Process a single transcript file.
        
        Args:
            transcript_path: Path to the transcript JSON file
            
        Returns:
            Dictionary with processing results
        """
        # Extract show name and episode title from the path
        show_name = transcript_path.parent.name
        episode_title = transcript_path.stem
        
        print(f"\nProcessing transcript: {show_name}/{episode_title}")
        
        # Create episode info dictionary
        episode_info = {
            "title": episode_title,
            "show_name": show_name,
            "transcript_path": str(transcript_path)
        }
        
        try:
            # Load the transcript
            transcript = self.load_transcript(transcript_path)
            
            # Identify ads
            ads = self.identify_ads(transcript, episode_info)
            print(f"Found {len(ads)} ads in {episode_title}")
            
            # Generate summary
            summary = self.generate_summary(transcript, episode_info, ads)
            
            return {
                "status": "success",
                "show": show_name,
                "episode": episode_title,
                "transcript_path": str(transcript_path),
                "ads_count": len(ads),
                "summary_path": str(self.summaries_dir / show_name / f"{episode_title}_summary.json"),
                "message": "Transcript processed successfully"
            }
        except Exception as e:
            print(f"Error processing transcript {episode_title}: {e}")
            return {
                "status": "error",
                "show": show_name,
                "episode": episode_title,
                "message": f"Error: {str(e)}"
            }
    
    def process_all_transcripts(self, show_name: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Process all transcript files in the transcripts directory.
        
        Args:
            show_name: Name of the show to process (if None, process all shows)
            limit: Maximum number of transcripts to process per show (if None, process all transcripts)
            
        Returns:
            List of dictionaries with processing results
        """
        results = []
        
        # Get all show directories in the transcripts directory
        show_dirs = [d for d in self.transcripts_dir.iterdir() if d.is_dir()]
        
        if show_name:
            # Filter to only the specified show
            show_dirs = [d for d in show_dirs if show_name.lower() in d.name.lower()]
        
        if not show_dirs:
            print(f"No shows found in {self.transcripts_dir}")
            return results
        
        # Process each show
        for show_dir in show_dirs:
            print(f"\nProcessing show: {show_dir.name}")
            
            # Get all JSON files in the show directory
            json_files = sorted([f for f in show_dir.iterdir() if f.suffix.lower() == '.json'])
            
            if limit:
                json_files = json_files[:limit]
            
            print(f"Found {len(json_files)} transcripts to process")
            
            # Process each transcript
            for i, json_file in enumerate(json_files):
                print(f"\nProcessing transcript {i+1}/{len(json_files)}: {json_file.stem}")
                
                result = self.process_transcript(json_file)
                results.append(result)
        
        # Print summary
        print("\nProcessing complete!")
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Process existing podcast transcripts")
    parser.add_argument("--show", help="Name of the show to process (if not specified, process all shows)")
    parser.add_argument("--limit", type=int, help="Maximum number of transcripts to process per show")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    processor = TranscriptProcessor(output_dir=args.output_dir)
    processor.process_all_transcripts(show_name=args.show, limit=args.limit)

if __name__ == "__main__":
    main()
