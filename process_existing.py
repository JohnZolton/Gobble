#!/usr/bin/env python3
"""
Process existing podcast episodes.

This script processes existing podcast episodes in the output/audio directory,
transcribing them and identifying advertisement segments.
"""

import os
import sys
from pathlib import Path
from main import PodcastProcessor, DEFAULT_OUTPUT_DIR, DEFAULT_WHISPER_MODEL
from process_transcripts import TranscriptProcessor
from find_ads import find_ads_in_file

def process_existing_episodes(show_name=None, limit=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Process existing episodes in the output/audio directory.
    
    Args:
        show_name: Name of the show to process (if None, process all shows)
        limit: Maximum number of episodes to process (if None, process all episodes)
    """
    # Initialize the processors
    processor = PodcastProcessor(
        url="dummy_url",
        output_dir=output_dir,
        whisper_model=DEFAULT_WHISPER_MODEL,
    )
    
    # Initialize the transcript processor for improved summaries
    transcript_processor = TranscriptProcessor(output_dir=output_dir)
    
    # Get the audio directory
    audio_dir = processor.audio_dir
    
    # Get all show directories
    show_dirs = [d for d in audio_dir.iterdir() if d.is_dir()]
    
    if show_name:
        # Filter to only the specified show
        show_dirs = [d for d in show_dirs if show_name.lower() in d.name.lower()]
        
    if not show_dirs:
        print(f"No shows found in {audio_dir}")
        return
    
    # Process each show
    for show_dir in show_dirs:
        print(f"\nProcessing show: {show_dir.name}")
        
        # Get all MP3 files in the show directory
        mp3_files = sorted([f for f in show_dir.iterdir() if f.suffix.lower() == '.mp3'])
        
        if limit:
            mp3_files = mp3_files[:limit]
        
        print(f"Found {len(mp3_files)} episodes to process")
        
        # Process each episode
        for i, mp3_file in enumerate(mp3_files):
            print(f"\nProcessing episode {i+1}/{len(mp3_files)}: {mp3_file.stem}")
            
            # Create episode info dictionary
            episode_info = {
                "title": mp3_file.stem,
                "show_name": show_dir.name,
                "audio_path": str(mp3_file)
            }
            
            try:
                # Transcribe the episode
                transcript = processor.transcribe_episode(mp3_file)
                
                # Save transcript to TXT file for easier processing
                transcript_txt_path = processor.transcripts_dir / show_dir.name / f"{mp3_file.stem}.txt"
                
                # Identify ads using the improved find_ads_in_file function
                ads = find_ads_in_file(str(transcript_txt_path))
                print(f"Found {len(ads)} ads in {mp3_file.stem}")
                
                # Generate summary using the improved TranscriptProcessor
                summary = transcript_processor.generate_summary(transcript, episode_info, ads)
                
                print(f"Successfully processed {mp3_file.stem}")
                print(f"Transcript saved to: {processor.transcripts_dir / show_dir.name / f'{mp3_file.stem}.txt'}")
                print(f"Ad information saved to: {processor.processed_dir / show_dir.name / f'{mp3_file.stem}_ads.json'}")
                print(f"Summary saved to: {processor.summaries_dir / show_dir.name / f'{mp3_file.stem}_summary.json'}")
                
            except Exception as e:
                print(f"Error processing episode {mp3_file.stem}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process existing podcast episodes")
    parser.add_argument("--show", help="Name of the show to process (if not specified, process all shows)")
    parser.add_argument("--limit", type=int, help="Maximum number of episodes to process per show")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    process_existing_episodes(show_name=args.show, limit=args.limit, output_dir=args.output_dir)
