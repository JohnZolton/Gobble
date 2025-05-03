#!/usr/bin/env python3
"""
Gobble - A podcast processor

This tool takes a podcast homepage URL, extracts episode links using Selenium,
downloads them, transcribes the audio, identifies advertisement segments,
removes them, and generates AI summaries with key insights and actionable takeaways.
"""

import argparse
import json
import os
import re
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import subprocess

import whisper
from tqdm import tqdm

# Import Selenium components
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys # Added for scrolling

from dotenv import load_dotenv # Keep dotenv for potential future use

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from transcribe import transcribe_audio, save_transcription, TranscriptionResult
from find_ads import find_sponsored_segments, Advertisement

# Constants
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_NUM_EPISODES = 5 # Keep for potential future use
DEFAULT_WHISPER_MODEL = "base" # Keep for potential future use

class PodcastProcessor:
    """Main class for processing podcasts"""

    def __init__(
        self,
        url: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        num_episodes: int = DEFAULT_NUM_EPISODES,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
    ):
        """
        Initialize the podcast processor.

        Args:
            url: URL of the podcast homepage
            output_dir: Directory to save output files
            num_episodes: Number of episodes to process
            whisper_model: Whisper model to use for transcription
        """
        self.url = url
        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.whisper_model = whisper_model
        
        # Create output directories (keep for potential future use)
        self.audio_dir = self.output_dir / "audio"
        self.transcripts_dir = self.output_dir / "transcripts"
        self.processed_dir = self.output_dir / "processed"
        self.summaries_dir = self.output_dir / "summaries"
        
        for directory in [self.audio_dir, self.transcripts_dir, 
                         self.processed_dir, self.summaries_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model (commented out for now)
        self.model = None  # Lazy loading
        
        # Configure Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode (no browser window)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)


    def __del__(self):
        """Close the browser when the object is deleted"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()

    def scroll_to_bottom(self):
        """Scroll to the bottom of the page to load dynamic content."""
        print("Scrolling to load all episodes...")
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new content to load
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        print("Finished scrolling.")

    def extract_episode_page_links_selenium(self, url: str) -> List[Dict]:
        """
        Extract links to individual episode pages from the homepage URL using Selenium.
        
        Args:
            url: URL of the podcast homepage
            
        Returns:
            List of episode dictionaries with title and URL to the episode page
        """
        print(f"Extracting episode page links from {url} using Selenium")
        
        try:
            self.driver.get(url)
            
            # Scroll to load all content
            self.scroll_to_bottom()
            
            episode_links = []
            
            # Find all 'a' tags with 'episode' in the href
            link_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='episode']")
            
            # Use a set to store unique URLs to avoid duplicates
            unique_urls = set()
            
            for element in link_elements:
                try:
                    href = element.get_attribute("href")
                    # Ensure the link is a full URL and contains "/episode/"
                    if href and href.startswith("http") and "/episode/" in href:
                        # Try to find a title associated with this link
                        # This might require navigating up the DOM tree or looking for sibling elements
                        # For simplicity, let's try to get the text of the link itself or a nearby element
                        title = element.text.strip()
                        if not title:
                             # Try finding a title in a parent or sibling element
                             try:
                                 parent = element.find_element(By.XPATH, "./..") # Parent element
                                 title_elem = parent.find_element(By.CSS_SELECTOR, "h1, h2, h3, .title, .episode-title")
                                 title = title_elem.text.strip()
                             except NoSuchElementException:
                                 title = f"Episode {len(episode_links) + 1}" # Default title
                                 
                        if href not in unique_urls:
                            episode_links.append({
                                "title": title,
                                "url": href,
                            })
                            unique_urls.add(href)
                            
                except NoSuchElementException:
                    # Continue if link element is problematic
                    continue
            
            if not episode_links:
                print("No episode page links found using Selenium. The selectors may need adjustment.")
                
            print(f"Found {len(episode_links)} unique episode page links using Selenium")
            return episode_links
            
        except TimeoutException:
            print(f"Timeout while waiting for page to load or elements to be present on {url}")
            return []
        except Exception as e:
            print(f"Error extracting episode page links with Selenium: {e}")
            return []

    def extract_mp3_link_from_episode_page_selenium(self, url: str) -> Optional[str]:
        """
        Extract the MP3 download link from an individual episode page URL using Selenium.
        
        Args:
            url: URL of the individual episode page
            
        Returns:
            MP3 download URL or None if not found
        """
        print(f"Attempting to extract MP3 link from episode page: {url} using Selenium")
        
        try:
            self.driver.get(url)
            
            # Wait for the page to load and for potential audio elements to be present
            wait = WebDriverWait(self.driver, 10) # Wait for up to 10 seconds
            
            # Look for audio elements or links to MP3 files
            # Using a combination of selectors
            try:
                audio_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "audio source, a[href$='.mp3'], a[href*='audio'], a[href*='mp3']")))
                mp3_url = audio_elem.get_attribute("src") or audio_elem.get_attribute("href")
                
                print(f"Found potential MP3 link on episode page using Selenium: {mp3_url}")
                return mp3_url
            except TimeoutException:
                print("Timeout while waiting for audio elements on episode page.")
                return None
            
        except Exception as e:
            print(f"Error extracting MP3 link from episode page with Selenium: {e}")
            return None

    def extract_mp3_from_fountain_fm(self, url: str) -> Dict:
        """
        Extract MP3 URL and podcast info directly from a fountain.fm episode URL.
        
        Args:
            url: URL of the fountain.fm episode
            
        Returns:
            Dictionary with episode info including title, show_name, and mp3_url
        """
        print(f"Extracting MP3 from fountain.fm URL: {url}")
        
        try:
            # Get the webpage content
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
            
            # Extract the MP3 URL using regex
            mp3_pattern = r'https://[^"]*\.mp3[^"]*'
            mp3_matches = re.findall(mp3_pattern, html_content)
            
            if not mp3_matches:
                print("No MP3 URL found in the webpage")
                return {}
            
            # Get the first MP3 URL and clean it up (replace escaped characters)
            mp3_url = mp3_matches[0].replace('\\u0026', '&')
            
            # Extract the title from the HTML
            title_pattern = r'<title>(.*?)</title>'
            title_match = re.search(title_pattern, html_content)
            full_title = title_match.group(1) if title_match else "Unknown Episode"
            
            # Extract show name from the title (usually before the first • or - character)
            show_name_match = re.search(r'^(.*?)(?:\s[•\-]\s|\s\|\s)', full_title)
            show_name = show_name_match.group(1).strip() if show_name_match else "Unknown Show"
            
            # Clean up the show name for use as a directory name
            safe_show_name = re.sub(r'[^\w\-\.]', '_', show_name)
            
            # Extract episode title (after the show name)
            episode_title_match = re.search(r'(?:\s[•\-]\s|\s\|\s)(.*?)(?:\s[•\-]\s|\s\|\s|$)', full_title)
            episode_title = episode_title_match.group(1).strip() if episode_title_match else full_title
            
            return {
                "title": episode_title,
                "show_name": safe_show_name,
                "full_title": full_title,
                "mp3_url": mp3_url
            }
            
        except Exception as e:
            print(f"Error extracting MP3 from fountain.fm URL: {e}")
            return {}

    def download_episode(self, episode: Dict) -> Optional[Path]:
        """
        Download a podcast episode.
        
        Args:
            episode: Episode dictionary with title, show_name, and mp3_url
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        # Create a safe filename from the episode title
        safe_title = re.sub(r'[^\w\-\.]', '_', episode["title"])
        
        # Create show-specific directory
        show_dir = self.audio_dir / episode["show_name"]
        show_dir.mkdir(parents=True, exist_ok=True)
        
        # Full path to the output file
        filename = show_dir / f"{safe_title}.mp3"
        
        # Skip if already downloaded
        if filename.exists():
            print(f"Episode already downloaded: {filename}")
            return filename
        
        print(f"Downloading: {episode['title']} to {filename}")
        try:
            response = requests.get(episode["mp3_url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(filename, 'wb') as file, tqdm(
                desc=safe_title,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
                    
            return filename
        except Exception as e:
            print(f"Error downloading episode {episode['title']}: {e}")
            return None

    def transcribe_episode(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an episode using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult object containing the transcription
        """
        print(f"Transcribing episode: {audio_path}")
        
        # Create transcript directory for this show if it doesn't exist
        show_name = audio_path.parent.name
        transcript_dir = self.transcripts_dir / show_name
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        base_name = audio_path.stem
        json_path = transcript_dir / f"{base_name}.json"
        txt_path = transcript_dir / f"{base_name}.txt"
        
        # Skip if already transcribed
        if json_path.exists() and txt_path.exists():
            print(f"Episode already transcribed: {txt_path}")
            # Load existing transcription
            with open(json_path, 'r') as f:
                return TranscriptionResult.model_validate(json.load(f))
        
        # Transcribe the audio
        result = transcribe_audio(str(audio_path))
        
        # Save the transcription
        save_transcription(result, str(txt_path))
        print(f"Transcription saved to {txt_path}")
        
        # Save raw JSON for later processing
        with open(json_path, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"Raw transcription data saved to {json_path}")
        
        return result

    def identify_ads(self, transcript: TranscriptionResult, episode_info: Dict) -> List[Advertisement]:
        """
        Identify advertisement segments in the transcript.
        
        Args:
            transcript: TranscriptionResult object containing the transcription
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
        base_name = re.sub(r'[^\w\-\.]', '_', episode_info["title"])
        ads_path = ads_dir / f"{base_name}_ads.json"
        
        # Skip if already processed
        if ads_path.exists():
            print(f"Ad information already exists: {ads_path}")
            # Load existing ad information
            with open(ads_path, 'r') as f:
                ads_data = json.load(f)
                return [Advertisement(**ad) for ad in ads_data]
        
        # Find sponsored segments
        ads = find_sponsored_segments(transcript.model_dump())
        
        # Save ad information
        with open(ads_path, 'w') as f:
            json.dump([ad.model_dump() for ad in ads], f, indent=2)
        print(f"Ad information saved to {ads_path}")
        
        return ads

    def generate_summary(self, transcript: TranscriptionResult, episode_info: Dict, ads: List[Advertisement]) -> Dict:
        """
        Generate an AI summary with key insights and actionable takeaways.
        
        Args:
            transcript: TranscriptionResult object containing the transcription
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
        base_name = re.sub(r'[^\w\-\.]', '_', episode_info["title"])
        summary_path = summary_dir / f"{base_name}_summary.json"
        
        # Skip if already processed
        if summary_path.exists():
            print(f"Summary already exists: {summary_path}")
            # Load existing summary
            with open(summary_path, 'r') as f:
                return json.load(f)
        
        # For now, just create a placeholder summary
        # In a future version, this would use an LLM to generate a real summary
        summary = {
            "title": episode_info["title"],
            "show": episode_info["show_name"],
            "ad_count": len(ads),
            "duration": transcript.segments[-1].end if transcript.segments else 0,
            "key_points": ["This is a placeholder for key points"],
            "actionable_takeaways": ["This is a placeholder for actionable takeaways"]
        }
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
        
        return summary

    def extract_show_name_from_url(self, url: str) -> str:
        """
        Extract the show name from a fountain.fm show URL.
        
        Args:
            url: URL of the fountain.fm show
            
        Returns:
            Show name or a default name if extraction fails
        """
        try:
            # Get the webpage content
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
            
            # Extract the title from the HTML
            title_pattern = r'<title>(.*?)</title>'
            title_match = re.search(title_pattern, html_content)
            full_title = title_match.group(1) if title_match else "Unknown Show"
            
            # Extract show name from the title (usually before the first • or - character)
            show_name_match = re.search(r'^(.*?)(?:\s[•\-]\s|\s\|\s|$)', full_title)
            show_name = show_name_match.group(1).strip() if show_name_match else "Unknown Show"
            
            # Clean up the show name for use as a directory name
            safe_show_name = re.sub(r'[^\w\-\.]', '_', show_name)
            
            return safe_show_name
            
        except Exception as e:
            print(f"Error extracting show name from URL: {e}")
            return "Unknown_Show"

    def run(self) -> List[Dict]:
        """
        Run the podcast processor to download episodes.
        
        Returns:
            List of processing results for the downloaded episodes
        """
        print(f"Starting Gobble podcast processor - Fountain.fm Download Mode")
        print(f"URL: {self.url}")
        
        results = []
        
        # Check if this is a fountain.fm URL
        if "fountain.fm/episode/" in self.url:
            # Direct fountain.fm episode URL
            print("Detected fountain.fm episode URL")
            
            # Extract MP3 and episode info directly
            episode_info = self.extract_mp3_from_fountain_fm(self.url)
            
            if episode_info and "mp3_url" in episode_info:
                print(f"Found MP3 URL: {episode_info['mp3_url']}")
                print(f"Show name: {episode_info['show_name']}")
                print(f"Episode title: {episode_info['title']}")
                
                # Download the episode
                result = self.download_episode(episode_info)
                if result:
                    results.append({
                        "status": "success",
                        "show": episode_info["show_name"],
                        "episode": episode_info["title"],
                        "audio_path": str(result),
                        "message": "Episode downloaded successfully"
                    })
                else:
                    results.append({
                        "status": "error",
                        "show": episode_info["show_name"],
                        "episode": episode_info["title"],
                        "message": "Download failed"
                    })
            else:
                print("Could not extract MP3 URL from the fountain.fm episode page")
                results.append({
                    "status": "error",
                    "episode": "Unknown",
                    "message": "Could not extract MP3 URL from the fountain.fm episode page"
                })
        
        elif "fountain.fm/show/" in self.url:
            # Fountain.fm show URL
            print("Detected fountain.fm show URL")
            
            # Extract show name from the show page
            show_name = self.extract_show_name_from_url(self.url)
            print(f"Show name: {show_name}")
            
            # Extract episode page links using Selenium
            print(f"Extracting episode links from show page...")
            episode_page_links = self.extract_episode_page_links_selenium(self.url)
            
            # Print the number of extracted links
            print(f"Found {len(episode_page_links)} unique episode page links")
            
            if episode_page_links:
                print(f"Will download all {len(episode_page_links)} episodes")
                
                # Process all episodes
                for i, episode_link_info in enumerate(episode_page_links):
                    print(f"\nProcessing episode {i+1}/{len(episode_page_links)}: {episode_link_info['title']}")
                    
                    # Extract MP3 link from the episode page
                    if "fountain.fm/episode/" in episode_link_info["url"]:
                        # Use direct extraction for fountain.fm episode URLs
                        episode_info = self.extract_mp3_from_fountain_fm(episode_link_info["url"])
                        if not episode_info or "mp3_url" not in episode_info:
                            print(f"Could not extract MP3 URL for episode: {episode_link_info['title']}")
                            results.append({
                                "status": "error",
                                "show": show_name,
                                "episode": episode_link_info["title"],
                                "message": "Could not extract MP3 URL from episode page"
                            })
                            continue
                    else:
                        # Use Selenium for non-fountain.fm episode URLs
                        mp3_url = self.extract_mp3_link_from_episode_page_selenium(episode_link_info["url"])
                        if not mp3_url:
                            print(f"Could not find MP3 link for episode: {episode_link_info['title']}")
                            results.append({
                                "status": "error",
                                "show": show_name,
                                "episode": episode_link_info["title"],
                                "message": "Could not find MP3 link on episode page"
                            })
                            continue
                        
                        # Create episode dictionary with MP3 URL
                        episode_info = {
                            "title": episode_link_info["title"],
                            "mp3_url": mp3_url,
                            "show_name": show_name,
                            "description": "", # Description would need to be scraped from episode page
                            "date": "", # Date would need to be scraped from episode page
                        }
                    
                    # Download the episode
                    result = self.download_episode(episode_info)
                    if result:
                        results.append({
                            "status": "success",
                            "show": show_name,
                            "episode": episode_info["title"],
                            "audio_path": str(result),
                            "message": "Episode downloaded successfully"
                        })
                    else:
                        results.append({
                            "status": "error",
                            "show": show_name,
                            "episode": episode_info["title"],
                            "message": "Download failed"
                        })
            else:
                print("No episode page links were extracted, cannot download any episodes.")
        
        else:
            # Use the original Selenium-based approach for non-fountain.fm URLs
            print("Not a fountain.fm URL, using Selenium-based extraction")
            
            # Extract episode page links using Selenium
            episode_page_links = self.extract_episode_page_links_selenium(self.url)
            
            # Print the number of extracted links
            print(f"Found {len(episode_page_links)} unique episode page links using Selenium")
                
            if episode_page_links:
                # Process only the first episode for now
                first_episode_link_info = episode_page_links[0]
                
                # Extract MP3 link from the first episode page using Selenium
                mp3_url = self.extract_mp3_link_from_episode_page_selenium(first_episode_link_info["url"])
                
                if mp3_url:
                    # Create episode dictionary with MP3 URL
                    episode_info = {
                        "title": first_episode_link_info["title"],
                        "mp3_url": mp3_url,
                        "show_name": "Unknown_Show",  # Default show name
                        "description": "", # Description would need to be scraped from episode page
                        "date": "", # Date would need to be scraped from episode page
                    }
                    # Download the episode
                    result = self.download_episode(episode_info)
                    if result:
                        results.append({
                            "status": "success",
                            "episode": episode_info["title"],
                            "audio_path": str(result),
                            "message": "Episode downloaded successfully"
                        })
                    else:
                         results.append({
                            "status": "error",
                            "episode": episode_info["title"],
                            "message": "Download failed"
                        })
                else:
                    print(f"Could not find MP3 link for episode: {first_episode_link_info['title']}")
                    results.append({
                        "status": "error",
                        "episode": first_episode_link_info["title"],
                        "message": "Could not find MP3 link on episode page"
                    })
            else:
                print("No episode page links were extracted, cannot download any episodes.")
            
        # Print summary
        print("\nProcessing complete!")
        successful_downloads = sum(1 for r in results if r["status"] == "success")
        failed_downloads = sum(1 for r in results if r["status"] == "error")
        
        if results:
            print(f"Attempted to download {len(results)} episodes.")
            print(f"Successfully downloaded: {successful_downloads}")
            print(f"Failed downloads: {failed_downloads}")
            
            if successful_downloads > 0:
                print("\nDownloaded episodes:")
                for r in results:
                    if r["status"] == "success":
                        print(f"- {r['episode']} -> {r['audio_path']}")
                
                # Process downloaded episodes
                print("\nProcessing downloaded episodes...")
                for r in results:
                    if r["status"] == "success":
                        try:
                            # Extract episode info
                            audio_path = Path(r["audio_path"])
                            episode_info = {
                                "title": r["episode"],
                                "show_name": r["show"],
                                "audio_path": r["audio_path"]
                            }
                            
                            # Transcribe the episode
                            transcript = self.transcribe_episode(audio_path)
                            
                            # Identify ads
                            ads = self.identify_ads(transcript, episode_info)
                            print(f"Found {len(ads)} ads in {r['episode']}")
                            
                            # Generate summary
                            summary = self.generate_summary(transcript, episode_info, ads)
                            
                            # Update result with processing information
                            r["transcript_path"] = str(self.transcripts_dir / r["show"] / f"{audio_path.stem}.txt")
                            r["ads_count"] = len(ads)
                            r["summary_path"] = str(self.summaries_dir / r["show"] / f"{audio_path.stem}_summary.json")
                            
                        except Exception as e:
                            print(f"Error processing episode {r['episode']}: {e}")
                            r["processing_error"] = str(e)
        else:
            print("No episodes were processed.")
            
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gobble - A podcast processing tool")
    parser.add_argument("url", help="URL of the podcast homepage")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("-n", "--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of episodes to process (influences scrolling)")
    parser.add_argument("-m", "--model", default=DEFAULT_WHISPER_MODEL, help="Whisper model to use (ignored in this mode)")
    args = parser.parse_args()
    
    processor = PodcastProcessor(
        url=args.url,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        whisper_model=args.model, # This is not used in this mode
    )
    
    processor.run()


if __name__ == "__main__":
    main()
