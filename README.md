# Gobble - Podcast Processing Tool

Gobble is a powerful podcast processing tool that automates the workflow of downloading, transcribing, analyzing, and summarizing podcast episodes. It extracts valuable content from podcasts while identifying and marking advertisement segments.

## Features

- **Podcast Episode Extraction**: Automatically extracts episode links from podcast homepages using Selenium
- **Audio Download**: Downloads podcast episodes in MP3 format
- **Transcription**: Transcribes audio using OpenAI's Whisper model
- **Advertisement Detection**: Identifies sponsored segments and advertisements using Gemini AI
- **Content Summarization**: Generates concise summaries with key insights and actionable takeaways
- **Batch Processing**: Process multiple episodes or entire shows at once

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -e .
   ```
3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### Processing a Podcast from URL

```bash
python main.py https://fountain.fm/show/your-podcast-url
```

### Processing Existing Audio Files

```bash
python process_existing.py --show "Show_Name" --limit 5
```

### Processing Existing Transcripts

```bash
python process_transcripts.py --show "Show_Name" --limit 5
```

## Command Line Options

- `--output-dir`: Directory to save output files (default: "output")
- `--show`: Name of the show to process (if not specified, process all shows)
- `--limit`: Maximum number of episodes to process per show
- `--model`: Whisper model to use for transcription (default: "tiny")

## Project Structure

- `main.py`: Main entry point for downloading and processing podcasts
- `transcribe.py`: Handles audio transcription using Whisper
- `find_ads.py`: Identifies advertisement segments using Gemini AI
- `process_transcripts.py`: Processes existing transcripts to generate summaries
- `process_existing.py`: Processes existing audio files in the output directory

## Output Directory Structure

```
output/
├── audio/                  # Downloaded podcast episodes
│   └── Show_Name/
│       └── Episode_Title.mp3
├── transcripts/            # Transcriptions of episodes
│   └── Show_Name/
│       ├── Episode_Title.json
│       └── Episode_Title.txt
├── processed/              # Processed data (e.g., ad information)
│   └── Show_Name/
│       └── Episode_Title_ads.json
└── summaries/              # Episode summaries
    └── Show_Name/
        └── Episode_Title_summary.json
```
