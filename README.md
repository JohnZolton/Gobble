# Gobble - Podcast Processing Tool

Gobble is a powerful podcast processing tool that automates the workflow of downloading, transcribing, analyzing, and summarizing podcast episodes. It extracts valuable content from podcasts while identifying and marking advertisement segments.

## Features

- **Podcast Episode Extraction**: Automatically extracts episode links from podcast homepages using Selenium
- **Audio Download**: Downloads podcast episodes in MP3 format
- **Transcription**: Transcribes audio using OpenAI's Whisper model
- **Advertisement Detection**: Identifies sponsored segments and advertisements using Gemini AI
- **Content Summarization**: Generates concise summaries with key insights and actionable takeaways
- **Batch Processing**: Process multiple episodes or entire shows at once
- **MCP Server Integration**: Access podcast data and functionality through Model Context Protocol

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

### Starting the MCP Server

```bash
python mcp_server.py
```

## Command Line Options

- `--output-dir`: Directory to save output files (default: "output")
- `--show`: Name of the show to process (if not specified, process all shows)
- `--limit`: Maximum number of episodes to process per show
- `--model`: Whisper model to use for transcription (default: "tiny")

## MCP Server

The Model Context Protocol (MCP) server provides an interface for AI models to interact with Gobble's functionality. It exposes tools and resources that allow AI models to search for podcasts, transcribe episodes, and access transcript data.

### MCP Tools

- **search_fountain**: Search for a specific podcast on fountain.fm
- **transcribe_episode_fountain**: Downloads and transcribes a specific podcast episode from a given fountain.fm URL
  - Returns immediately with status information while processing continues in the background
  - Supports the "tiny.en" model for faster transcription
- **create_vector_db_from_series_fountain**: Downloads, transcribes, and stores entire series of a fountain.fm show URL into a vector db (TODO)
- **search_youtube**: Search YouTube for a specific channel/episode (TODO)
- **transcribe_youtube**: Transcribe a YouTube video (TODO)

### MCP Resources

- **gobble://transcripts/{filename\*}**: Retrieve the content of a specific transcript file
  - Example: `gobble://transcripts/The_Survival_Podcast/Episode_Title.txt`
- **gobble://transcripts**: List all available transcript files organized by show
- **gobble://search/{query}**: Retrieve search results for a specific query (TODO)

## TODO List

- [ ] Implement `create_vector_db_from_series_fountain` tool to process entire podcast series
- [ ] Implement `search_youtube` tool to search for YouTube videos
- [ ] Implement `transcribe_youtube` tool to transcribe YouTube videos
- [ ] Complete the `gobble://search/{query}` resource for searching podcast content
- [ ] Add vector database integration for semantic search of podcast content
- [ ] Add support for more podcast platforms beyond Fountain.fm


## Project Structure

- `main.py`: Main entry point for downloading and processing podcasts
- `transcribe.py`: Handles audio transcription using Whisper
- `find_ads.py`: Identifies advertisement segments using Gemini AI
- `process_transcripts.py`: Processes existing transcripts to generate summaries
- `process_existing.py`: Processes existing audio files in the output directory
- `mcp_server.py`: Model Context Protocol server for AI model integration

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
