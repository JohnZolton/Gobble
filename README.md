# Gobble - Podcast Processing and Search Platform

Gobble is a powerful podcast processing platform that automates the workflow of downloading, transcribing, analyzing, and searching podcast content. It extracts valuable content from podcasts while providing advanced search capabilities through a Model Context Protocol (MCP) server.

## Core Features

- **MCP Server Integration**: Access podcast data and functionality through Model Context Protocol
- **Advanced Search**: Search transcripts with metadata filtering and content filtering
- **Podcast Episode Processing**: Download and transcribe podcast episodes from Fountain.fm and YouTube
- **Transcription**: Transcribes audio using OpenAI's Whisper model
- **Advertisement Detection**: Identifies sponsored segments and advertisements
- **Vector Database**: Store and search podcast transcripts using semantic search
- **Read Transcripts**: entire transcripts available as MCP resources for in-depth analysis

## Installation

1. Clone this repository
2. Install dependencies using uv:
   ```
   uv add -e .
   ```
3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## MCP Server

The Model Context Protocol (MCP) server is the main interface for interacting with Gobble's functionality. It exposes tools and resources that allow AI models to search for podcasts, transcribe episodes, and access transcript data.

### Starting the MCP Server

```bash
uv run python mcp_server.py
```

### MCP Tools

- **search_transcripts**: Search transcript vector store with advanced filtering options
  - Supports metadata filtering with operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
  - Supports document content filtering with $contains and $not_contains
  - Example: Search for "biochar" in transcripts from "The_Survival_Podcast" channel
- **transcribe_episode_fountain**: Downloads and transcribes a specific podcast episode from a given fountain.fm URL

  - Returns immediately with status information while processing continues in the background
  - Supports the "tiny.en" model for faster transcription

- **search_youtube**: Search YouTube for specific channels or episodes

  - Returns detailed information about videos including title, channel, URL, and description

- **transcribe_youtube**: Download and transcribe YouTube videos

  - Processes videos in the background and stores transcripts in the output directory

- **remove_ads**: Remove advertisements from an audio file using its transcript
  - Uses the transcript to identify ad segments and creates a clean version of the audio

### MCP Resources

- **gobble://transcripts/{filename\*}**: Retrieve the content of a specific transcript file
  - Example: `gobble://transcripts/The_Survival_Podcast/Episode_Title.txt`
- **gobble://transcripts**: List all available transcript files organized by show

## Usage Examples

### Searching Transcripts with Metadata Filtering

```python
# Search for "biochar" in The Survival Podcast channel
result = await search_transcripts(
    query="biochar",
    where={"channel": "The_Survival_Podcast"},
    max_results=10
)

# Search with complex metadata filters
result = await search_transcripts(
    query="bitcoin",
    where={
        "$or": [
            {"channel": "The_Survival_Podcast"},
            {
                "$and": [
                    {"source": "youtube"},
                    {"title": {"$contains": "crypto"}}
                ]
            }
        ]
    }
)

# Search with document content filtering
result = await search_transcripts(
    query="investing",
    where_document={"$contains": "retirement"},
    max_results=5
)
```

### Transcribing Podcast Episodes

```python
# Transcribe a Fountain.fm episode
result = await transcribe_episode_fountain(
    episode_url="https://fountain.fm/episode/your-episode-id"
)

# Transcribe YouTube videos
result = await transcribe_youtube(
    urls=["https://www.youtube.com/watch?v=video-id"]
)
```

## Command Line Tools

Gobble also provides command-line tools for processing podcasts:

### Processing a Podcast from URL

```bash
uv run python main.py https://fountain.fm/show/your-podcast-url
```

### Processing Existing Audio Files

```bash
uv run python process_existing.py --show "Show_Name" --limit 5
```

### Processing Existing Transcripts

```bash
uv run python process_transcripts.py --show "Show_Name" --limit 5
```

## Command Line Options

- `--output-dir`: Directory to save output files (default: "output")
- `--show`: Name of the show to process (if not specified, process all shows)
- `--limit`: Maximum number of episodes to process per show
- `--model`: Whisper model to use for transcription (default: "tiny")

## Project Structure

- `mcp_server.py`: Model Context Protocol server for AI model integration
- `main.py`: Main entry point for downloading and processing podcasts
- `transcribe.py`: Handles audio transcription using Whisper
- `find_ads.py`: Identifies advertisement segments
- `process_transcripts.py`: Processes existing transcripts
- `process_existing.py`: Processes existing audio files in the output directory

## Output Directory Structure

```
output/
├── audio/                  # Downloaded podcast episodes
│   ├── fountain/           # Fountain.fm episodes
│   │   └── Show_Name/
│   │       └── Episode_Title.mp3
│   └── youtube/            # YouTube videos
│       └── Channel_Name/
│           └── Video_Title.mp3
├── transcripts/            # Transcriptions of episodes
│   ├── fountain/           # Fountain.fm transcripts
│   │   └── Show_Name/
│   │       ├── Episode_Title.json
│   │       └── Episode_Title.txt
│   └── youtube/            # YouTube transcripts
│       └── Channel_Name/
│           ├── Video_Title.json
│           └── Video_Title.txt
├── processed/              # Processed audio (e.g., ad-free versions)
│   └── Episode_Title_no_ads.mp3
└── temp/                   # Temporary files during processing
```

## Vector Database

Gobble uses Chroma as a vector database to store and search podcast transcripts. The database is initialized automatically when the MCP server starts, processing any new or modified transcripts in the output directory.

### Database Structure

- Each transcript is split into chunks of approximately 800 characters
- Each chunk is stored with metadata including:
  - source: The source platform (youtube or fountain)
  - channel: The channel or show name
  - title: The episode or video title
  - chunk: The chunk number within the transcript
  - path: The path to the transcript file

## Future Development

- [ ] Add support for more podcast platforms beyond Fountain.fm and YouTube
- [ ] Add RSS support/monitoring to automatically process new episodes
