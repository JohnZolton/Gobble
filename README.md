# Gobble

an mcp server for managing a podcast-based knowledge base.

## Features:

- download and transcribe youtube videos with yt-dlp and parakeet (fast, transcribes 2 hours in ~45 seconds)
- search episodes, load into your chatbot
- vector db + search over transcriptions

## requirements

a GPU for local transcription

## install

```bash
uv sync
```

## To run:

```bash
# Run with SSE on default port 8000
uv run mcp_server.py

# Run with SSE on custom port
uv run mcp_server.py --transport sse --port 9000

# Run with stdio transport
uv run mcp_server.py --transport stdio
```

To run over stdio in something like goose or cline:

`chmod +x path/to/run_gobble.sh`

and then just put `path/to/gobble.sh` in the command box of your app

coming soon™:

- support podcasts
- autodownload subscriptions (yt/podcast rss feed)

### utils

download a video

```bash
python -m tools.youtube.download_youtube {url_to_video}
```
