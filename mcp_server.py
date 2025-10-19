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

from tools.youtube.register_tools import register_youtube_tools

# Load environment variables from .env file
load_dotenv()

from fastmcp import FastMCP
import mcp.types as types
import asyncio
from pydantic import Field, BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gobble-server")

# Increase timeout to 5 minutes (300 seconds)
mcp = FastMCP("Podcast MCP Server", host="0.0.0.0", port=8000, timeout=300)
 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_youtube_tools(mcp)
    
    asyncio.run(mcp.run_sse_async())
