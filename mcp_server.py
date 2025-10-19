"""
Gobble: MCP server for Chat-with-podcast integration.
"""
import os
import json
import logging
import tempfile
import argparse
from typing import Optional, Sequence, Dict, Any, Tuple, List
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from pathlib import Path
import re

from tools.youtube.register_tools import register_youtube_tools
from tools.knowledge_retrieval.transcripts import register_knowledgebase_tools
from tools.knowledge_retrieval.vector_store import initialize_knowledge_base


# Load environment variables from .env file
load_dotenv()

from fastmcp import FastMCP
import mcp.types as types
import asyncio
from pydantic import Field, BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gobble-server")
 

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gobble MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="sse",
        help="Transport type: 'sse' for Server-Sent Events or 'stdio' for standard input/output (default: sse)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for SSE transport (default: 8000, only used with --transport sse)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for SSE transport (default: 0.0.0.0, only used with --transport sse)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize FastMCP with appropriate settings based on transport type
    if args.transport == "sse":
        logger.info(f"Starting MCP server with SSE transport on {args.host}:{args.port}")
        # Increase timeout to 5 minutes (300 seconds)
        mcp = FastMCP("Podcast MCP Server", host=args.host, port=args.port, timeout=300)
    else:
        logger.info("Starting MCP server with stdio transport")
        # For stdio, host and port are not used
        mcp = FastMCP("Podcast MCP Server", timeout=300)

    initialize_knowledge_base()

    register_youtube_tools(mcp)
    register_knowledgebase_tools(mcp)
    
    # Run with the appropriate transport
    if args.transport == "sse":
        asyncio.run(mcp.run_sse_async())
    else:
        mcp.run()
