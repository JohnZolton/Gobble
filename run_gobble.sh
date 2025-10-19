#!/usr/bin/env bash
cd /home/john/Documents/Programming/gobble
exec uv run mcp_server.py --transport stdio
