
# MCP resources for accessing transcripts
@mcp.tool()
async def get_transcript(filename: str) -> str:
    """Retrieve the content of a transcript file.
    
    Args:
        filename: Path to the transcript file within the output/transcripts directory
        
    Returns:
        Content of the transcript file as text
    """
    logger.info(f"Resource request for transcript: {filename}")
    
    # Construct the full path to the transcript file
    transcript_path = Path("output/transcripts") / filename
    
    # Check if the file exists
    if not transcript_path.exists():
        return f"Transcript file not found: {filename}"
    
    # Read and return the content of the file
    try:
        with open(transcript_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading transcript file: {str(e)}")
        return f"Error reading transcript file: {str(e)}"

@mcp.tool()
async def list_shows() -> dict:
    """List all available show names with transcripts.
    
    Returns:
        Dictionary containing a list of available show names
    """
    logger.info("Resource request for show listing")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("output/transcripts")
    
    # Check if the directory exists
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # List to store the show names
    shows = []
    
    # Walk through the directory and collect show names (directories)
    for show_dir in transcripts_dir.iterdir():
        if show_dir.is_dir():
            shows.append(show_dir.name)
    
    return {"shows": sorted(shows)}

@mcp.tool()
async def list_episodes(show_name: str) -> dict:
    """List all available episodes for a specific show.
    
    Args:
        show_name: Name of the show to list episodes for
        
    Returns:
        Dictionary containing a list of available episodes for the specified show
    """
    logger.info(f"Resource request for episodes listing for show: {show_name}")
    
    # Get the path to the transcripts directory
    transcripts_dir = Path("output/transcripts")
    
    # Check if the directory exists
    if not transcripts_dir.exists():
        return {"error": "Transcripts directory not found"}
    
    # Check if the specific show directory exists
    show_dir = transcripts_dir / show_name
    if not show_dir.exists():
        return {"error": f"Show directory not found: {show_name}"}
    
    # List to store the episode files
    episodes = []
    
    # Recursively find all .txt files in this show directory
    for transcript_file in show_dir.rglob("*.txt"):
        # Get the relative path from the show directory
        rel_path = transcript_file.relative_to(show_dir)
        episodes.append(str(rel_path))
    
    return {
        "show_name": show_name,
        "episodes": sorted(episodes)
    }

# Search resource
@mcp.tool()
async def search_transcripts(
    query: str, 
    max_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search transcript vector store for a specific query with advanced filtering options.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        where: Optional dictionary for metadata filtering with operators:
               - $eq, $ne: Equal/not equal (string, int, float)
               - $gt, $gte, $lt, $lte: Greater/less than (int, float)
               - $in, $nin: In/not in list (string, int, float)
               - $and, $or: Logical operators for combining filters
               Example: {"source": "youtube", "channel": {"$in": ["Channel1", "Channel2"]}}
        where_document: Optional dictionary for document content filtering:
               - $contains: Text the document must contain
               - $not_contains: Text the document must not contain
               Example: {"$contains": "bitcoin", "$not_contains": "ethereum"}
        ids: Optional list of specific document IDs to search within
        
    Returns:
        Dictionary containing the search results with metadata
    """
    logger.info(f"Searching transcripts with query: {query}, max_results: {max_results}")
    logger.info(f"Filters - where: {where}, where_document: {where_document}, ids: {ids}")
    
    include = ["documents", "metadatas", "distances"]
    
    try:
        # Execute the query with all parameters
        results = transcript_collection.query(
            query_texts=[query],
            n_results=max_results,
            where=where,
            where_document=where_document,
            include=include,
            ids=ids
        )
        
        # Format the results in a more user-friendly structure
        formatted_results = {
            "query": query,
            "total_results": len(results["ids"][0]) if results["ids"] else 0,
            "results": []
        }
        
        # Process each result
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                result_item = {
                    "id": results["ids"][0][i]
                }
                
                # Add included fields
                if "documents" in include and "documents" in results:
                    result_item["document"] = results["documents"][0][i]
                
                if "metadatas" in include and "metadatas" in results:
                    result_item["metadata"] = results["metadatas"][0][i]
                
                if "distances" in include and "distances" in results:
                    result_item["distance"] = results["distances"][0][i]
                
                formatted_results["results"].append(result_item)
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching transcripts: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "total_results": 0,
            "results": []
        }
    