import os
import json
import logging
import tempfile
from typing import Optional, Sequence, Dict, Any, Tuple, List
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from pathlib import Path
import re
import chromadb
from chromadb.utils import embedding_functions

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="knowledge/chroma")

# Create embedding function (using default sentence transformers)
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Get or create collection with proper embedding function
transcript_collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=default_ef
)

# Path to store processed files metadata
PROCESSED_FILES_PATH = Path("knowledge/chroma/processed_files.json")

def load_processed_files():
    """Load record of previously processed files"""
    if PROCESSED_FILES_PATH.exists():
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading processed files record, starting fresh")
            return {}
    return {}

def save_processed_files(processed):
    """Save record of processed files"""
    # Create directory if it doesn't exist
    PROCESSED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed, f, indent=2)

def get_file_metadata(path: Path) -> dict:
    """Get metadata for a file to track changes"""
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "num_chunks": 0  # Will be updated after processing
    }

def initialize_knowledge_base(force_reprocess=False):
    """Initialize Chroma knowledge base with new or modified transcripts
    
    Args:
        force_reprocess: If True, reprocess all transcripts regardless of whether they've changed
    """
    logger.info("Initializing Chroma knowledge base...")
    
    # Load record of previously processed files
    processed_files = load_processed_files()
    
    transcripts_dir = Path("knowledge")
    if not transcripts_dir.exists():
        logger.warning(f"Transcripts directory not found: {transcripts_dir}")
        return
    
    transcript_files = list(transcripts_dir.rglob("*.txt"))
    logger.info(f"Found {len(transcript_files)} transcript files")
    
    # Track files to process
    files_to_process = []
    for transcript_path in transcript_files:
        file_key = str(transcript_path)
        current_metadata = get_file_metadata(transcript_path)
        
        # Skip if file hasn't changed and we're not forcing reprocessing
        if not force_reprocess and file_key in processed_files:
            old_metadata = processed_files[file_key]
            if (old_metadata["size"] == current_metadata["size"] and 
                old_metadata["mtime"] == current_metadata["mtime"]):
                logger.info(f"Skipping unchanged transcript: {transcript_path}")
                continue
        
        files_to_process.append((transcript_path, current_metadata))
    
    logger.info(f"Processing {len(files_to_process)} new or modified transcripts")
    
    # Process files that need updating
    total_chunks = 0
    for transcript_path, current_metadata in files_to_process:
        try:
            # Extract metadata from path
            source = transcript_path.parent.parent.name  # youtube or fountain
            channel = transcript_path.parent.name
            title = transcript_path.stem
            file_key = str(transcript_path)
            
            logger.info(f"Processing transcript: {source}/{channel}/{title}")
            
            # Read transcript content
            with open(transcript_path, 'r') as f:
                content = f.read()
            
            # Clean transcript content (remove timestamps)
            cleaned_lines = [
                line[25:].strip() 
                for line in content.split('\n')
                if line.strip()
            ]
            cleaned_text = ' '.join(cleaned_lines)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len
            )
            chunks = text_splitter.split_text(cleaned_text)
            
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Remove existing chunks for this transcript if any
            try:
                # Get existing IDs for this transcript
                existing_ids = transcript_collection.get(
                    where={"path": str(transcript_path)}
                )["ids"]
                
                if existing_ids:
                    logger.info(f"Removing {len(existing_ids)} existing chunks for {title}")
                    transcript_collection.delete(ids=existing_ids)
            except Exception as e:
                logger.warning(f"Error checking for existing chunks: {str(e)}")
            
            # Add chunks to collection with metadata
            for i, chunk in enumerate(chunks):
                transcript_collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "source": source,
                        "channel": channel,
                        "title": title,
                        "chunk": i,
                        "path": str(transcript_path)
                    }],
                    ids=[f"{source}-{channel}-{title}-chunk-{i}"]
                )
            
            # Update metadata with number of chunks
            current_metadata["num_chunks"] = len(chunks)
            processed_files[file_key] = current_metadata
            total_chunks += len(chunks)
            
            logger.info(f"Successfully added {len(chunks)} chunks to collection for {title}")
            
            # Save processed files record periodically
            save_processed_files(processed_files)
            
        except Exception as e:
            logger.error(f"Error processing transcript {transcript_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Final save of processed files record
    save_processed_files(processed_files)
    
    logger.info(f"Knowledge base initialization complete!")
    logger.info(f"Processed {len(files_to_process)} transcripts")
    logger.info(f"Total chunks added to collection: {total_chunks}")


def embed_transcript(transcript_path: str) -> Dict[str, Any]:
    """Embed a single transcript file into the vector database.
    
    This utility function processes a single transcript file and adds it to the
    ChromaDB collection. Useful for adding new transcripts immediately after
    they are created, without needing to run a full knowledge base initialization.
    
    Args:
        transcript_path: Path to the transcript file (relative or absolute)
        
    Returns:
        Dictionary with status information:
        - success: bool indicating if embedding was successful
        - chunks_added: number of chunks added to the database
        - error: error message if unsuccessful
    """
    try:
        transcript_path = Path(transcript_path)
        
        if not transcript_path.exists():
            return {
                "success": False,
                "error": f"Transcript file not found: {transcript_path}"
            }
        
        # Extract metadata from path
        # Expected structure: knowledge/youtube/{channel}/{title}-transcript.txt
        source = transcript_path.parent.parent.name  # youtube or fountain
        channel = transcript_path.parent.name
        title = transcript_path.stem
        
        logger.info(f"Embedding transcript: {source}/{channel}/{title}")
        
        # Read transcript content
        with open(transcript_path, 'r') as f:
            content = f.read()
        
        # Clean transcript content (remove timestamps)
        cleaned_lines = [
            line[25:].strip() 
            for line in content.split('\n')
            if line.strip()
        ]
        cleaned_text = ' '.join(cleaned_lines)
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len
        )
        chunks = text_splitter.split_text(cleaned_text)
        
        logger.info(f"Split transcript into {len(chunks)} chunks")
        
        # Remove existing chunks for this transcript if any
        try:
            existing_ids = transcript_collection.get(
                where={"path": str(transcript_path)}
            )["ids"]
            
            if existing_ids:
                logger.info(f"Removing {len(existing_ids)} existing chunks for {title}")
                transcript_collection.delete(ids=existing_ids)
        except Exception as e:
            logger.warning(f"Error checking for existing chunks: {str(e)}")
        
        # Add chunks to collection with metadata
        for i, chunk in enumerate(chunks):
            transcript_collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": source,
                    "channel": channel,
                    "title": title,
                    "chunk": i,
                    "path": str(transcript_path)
                }],
                ids=[f"{source}-{channel}-{title}-chunk-{i}"]
            )
        
        # Update processed files record
        processed_files = load_processed_files()
        file_metadata = get_file_metadata(transcript_path)
        file_metadata["num_chunks"] = len(chunks)
        processed_files[str(transcript_path)] = file_metadata
        save_processed_files(processed_files)
        
        logger.info(f"Successfully embedded {len(chunks)} chunks for {title}")
        
        return {
            "success": True,
            "chunks_added": len(chunks),
            "path": str(transcript_path),
            "source": source,
            "channel": channel,
            "title": title
        }
        
    except Exception as e:
        logger.error(f"Error embedding transcript {transcript_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
            

async def search_topics(
    query: str, 
    max_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Search transcript vector store for topics using semantic/vector search.
    
    This performs semantic search across transcript embeddings to find content related
    to the query topic, even if the exact words aren't present. Useful for finding
    discussions about concepts, themes, or topics across multiple episodes.
    
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
        
    Returns:
        Dictionary containing the search results with metadata
    """
    logger.info(f"Searching transcripts with query: {query}, max_results: {max_results}")
    logger.info(f"Filters - where: {where}, where_document: {where_document}")
    
    include = ["documents", "metadatas", "distances"]
    
    try:
        # Execute the query with all parameters
        results = transcript_collection.query(
            query_texts=[query],
            n_results=max_results,
            where=where,
            where_document=where_document,
            include=include
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
