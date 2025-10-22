"""MCP tools for ebook analysis and management."""
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import Field

from .book_utils import (
    extract_book_metadata,
    extract_book_content,
    scan_books_directory,
    sanitize_filename,
    get_book_directory
)
# Ensure we have the scan_books_directory function available
from ..knowledge_retrieval.vector_store import embed_transcript
import concurrent.futures
import os

def _cache_book_metadata(book_path: Path) -> Dict[str, Any]:
    """Cache book metadata to JSON file."""
    book_id = sanitize_filename(book_path.stem)
    cache_file = get_book_directory(book_id) / "metadata.json"
    
    metadata = extract_book_metadata(book_path)
    chapters = extract_book_content(book_path)
    
    # Filter out producer and other unnecessary fields from metadata
    filtered_metadata = {
        key: value for key, value in metadata.items()
        if key not in ["producer", "creator"] and value is not None and value != ""
    }
    
    # Cache complete data for book information tool
    book_data = {
        "book_id": book_id,
        "metadata": filtered_metadata,
        "chapters": [{"title": ch["title"], "filename": ch["filename"]} for ch in chapters],
        "total_chapters": len(chapters)
    }
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(book_data, indent=2))
    
    return book_data

def _get_list_book_entry(book_path: Path) -> Dict[str, Any]:
    """Get book entry for list_books - excludes chapters and clean metadata."""
    book_id = sanitize_filename(book_path.stem)
    cache_file = get_book_directory(book_id) / "metadata.json"
    
    if cache_file.exists():
        cached_data = json.loads(cache_file.read_text())
    else:
        metadata = extract_book_metadata(book_path)
        chapters = extract_book_content(book_path)
        cached_data = {
            "book_id": book_id,
            "metadata": metadata,
            "total_chapters": len(chapters)
        }

    # Clean metadata for list_books
    clean_metadata = {
        key: value for key, value in cached_data["metadata"].items()
        if key not in ["producer", "creator"] and value is not None and value != ""
    }
    
    return {
        "book_id": book_id,
        "metadata": clean_metadata,
        "total_chapters": cached_data["total_chapters"]
    }

def list_books() -> List[Dict[str, Any]]:
    """List all available ebooks in the knowledge base.
    
    Returns:
        List of book metadata including title, author, and ID
    """
    books = scan_books_directory()
    result = []
    
    for book in books:
        book_path = Path(book["path"])
        book_data = _get_list_book_entry(book_path)
        result.append(book_data)
    
    return result

def search_books(query: Field(description="Search query for book titles or authors")) -> List[Dict[str, Any]]:
    """Search for books by title or author.
    
    Args:
        query: Search term for title or author
        
    Returns:
        List of matching books with metadata
    """
    query_lower = query.lower()
    all_books = list_books()
    
    return [
        book for book in all_books
        if query_lower in book["metadata"].get("title", "").lower() or
           query_lower in book["metadata"].get("author", "").lower()
    ]

def book_information(book_id: Field(description="Unique book identifier")) -> Dict[str, Any]:
    """Get detailed information about a specific book.
    
    Args:
        book_id: The book identifier
        
    Returns:
        Book metadata, chapter list, and other details
    """
    books = scan_books_directory()
    
    for book in books:
        if sanitize_filename(Path(book["path"]).stem) == book_id:
            return _cache_book_metadata(Path(book["path"]))
    
    return {"error": "Book not found", "book_id": book_id}

def read_pages(
    book_id: Field(description="Unique book identifier"),
    start_chapter: Field(description="Starting chapter number (1-based)", gt=0),
    end_chapter: Field(description="Ending chapter number (1-based, optional)", gt=0) = None,
) -> Dict[str, Any]:
    """Read specified chapters from a book.
    
    Args:
        book_id: Book identifier
        start_chapter: First chapter to read
        end_chapter: Last chapter to read (defaults to start_chapter)
        
    Returns:
        Content of specified chapters
    """
    end_chapter = end_chapter or start_chapter
    
    books = scan_books_directory()
    for book in books:
        if sanitize_filename(Path(book["path"]).stem) == book_id:
            chapters = extract_book_content(Path(book["path"]))
            
            if start_chapter > len(chapters):
                return {"error": "Chapter not found", "total_chapters": len(chapters)}
            
            end_chapter = min(end_chapter, len(chapters))
            selected_chapters = chapters[start_chapter-1:end_chapter]
            
            return {
                "book_id": book_id,
                "chapters": [{
                    "chapter_number": i + start_chapter,
                    "title": ch["title"],
                    "content": ch["content"]
                } for i, ch in enumerate(selected_chapters)]
            }
    
    return {"error": "Book not found", "book_id": book_id}

def search_contents(
    query: Field(description="Search query for book content"),
    book_id: Field(description="Book to search in (optional, searches all books if omitted") = None
) -> Dict[str, Any]:
    """Search book content using vector similarity.
    
    Args:
        query: Search query
        book_id: Specific book to search (optional)
        
    Returns:
        Matching content with context
    """
    from ..knowledge_retrieval.vector_store import search_topics
    import asyncio
    
    # Build filter for specific book if provided
    # ChromaDB requires $and operator for multiple conditions
    where_filter = None
    if book_id:
        where_filter = {
            "$and": [
                {"source": {"$eq": "ebooks"}},
                {"book_id": {"$eq": book_id}}
            ]
        }
    else:
        where_filter = {"source": {"$eq": "ebooks"}}
    
    # Execute search
    try:
        # Run async search in sync context
        try:
            loop = asyncio.get_running_loop()
            # Event loop is already running, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    search_topics(query, max_results=10, where=where_filter)
                )
                results = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            results = asyncio.run(search_topics(query, max_results=10, where=where_filter))
        
        return results
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "book_id": book_id,
            "total_results": 0,
            "results": []
        }

def _process_chapter_for_embedding(book_id: str, chapter: Dict[str, str], chapter_num: int, book_title: str = "") -> Dict[str, Any]:
    """Process a single chapter for embedding into vector store."""
    from ..knowledge_retrieval.vector_store import transcript_collection
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    try:
        # Skip empty chapters
        if not chapter["content"].strip():
            return {
                "success": False,
                "chunks_added": 0,
                "error": "Chapter has no content"
            }
        
        # Split chapter into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len
        )
        chunks = text_splitter.split_text(chapter["content"])
        
        # Remove existing chunks for this chapter if any
        try:
            existing_ids = transcript_collection.get(
                where={"source": "ebooks", "book_id": book_id, "chapter": chapter_num}
            )["ids"]
            
            if existing_ids:
                transcript_collection.delete(ids=existing_ids)
        except Exception:
            pass  # No existing chunks, continue
        
        # Add chunks to collection with ebook-specific metadata
        for i, chunk in enumerate(chunks):
            transcript_collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": "ebooks",
                    "book_id": book_id,
                    "book_title": book_title,
                    "chapter": chapter_num,
                    "chapter_title": chapter["title"],
                    "chunk": i
                }],
                ids=[f"ebook-{book_id}-chapter-{chapter_num}-chunk-{i}"]
            )
        
        return {
            "success": True,
            "chunks_added": len(chunks),
            "chapter": chapter_num,
            "chapter_title": chapter["title"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "chunks_added": 0,
            "error": str(e)
        }

def embed_book(book_id: Field(description="Book identifier to add to vector store")) -> Dict[str, Any]:
    """Process and embed a book's content into the vector search database.
    
    Args:
        book_id: Book identifier
        
    Returns:
        Processing status and details
    """
    books = scan_books_directory()
    
    for book in books:
        if sanitize_filename(Path(book["path"]).stem) == book_id:
            book_path = Path(book["path"])
            metadata = extract_book_metadata(book_path)
            chapters = extract_book_content(book_path)
            book_title = metadata.get("title", book_id)
            
            results = []
            total_chunks = 0
            for i, chapter in enumerate(chapters):
                result = _process_chapter_for_embedding(
                    book_id, chapter, i + 1, book_title
                )
                results.append({
                    "chapter": i + 1,
                    "title": chapter["title"],
                    "embedding_result": result
                })
                if result.get("success"):
                    total_chunks += result.get("chunks_added", 0)
            
            return {
                "book_id": book_id,
                "book_title": book_title,
                "chapters_processed": len(chapters),
                "total_chunks_added": total_chunks,
                "results": results
            }
    
    return {"error": "Book not found", "book_id": book_id}

def summarize_book(
    book_id: Field(description="Book identifier to summarize"),
    detail_level: Field(description="Summary detail level", default="medium") = "medium"
) -> Dict[str, Any]:
    """Generate a summary of a book using parallel processing.
    
    Args:
        book_id: Book identifier
        detail_level: Level of detail ("brief", "medium", "detailed")
        
    Returns:
        Book summary with chapter summaries
    """
    from .summarize import summarizer
    
    books = scan_books_directory()
    
    for book in books:
        if sanitize_filename(Path(book["path"]).stem) == book_id:
            book_path = Path(book["path"])
            chapters = extract_book_content(book_path)
            
            # Get book content for full summary
            full_content = " ".join([ch["content"] for ch in chapters])
            
            # Ensure we have content
            if not full_content.strip():
                return {
                    "book_id": book_id,
                    "total_chapters": len(chapters),
                    "summary_level": detail_level,
                    "summary": "Book has no readable content",
                    "processing_details": {
                        "chunks_processed": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0
                    },
                    "chapter_summaries": []
                }
            
            try:
                # Generate book-level summary
                book_summary = summarizer.summarize_book(full_content, detail_level)
                
                # Generate chapter-level summaries
                chapter_summaries = []
                for i, chapter in enumerate(chapters):
                    if chapter["content"].strip():
                        chapter_summary = summarizer.summarize_chapter(
                            chapter["content"], 
                            "medium"
                        )
                        summary_result = chapter_summary.get("summary", "Summary generation failed")
                    else:
                        summary_result = "Chapter has no readable content"
                    
                    chapter_summaries.append({
                        "chapter": i + 1,
                        "title": chapter["title"],
                        "summary": summary_result
                    })
                
                return {
                    "book_id": book_id,
                    "total_chapters": len(chapters),
                    "summary_level": detail_level,
                    "summary": book_summary.get("summary", "Summary generation failed"),
                    "processing_details": {
                        "chunks_processed": book_summary.get("chunks_processed", 0),
                        "total_input_tokens": book_summary.get("total_input_tokens", 0),
                        "total_output_tokens": book_summary.get("total_output_tokens", 0)
                    },
                    "chapter_summaries": chapter_summaries
                }
                
            except Exception as e:
                return {
                    "book_id": book_id,
                    "total_chapters": len(chapters),
                    "summary_level": detail_level,
                    "summary": f"Summary generation failed: {str(e)}",
                    "processing_details": {
                        "chunks_processed": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0
                    },
                    "chapter_summaries": []
                }
    
    return {"error": "Book not found", "book_id": book_id}

def summarize_chapter(
    book_id: Field(description="Book identifier"),
    chapter_number: Field(description="Chapter number to summarize", gt=0)
) -> Dict[str, Any]:
    """Generate a summary of a specific chapter.
    
    Args:
        book_id: Book identifier
        chapter_number: Chapter to summarize
        
    Returns:
        Chapter summary
    """
    from .summarize import summarizer
    
    result = read_pages(book_id, chapter_number)
    if "error" in result:
        return result
    
    chapter = result["chapters"][0]
    
    # Generate actual chapter summary
    chapter_summary = summarizer.summarize_chapter(
        chapter["content"], 
        detail_level="medium"
    )
    
    return {
        "book_id": book_id,
        "chapter_number": chapter_number,
        "title": chapter["title"],
        "summary": chapter_summary.get("summary", "Summary generation failed"),
        "processing_details": {
            "input_tokens": chapter_summary.get("total_input_tokens", 0),
            "output_tokens": chapter_summary.get("total_output_tokens", 0),
            "chunks_processed": chapter_summary.get("chunks_processed", 1)
        }
    }

def register_ebook_tools(mcp):
    """Register ebook tools with the MCP server."""
    mcp.tool()(list_books)
    mcp.tool()(search_books)
    mcp.tool()(book_information)
    mcp.tool()(read_pages)
    mcp.tool()(search_contents)
    mcp.tool()(summarize_book)
    mcp.tool()(summarize_chapter)
