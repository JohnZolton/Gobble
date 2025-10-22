"""Unified book processing utilities for EPUB and PDF files."""
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Import format-specific utilities
from .pdf.pdf_utils import extract_pdf_metadata, extract_pdf_chapters

def scan_books_directory() -> List[Dict[str, Any]]:
    """Scan for book files (EPUB and PDF) in knowledge/ebooks/."""
    book_dir = Path("knowledge/ebooks")
    if not book_dir.exists():
        return []
    
    books = []
    for ext in ['*.epub', '*.pdf']:
        for book_file in book_dir.rglob(ext):
            book_id = sanitize_filename(book_file.stem)
            books.append({
                'path': str(book_file),
                'book_id': book_id,
                'name': book_file.name,
                'type': book_file.suffix.lower().lstrip('.')
            })
    
    return books

def sanitize_filename(name: str) -> str:
    """Sanitize book title for directory names."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '-', name)
    return name.strip().lower()[:50]

def extract_book_metadata(book_path: Path) -> Dict[str, Any]:
    """Extract metadata from any book file (EPUB or PDF)."""
    if book_path.suffix.lower() == '.epub':
        return extract_epub_metadata(book_path)
    elif book_path.suffix.lower() == '.pdf':
        return extract_pdf_metadata(book_path)
    else:
        return {
            'title': book_path.stem,
            'author': 'Unknown',
            'type': 'unsupported',
            'error': f'Unsupported format: {book_path.suffix}'
        }

def extract_epub_metadata(epub_path: Path) -> Dict[str, Any]:
    """Extract metadata from EPUB file."""
    try:
        book = epub.read_epub(str(epub_path))
        
        metadata = {
            'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else epub_path.stem,
            'author': book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else 'Unknown',
            'language': book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else 'Unknown',
            'description': book.get_metadata('DC', 'description')[0][0] if book.get_metadata('DC', 'description') else '',
            'identifier': book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else str(epub_path.stem),
            'publisher': book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else '',
            'publish_date': book.get_metadata('DC', 'date')[0][0] if book.get_metadata('DC', 'date') else '',
            'type': 'epub'
        }
        
        return metadata
    except Exception as e:
        return {
            'title': epub_path.stem,
            'author': 'Unknown',
            'type': 'epub',
            'error': str(e)
        }

def extract_epub_chapters(epub_path: Path) -> List[Dict[str, str]]:
    """Extract chapter content from EPUB file."""
    try:
        book = epub.read_epub(str(epub_path))
        
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Get chapter title from file name or first heading
                title = item.get_name()
                if soup.find('h1'):
                    title = soup.find('h1').get_text().strip()
                elif soup.find('h2'):
                    title = soup.find('h2').get_text().strip()
                
                chapters.append({
                    'title': title,
                    'content': text,
                    'filename': item.get_name()
                })
        
        return chapters
    except Exception as e:
        return [{'title': 'Error', 'content': f'Error processing EPUB: {str(e)}', 'filename': 'error'}]

def extract_book_content(book_path: Path) -> List[Dict[str, str]]:
    """Extract content from any book file (EPUB or PDF)."""
    if book_path.suffix.lower() == '.epub':
        return extract_epub_chapters(book_path)
    elif book_path.suffix.lower() == '.pdf':
        return extract_pdf_chapters(book_path)
    else:
        return [{'title': 'Error', 'content': f'Unsupported format: {book_path.suffix}', 'filename': 'error'}]

def get_book_directory(book_id: str) -> Path:
    """Get directory path for a book."""
    return Path("knowledge/ebooks") / book_id

def ensure_book_directories() -> None:
    """Ensure ebook directories exist."""
    Path("knowledge/ebooks").mkdir(parents=True, exist_ok=True)
