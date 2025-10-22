"""PDF processing utilities for ebook analysis."""
from pathlib import Path
from typing import Dict, List, Any

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

def extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """Extract metadata from PDF file."""
    if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        return {
            'title': pdf_path.stem,
            'author': 'Unknown',
            'type': 'pdf',
            'error': 'PDF processing libraries not installed'
        }
    
    try:
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(str(pdf_path)) as pdf:
                metadata = {
                    'title': pdf.metadata.get('Title', pdf_path.stem),
                    'author': pdf.metadata.get('Author', 'Unknown'),
                    'subject': pdf.metadata.get('Subject', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'creation_date': str(pdf.metadata.get('CreationDate', '')),
                    'modification_date': str(pdf.metadata.get('ModDate', '')),
                    'pages': len(pdf.pages),
                    'type': 'pdf'
                }
                return metadata
        else:
            with PyPDF2.PdfReader(str(pdf_path)) as reader:
                metadata = {
                    'title': reader.metadata.get('/Title', pdf_path.stem) if reader.metadata else pdf_path.stem,
                    'author': reader.metadata.get('/Author', 'Unknown') if reader.metadata else 'Unknown',
                    'pages': len(reader.pages),
                    'type': 'pdf'
                }
                return metadata
                
    except Exception as e:
        return {
            'title': pdf_path.stem,
            'author': 'Unknown',
            'type': 'pdf',
            'error': str(e)
        }

def extract_pdf_chapters(pdf_path: Path) -> List[Dict[str, str]]:
    """Extract content from PDF file. Treats pages as chapters."""
    if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        return [{'title': 'Error', 'content': 'PDF processing libraries not installed', 'filename': 'error'}]
    
    try:
        chapters = []
        
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    # Clean up text
                    text = ' '.join(text.split())
                    
                    chapters.append({
                        'title': f'Page {i + 1}',
                        'content': text,
                        'filename': f'page_{i + 1}'
                    })
        else:
            with PyPDF2.PdfReader(str(pdf_path)) as reader:
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = ' '.join(text.split())
                    
                    chapters.append({
                        'title': f'Page {i + 1}',
                        'content': text,
                        'filename': f'page_{i + 1}'
                    })
        
        return chapters
    except Exception as e:
        return [{'title': 'Error', 'content': f'Error processing PDF: {str(e)}', 'filename': 'error'}]
