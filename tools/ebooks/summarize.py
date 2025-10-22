"""Ebook summarization functionality using OpenAI-compatible APIs."""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextSummarizer:
    """Handle text summarization with automatic chunking and parallel processing."""
    
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_url = os.getenv("API_URL", "https://api.openai.com/v1/chat/completions")
        self.model = os.getenv("SUMMARY_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def count_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English text)."""
        return len(text) // 4
    
    def split_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Split text into chunks under token limit, trying to split at natural boundaries."""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            chunk_tokens = self.count_tokens(current_chunk)
            
            if chunk_tokens + paragraph_tokens < max_tokens:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
                else:
                    # Paragraph too big, split by sentences
                    sentences = paragraph.split('. ')
                    sentence_chunk = ""
                    for sentence in sentences:
                        if self.count_tokens(sentence_chunk + sentence + '. ') < max_tokens:
                            sentence_chunk += sentence + '. '
                        else:
                            if sentence_chunk:
                                chunks.append(sentence_chunk.strip())
                                sentence_chunk = sentence + '. '
                            else:
                                # Sentence too big, split by words
                                words = sentence.split()
                                word_chunk = ""
                                for word in words:
                                    if self.count_tokens(word_chunk + word + " ") < max_tokens:
                                        word_chunk += word + " "
                                    else:
                                        if word_chunk:
                                            chunks.append(word_chunk.strip())
                                            word_chunk = word + " "
                                if word_chunk:
                                    chunks.append(word_chunk.strip())
                    
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def summarize_chunk(self, session: aiohttp.ClientSession, text: str, detail_level: str) -> str:
        """Summarize a single text chunk."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        detail_prompts = {
            "brief": "Provide a concise summary focusing on the main points.",
            "medium": "Provide a detailed summary covering key topics and themes.",
            "detailed": "Provide a comprehensive summary with in-depth coverage of all topics."
        }
        
        prompt = f"{detail_prompts.get(detail_level, detail_prompts['medium'])}\n\nText to summarize:\n{text}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides accurate and clear summaries."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000 if detail_level == "detailed" else 1000,
            "temperature": 0.5
        }
        
        async with session.post(self.api_url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API call failed with status {response.status}")
    
    async def summarize_chunks_parallel(self, chunks: List[str], detail_level: str) -> str:
        """Summarize multiple chunks in parallel with retries."""
        max_retries = 3
        retry_delay = 1
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    # Process chunks in parallel
                    tasks = [
                        self.summarize_chunk(session, chunk, detail_level) 
                        for chunk in chunks
                    ]
                    
                    summaries = await asyncio.gather(*tasks)
                    
                    # Combine summaries if multiple chunks
                    if len(summaries) == 1:
                        return summaries[0]
                    else:
                        combined_text = "\n\n".join(summaries)
                        
                        # If the combined summary is too long, summarize it again
                        if self.count_tokens(combined_text) > 4000:
                            final_chunk = self.split_text(combined_text, 4000)[0]
                            final_task = [self.summarize_chunk(session, final_chunk, detail_level)]
                            final_summary = await asyncio.gather(*final_task)
                            return final_summary[0]
                        else:
                            return combined_text
                            
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
    
    def summarize_book(self, book_content: str, detail_level: str = "medium") -> Dict[str, Any]:
        """Summarize an entire book with parallel processing."""
        if self.count_tokens(book_content) == 0:
            return {"error": "Empty book content provided"}
        
        chunks = self.split_text(book_content, 8000)
        
        try:
            # Handle async execution properly - check if event loop is running
            try:
                loop = asyncio.get_running_loop()
                # Event loop is already running, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.summarize_chunks_parallel(chunks, detail_level)
                    )
                    summary = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                summary = asyncio.run(self.summarize_chunks_parallel(chunks, detail_level))
            
            return {
                "summary": summary,
                "detail_level": detail_level,
                "chunks_processed": len(chunks),
                "total_input_tokens": self.count_tokens(book_content),
                "total_output_tokens": self.count_tokens(summary)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def summarize_chapter(self, chapter_content: str, detail_level: str = "medium") -> Dict[str, Any]:
        """Summarize a single chapter."""
        return self.summarize_book(chapter_content, detail_level)

# Global instance for easy access
summarizer = TextSummarizer()
