# ============================================================================

import asyncio
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
import pytesseract
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class AdvancedDocumentProcessor:
    """Multi-method document extraction with high reliability"""
    
    def __init__(self):
        self.extraction_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Document processor initialized")
    
    async def process_document(self, url: str) -> Dict:
        """Main document processing pipeline"""
        logger.info(f"🔍 Processing document: {url[:50]}...")
        
        # Extract text using multiple methods
        extraction_result = await self.extract_with_all_methods(url)
        
        if not extraction_result.get('text'):
            logger.error(f"❌ Failed to extract text from document")
            return {"chunks": [], "metadata": {"error": "text_extraction_failed"}}
        
        # Chunk the document
        chunks = await self.create_smart_chunks(
            text=extraction_result['text'],
            metadata=extraction_result
        )
        
        # Generate embeddings for chunks
        chunks_with_embeddings = await self.generate_embeddings(chunks)
        
        logger.info(f"✅ Processed document: {len(chunks_with_embeddings)} chunks created")
        
        return {
            "chunks": chunks_with_embeddings,
            "metadata": {
                "total_chunks": len(chunks_with_embeddings),
                "extraction_method": extraction_result.get('method', 'unknown'),
                "confidence": extraction_result.get('confidence', 0.0),
                "document_length": len(extraction_result['text'])
            }
        }
    
    async def extract_with_all_methods(self, url: str) -> Dict:
        """Extract using ALL available methods and merge results"""
        doc_hash = hashlib.md5(url.encode()).hexdigest()
        
        if doc_hash in self.extraction_cache:
            logger.info(f"📋 Using cached extraction")
            return self.extraction_cache[doc_hash]
        
        try:
            # Download document
            pdf_bytes = await self._download_document(url)
            if not pdf_bytes:
                return {"text": "", "confidence": 0.0, "method": "download_failed"}
            
            # Try multiple extraction methods
            methods = [
                self._extract_with_pymupdf(pdf_bytes),
                self._extract_with_pdfplumber(pdf_bytes)
            ]
            
            # Add OCR for difficult documents
            if len(pdf_bytes.getvalue()) < 5 * 1024 * 1024:  # Only OCR files < 5MB
                methods.append(self._extract_with_ocr(pdf_bytes))
            
            results = []
            for method in methods:
                try:
                    result = await method
                    if result and result.get('text') and len(result['text']) > 100:
                        results.append(result)
                        logger.info(f"✅ {result['method']}: {len(result['text'])} chars")
                except Exception as e:
                    logger.warning(f"⚠️ Extraction method failed: {e}")
            
            if not results:
                logger.error("❌ All extraction methods failed")
                return {"text": "", "confidence": 0.0, "method": "all_failed"}
            
            # Merge results
            merged_result = self._merge_extraction_results(results)
            self.extraction_cache[doc_hash] = merged_result
            
            return merged_result
            
        except Exception as e:
            logger.error(f"💥 Document extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "failed", "error": str(e)}
    
    async def _download_document(self, url: str, max_retries: int = 3) -> Optional[BytesIO]:
        """Download document with retries"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                return BytesIO(response.content)
            except Exception as e:
                logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
        return None
    
    async def _extract_with_pymupdf(self, pdf_bytes: BytesIO) -> Dict:
        """Enhanced PyMuPDF extraction"""
        try:
            pdf_bytes.seek(0)
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            
            full_text = ""
            page_count = 0
            
            for page_num in range(min(doc.page_count, 50)):
                try:
                    page = doc[page_num]
                    page_count += 1
                    
                    # Extract text
                    text = page.get_text("text")
                    
                    if not text.strip():
                        # Try dict method for complex layouts
                        blocks = page.get_text("dict")
                        page_text = ""
                        for block in blocks.get("blocks", []):
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line.get("spans", []):
                                        page_text += span.get("text", "") + " "
                                    page_text += "\n"
                        text = page_text
                    
                    if text and text.strip():
                        cleaned_text = re.sub(r'\s+', ' ', text.strip())
                        full_text += cleaned_text + "\n"
                
                except Exception as e:
                    logger.warning(f"⚠️ Page {page_num} extraction failed: {e}")
                    continue
            
            doc.close()
            
            confidence = 0.95 if len(full_text) > 500 else 0.3
            confidence *= min(1.0, page_count / 10)
            
            return {
                "method": "pymupdf",
                "text": full_text.strip(),
                "confidence": confidence,
                "pages_processed": page_count
            }
            
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF extraction failed: {e}")
            return {"method": "pymupdf", "text": "", "confidence": 0.0}
    
    async def _extract_with_pdfplumber(self, pdf_bytes: BytesIO) -> Dict:
        """PDFPlumber extraction with table handling"""
        try:
            pdf_bytes.seek(0)
            full_text = ""
            page_count = 0
            table_count = 0
            
            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages[:30]:
                    try:
                        page_count += 1
                        
                        # Extract text
                        text = page.extract_text()
                        if text:
                            cleaned_text = re.sub(r'\s+', ' ', text.strip())
                            full_text += cleaned_text + "\n"
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                table_count += 1
                                table_text = "\n[TABLE START]\n"
                                for row in table:
                                    if row:
                                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                                        table_text += " | ".join(clean_row) + "\n"
                                table_text += "[TABLE END]\n"
                                full_text += table_text
                    
                    except Exception as e:
                        logger.warning(f"⚠️ PDFPlumber page error: {e}")
                        continue
            
            confidence = 0.9 if len(full_text) > 500 else 0.3
            if table_count > 0:
                confidence += 0.1
            
            return {
                "method": "pdfplumber",
                "text": full_text.strip(),
                "confidence": confidence,
                "pages_processed": page_count,
                "tables_found": table_count
            }
            
        except Exception as e:
            logger.warning(f"⚠️ PDFPlumber extraction failed: {e}")
            return {"method": "pdfplumber", "text": "", "confidence": 0.0}
    
    async def _extract_with_ocr(self, pdf_bytes: BytesIO) -> Dict:
        """OCR extraction for image-heavy PDFs"""
        try:
            pdf_bytes.seek(0)
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            
            full_text = ""
            max_pages = min(doc.page_count, 3)
            pages_processed = 0
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    pages_processed += 1
                    
                    # Convert to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    img = Image.open(BytesIO(img_data))
                    
                    # Preprocessing
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    denoised = cv2.fastNlMeansDenoising(gray)
                    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR
                    custom_config = r'--oem 3 --psm 6 -l eng'
                    text = pytesseract.image_to_string(thresh, config=custom_config)
                    
                    if text and text.strip():
                        cleaned_text = re.sub(r'\s+', ' ', text.strip())
                        full_text += cleaned_text + "\n"
                
                except Exception as e:
                    logger.warning(f"⚠️ OCR page {page_num} failed: {e}")
                    continue
            
            doc.close()
            confidence = 0.7 if len(full_text) > 200 else 0.2
            
            return {
                "method": "ocr",
                "text": full_text.strip(),
                "confidence": confidence,
                "pages_processed": pages_processed
            }
            
        except Exception as e:
            logger.warning(f"⚠️ OCR extraction failed: {e}")
            return {"method": "ocr", "text": "", "confidence": 0.0}
    
    def _merge_extraction_results(self, results: List[Dict]) -> Dict:
        """Intelligently merge extraction results"""
        if not results:
            return {"text": "", "confidence": 0.0, "method": "none"}
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        primary = results[0]
        merged_text = primary['text']
        
        # Merge unique content from other results
        for result in results[1:]:
            if result['confidence'] > 0.4:
                primary_sentences = set(s.strip() for s in merged_text.split('.') if len(s.strip()) > 20)
                result_sentences = set(s.strip() for s in result['text'].split('.') if len(s.strip()) > 20)
                unique_sentences = result_sentences - primary_sentences
                
                if unique_sentences:
                    additional_text = ". ".join(list(unique_sentences)[:10])
                    if additional_text:
                        merged_text += "\n\n" + additional_text
        
        final_confidence = primary['confidence']
        if len(results) > 1:
            final_confidence = min(0.98, final_confidence + 0.1)
        
        return {
            "text": merged_text,
            "confidence": final_confidence,
            "method": f"merged_{len(results)}_methods",
            "sources": [r['method'] for r in results],
            "total_length": len(merged_text)
        }
    
    async def create_smart_chunks(self, text: str, metadata: Dict, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Create intelligent text chunks with semantic boundaries"""
        if not text:
            return []
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                # Create chunk
                chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:16]
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": current_chunk.strip(),
                    "metadata": {
                        "sentence_count": len(current_sentences),
                        "char_count": len(current_chunk),
                        "source_confidence": metadata.get('confidence', 0.0)
                    }
                })
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_sentences) > 1:
                    overlap_sentences = current_sentences[-(overlap // 100):]  # Approximate overlap
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:16]
            chunks.append({
                "chunk_id": chunk_id,
                "content": current_chunk.strip(),
                "metadata": {
                    "sentence_count": len(current_sentences),
                    "char_count": len(current_chunk),
                    "source_confidence": metadata.get('confidence', 0.0)
                }
            })
        
        logger.info(f"📝 Created {len(chunks)} intelligent chunks")
        return chunks
    
    async def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for document chunks"""
        if not chunks:
            return []
        
        try:
            # Extract text content for embedding
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
            
            logger.info(f"🔢 Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return chunks

# ============================================================================
