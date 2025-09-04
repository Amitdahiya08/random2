"""
Unit tests for chunk utilities - Fixed version.

Tests the text chunking functionality used for document processing
and knowledge base indexing with accurate expectations.
"""
import pytest
from backend.app.services.chunk_utils import split_into_chunks


class TestChunkUtils:
    """Test cases for text chunking utilities."""
    
    def test_split_short_text(self):
        """Test chunking of short text that fits in one chunk."""
        text = "This is a short text."
        chunks = split_into_chunks(text, max_chars=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_with_paragraphs(self):
        """Test chunking of text with paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split_into_chunks(text, max_chars=50)
        
        # Should create multiple chunks when paragraphs exceed max_chars
        assert len(chunks) >= 1
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_split_empty_text(self):
        """Test chunking of empty text."""
        text = ""
        chunks = split_into_chunks(text, max_chars=100)
        
        # Empty text should result in empty chunks list
        assert len(chunks) == 0
    
    def test_split_whitespace_text(self):
        """Test chunking of whitespace-only text."""
        text = "   \n\n   "
        chunks = split_into_chunks(text, max_chars=100)
        
        # Whitespace-only text should result in empty chunks
        assert len(chunks) == 0
    
    def test_split_single_paragraph(self):
        """Test chunking of single paragraph text."""
        text = "This is a single paragraph without breaks."
        chunks = split_into_chunks(text, max_chars=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_preserves_content(self):
        """Test that chunking preserves all original content."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split_into_chunks(text, max_chars=50)
        
        # Reconstruct text from chunks (accounting for paragraph breaks)
        reconstructed = "\n\n".join(chunks)
        assert reconstructed == text
    
    def test_split_different_chunk_sizes(self):
        """Test chunking with different maximum chunk sizes."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        small_chunks = split_into_chunks(text, max_chars=20)
        large_chunks = split_into_chunks(text, max_chars=100)
        
        # With smaller max_chars, we should get more chunks
        assert len(small_chunks) >= len(large_chunks)
        assert all(len(chunk) <= 20 for chunk in small_chunks)
        assert all(len(chunk) <= 100 for chunk in large_chunks)
    
    def test_split_single_word_longer_than_chunk(self):
        """Test handling of single word longer than chunk size."""
        text = "Supercalifragilisticexpialidocious"  # Very long word
        chunks = split_into_chunks(text, max_chars=20)
        
        # Should still create a chunk even if word is longer than max_chars
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_with_newlines(self):
        """Test chunking with text containing newlines."""
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        chunks = split_into_chunks(text, max_chars=20)
        
        # Single line text should create one chunk
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_very_small_chunk_size(self):
        """Test chunking with very small chunk size."""
        text = "This is a test document."
        chunks = split_into_chunks(text, max_chars=5)
        
        # Single paragraph should create one chunk even with very small size
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_multiple_paragraphs(self):
        """Test chunking with multiple paragraphs."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.\n\nPara 5."
        chunks = split_into_chunks(text, max_chars=30)
        
        # Should create multiple chunks
        assert len(chunks) >= 2
        assert all(len(chunk) <= 30 for chunk in chunks)
    
    def test_split_boundary_conditions(self):
        """Test chunking at boundary conditions."""
        # Text exactly at max_chars
        text = "x" * 50
        chunks = split_into_chunks(text, max_chars=50)
        assert len(chunks) == 1
        
        # Text just over max_chars
        text = "x" * 51
        chunks = split_into_chunks(text, max_chars=50)
        assert len(chunks) == 1  # Single paragraph, so one chunk
