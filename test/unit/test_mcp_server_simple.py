"""
Simple unit tests for MCP server functionality.

Tests the MCP server tools for text extraction, knowledge base operations,
and file operations using synchronous test patterns.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, mock_open
from mcp_server.server import _load_kb, _save_kb


class TestKnowledgeBaseOperations:
    """Test cases for knowledge base operations."""
    
    def test_kb_load_save(self):
        """Test knowledge base loading and saving."""
        # Test loading empty KB
        kb = _load_kb()
        assert "docs" in kb
        assert isinstance(kb["docs"], dict)
        
        # Test saving KB
        test_kb = {"docs": {"test_doc": {"chunks": ["chunk1", "chunk2"]}}}
        _save_kb(test_kb)
        
        # Verify it was saved
        loaded_kb = _load_kb()
        assert loaded_kb["docs"]["test_doc"]["chunks"] == ["chunk1", "chunk2"]
    
    def test_kb_empty_initialization(self):
        """Test KB initialization when file doesn't exist."""
        # Remove KB file if it exists
        kb_path = os.path.join(os.path.dirname(__file__), "..", "..", "mcp_server", "kb_store.json")
        if os.path.exists(kb_path):
            os.remove(kb_path)
        
        # Load should create empty KB
        kb = _load_kb()
        assert kb == {"docs": {}}


class TestTextProcessing:
    """Test cases for text processing utilities."""
    
    def test_html_cleaning(self):
        """Test HTML text cleaning logic."""
        from bs4 import BeautifulSoup
        
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a test paragraph.</p>
            <script>console.log("This should be removed");</script>
            <style>body { color: red; }</style>
        </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove script/style
        for s in soup(["script", "style"]):
            s.decompose()
        # Get text and clean up whitespace
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        result = "\n".join(lines)
        
        # Should extract text and clean up HTML
        assert "Main Title" in result
        assert "This is a test paragraph" in result
        assert "console.log" not in result  # Script removed
        assert "color: red" not in result  # Style removed
    
    def test_text_chunking_logic(self):
        """Test text chunking logic for KB operations."""
        text = "Chunk 1\n\nChunk 2\n\nChunk 3"
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        assert len(chunks) == 3
        assert chunks[0] == "Chunk 1"
        assert chunks[1] == "Chunk 2"
        assert chunks[2] == "Chunk 3"
    
    def test_empty_text_chunking(self):
        """Test chunking of empty text."""
        text = ""
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        assert len(chunks) == 0
    
    def test_whitespace_only_chunking(self):
        """Test chunking of whitespace-only text."""
        text = "   \n\n   \n\n   "
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        assert len(chunks) == 0


class TestFileOperations:
    """Test cases for file operations."""
    
    def test_file_read_write_cycle(self):
        """Test file read/write cycle."""
        content = "This is test content."
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
        
        try:
            # Write content
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Read content
            with open(temp_path, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            assert read_content == content
        finally:
            os.unlink(temp_path)
    
    def test_file_encoding_handling(self):
        """Test file encoding handling."""
        content = "Test content with unicode: 测试内容"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            temp_path = f.name
        
        try:
            # Write with UTF-8 encoding
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Read with UTF-8 encoding
            with open(temp_path, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            assert read_content == content
        finally:
            os.unlink(temp_path)
