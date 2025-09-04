"""
Pytest configuration and shared fixtures for all tests.

This module provides common fixtures and configuration for unit tests,
integration tests, and performance tests.
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Main Title</h1>
        <p>This is a test paragraph with some content.</p>
        <h2>Subsection</h2>
        <p>Another paragraph with more detailed information.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
            <li>Third item</li>
        </ul>
    </body>
    </html>
    """


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    Section 1: Introduction
    This document discusses machine learning concepts and applications.
    
    Section 2: Main Content
    Machine learning is a subset of artificial intelligence that focuses on algorithms.
    There are three main types: supervised, unsupervised, and reinforcement learning.
    
    Section 3: Conclusion
    This concludes our discussion of machine learning.
    """


@pytest.fixture
def mock_agent_responses():
    """Mock responses for agent operations."""
    return {
        "parser": {
            "sections": ["Section 1: Introduction", "Section 2: Main Content"],
            "raw_text": "This is a test document about machine learning."
        },
        "summarizer": "This document discusses machine learning concepts and applications.",
        "entity_extractor": ["machine learning", "artificial intelligence", "algorithms"],
        "qa": "Machine learning is a subset of artificial intelligence."
    }


@pytest.fixture
def mock_mcp_responses():
    """Mock responses for MCP operations."""
    return {
        "extract_text": "This is extracted text from the document.",
        "kb_add": "Indexed 3 chunks for test_doc",
        "kb_search": "Machine learning is a subset of artificial intelligence."
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ["APP_ENV"] = "test"
    os.environ["LANGSMITH_TRACING"] = "0"  # Disable tracing in tests
    
    yield
    
    # Cleanup after test
    # Remove any test-specific environment variables if needed
    pass


@pytest.fixture
def mock_azure_client():
    """Mock Azure OpenAI client for testing."""
    with patch('backend.app.services.agent_registry._azure_client') as mock_client:
        mock_client.return_value = AsyncMock()
        yield mock_client


@pytest.fixture
def mock_mcp_bridge():
    """Mock MCP bridge for testing."""
    with patch('backend.app.services.mcp_bridge.mcp_bridge') as mock_bridge:
        mock_bridge.start = AsyncMock()
        mock_bridge.stop = AsyncMock()
        mock_bridge.call = AsyncMock(return_value="Mock MCP response")
        yield mock_bridge


# Performance test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")


# Skip performance tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip performance tests by default."""
    if not config.getoption("--run-performance"):
        skip_performance = pytest.mark.skip(reason="Performance tests require --run-performance flag")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
