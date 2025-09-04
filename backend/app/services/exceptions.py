class PipelineError(Exception):
    """Base class for pipeline exceptions."""

class ParsingError(PipelineError):
    """Raised when parsing fails or returns unusable output."""

class SummarizationError(PipelineError):
    """Raised when summarization fails."""

class EntityExtractionError(PipelineError):
    """Raised when entity extraction fails."""

class QAError(PipelineError):
    """Raised when Q&A fails."""

class ValidationError(PipelineError):
    """Raised when validation fails; include details in message."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}
