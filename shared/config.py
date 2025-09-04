import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    app_env: str = Field(default=os.getenv("APP_ENV", "dev"))
    data_dir: str = Field(default=os.getenv("DATA_DIR", "./storage_data"))

    # Azure OpenAI
    az_endpoint: str = Field(default=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    az_api_key: str = Field(default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    az_api_version: str = Field(default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
    az_deployment: str = Field(default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
    az_model: str = Field(default=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-2025-04-14"))

    # MCP
    mcp_server_path: str = Field(default=os.getenv("MCP_SERVER_PATH", "./mcp_server/server.py"))

    # LangSmith
    langsmith_api_key: str = Field(default=os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = Field(default=os.getenv("LANGSMITH_PROJECT", "docqa-mcp"))
    langsmith_tracing: bool = Field(default=os.getenv("LANGSMITH_TRACING", "0") == "1")

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)
