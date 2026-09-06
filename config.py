import os
from pydantic_settings import BaseSettings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class BaseSettingsWrapper(BaseSettings):
    class Config:
        env_file = os.path.join(PROJECT_ROOT, ".env")
        case_sensitive = True
        extra = "allow"
 
class AppConfig(BaseSettingsWrapper):
    OPENAI_MODEL: str = "gpt-4o-mini"
    PINECONE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    INDEX_NAME: str = "document-analysis"
    MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models")
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    OLLAMA_API_BASE: str = "http://localhost:11434/api"
    OLLAMA_MODEL: str = "phi3:mini"
    COMPANY_PDF_MAPPING: dict = {
        "amazon": ["Amazon10k2022.pdf"],
        "apple": ["Apple_10-K-2021.pdf"],
        "nvidia": ["Nvidia.pdf"],
        "tesla": ["Tesla.pdf"]
    }

app_config = AppConfig()
