from pydantic import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://ollama:11434"
    chroma_db_url: str = "http://chroma:8000"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    class Config:
        env_file = ".env"

settings = Settings()
