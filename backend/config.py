"""Configuration shared by the application and benchmark."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str = 'http://127.0.0.1:11434'
    generation_model: str = 'llama3.2'
    embedding_model: str = 'nomic-embed-text'
    context_tokens: int = 32768
    output_tokens: int = 256
    boundary_words: int = 400
    minimum_region_words: int = 500
    maximum_depth: int = 8
    embedding_chunk_words: int = 256
    maximum_text_characters: int = 5000000
    request_timeout_seconds: int = 600
    seed: int = 42

    @classmethod
    def from_environment(cls):
        return cls(
            ollama_url=os.getenv('OLLAMA_URL', cls.ollama_url).rstrip('/'),
            generation_model=os.getenv('GENERATION_MODEL', cls.generation_model),
            embedding_model=os.getenv('EMBEDDING_MODEL', cls.embedding_model),
            context_tokens=int(os.getenv('CONTEXT_TOKENS', str(cls.context_tokens))),
            output_tokens=int(os.getenv('OUTPUT_TOKENS', str(cls.output_tokens))),
        )
