"""
Configuration settings for LLM evaluation framework.
All sensitive values must be provided via environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    eval_model_name: str = "ADD_YOUR_OWN_MODEL_NAME"
    llm_provider: str = "openai"
    
    # Evaluation Configuration
    eval_dataset_path: str = "data/samples/eval_dataset.json"
    batch_size: int = 10
    max_concurrent_evals: int = 5
    eval_timeout: int = 60
    
    # Hallucination Detection
    hallucination_threshold: float = 0.7
    grounding_threshold: float = 0.6
    citation_required: bool = True
    unsupported_claim_detection: bool = True
    
    # Grounding Configuration
    context_overlap_threshold: float = 0.5
    retrieval_confidence_threshold: float = 0.7
    answer_context_alignment: bool = True
    
    # Adversarial Testing
    adversarial_enabled: bool = True
    adversarial_variants: int = 5
    stress_test_enabled: bool = True
    
    # Vector Database
    vector_db_url: Optional[str] = None
    vector_db_type: str = "pinecone"
    
    # Output Configuration
    output_format: str = "json"
    reports_dir: str = "reports"
    generate_summary: bool = True
    generate_detailed_report: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
