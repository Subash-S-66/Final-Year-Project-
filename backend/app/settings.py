from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISL_", env_file=".env", extra="ignore")

    api_name: str = "isl-translator-api"
    debug: bool = True

    # Streaming defaults (can be overridden via env vars)
    expected_encoding: str = "jpeg"
    max_frame_bytes: int = 1_500_000  # ~1.5MB safety limit

    # Windowing for inference
    window_size: int = 30  # frames (T)
    min_confidence: float = 0.7  # threshold for committed predictions

    # Phase 8: ONNX inference
    use_onnx: bool = True
    onnx_model_path: str = "artifacts/model_lstm_v1.onnx"
    vocab_path: str = "configs/vocab_v1_51.json"


settings = Settings()
