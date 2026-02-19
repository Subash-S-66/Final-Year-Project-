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
    vote_window: int = 8
    # Tuned for low-data models where calibrated top-1 confidence is often ~0.10-0.30.
    min_confidence_enter: float = 0.18
    min_confidence_exit: float = 0.12
    min_vote_ratio: float = 0.50
    debounce_frames: int = 3
    release_frames: int = 6
    # Fallback stricter thresholds when vocab has no explicit NO_SIGN class.
    min_confidence_enter_no_no_sign: float = 0.30
    min_confidence_exit_no_no_sign: float = 0.20
    min_vote_ratio_no_no_sign: float = 0.45
    min_margin_no_no_sign: float = 0.08
    # Require sustained hand presence + motion before sign emission.
    min_hand_presence_ratio: float = 0.25
    min_motion_score: float = 0.020
    min_motion_score_no_no_sign: float = 0.020
    # For sign vocab-only models, require at least one hand to consider sign emission.
    require_hand_for_sign: bool = True

    # Phase 8: ONNX inference
    use_onnx: bool = True
    onnx_model_path: str = "artifacts/model_lstm_include30.onnx"
    onnx_meta_path: str = "artifacts/model_lstm_include30.meta.json"
    vocab_path: str = "configs/vocab_include_30.json"
    no_sign_token: str = "NO_SIGN"


settings = Settings()
