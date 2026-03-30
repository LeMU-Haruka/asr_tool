from paths import resolve_model_path

CONFIG = {
    "active_model": "qwen_asr",
    "models": {
        "qwen_asr": {
            "provider": "qwen_asr",
            "model_path": "Qwen3-ASR-1.7B",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "max_inference_batch_size": 32,
            "max_new_tokens": 256,
            "language": None,
        },
        "whisper_asr": {
            "provider": "whisper_asr",
            "model_path": "whisper-large-v3",
            "fallback_model_paths": ["/private/models/whisper-large-v3"],
            "device": "cuda:0",
            "dtype": "float16",
            "language": None,
            "task": "transcribe",
            "chunk_length_s": 30,
        },
    },
}


def get_model_config(model_name: str) -> dict:
    model_config = dict(CONFIG["models"][model_name])
    model_config["model_path"] = resolve_model_path(
        model_config["model_path"],
        model_config.get("fallback_paths") or model_config.get("fallback_model_paths"),
    )
    return model_config
