import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import librosa
import torch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@contextmanager
def quiet_transformers_logging():
    logger_names = [
        "transformers",
        "transformers.generation.utils",
        "transformers.generation.configuration_utils",
        "transformers.pipelines.base",
    ]
    loggers = [logging.getLogger(name) for name in logger_names]
    previous_levels = [logger.level for logger in loggers]

    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.setLevel(level)


def get_whisper_modules():
    sys.modules["torchvision"] = None

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    return AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def resolve_pipeline_device(device: str):
    if str(device).startswith("cuda:"):
        return int(str(device).split(":", 1)[1])
    if str(device).startswith("cuda"):
        return 0
    return -1


def load_model(model_config: dict):
    if str(model_config["device"]).startswith("cuda"):
        torch.backends.cudnn.enabled = False

    auto_model, auto_processor, asr_pipeline = get_whisper_modules()
    dtype = DTYPE_MAP[model_config["dtype"]]

    with quiet_transformers_logging():
        model = auto_model.from_pretrained(
            model_config["model_path"],
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = auto_processor.from_pretrained(model_config["model_path"])

    if str(model_config["device"]).startswith("cuda"):
        model = model.to(model_config["device"])

    with quiet_transformers_logging():
        return asr_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=resolve_pipeline_device(model_config["device"]),
            chunk_length_s=model_config["chunk_length_s"],
            ignore_warning=True,
        )


def transcribe(model, audio_path: str, model_config: dict):
    generate_kwargs = {
        "task": model_config["task"],
    }
    if model_config["language"] is not None:
        generate_kwargs["language"] = model_config["language"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="PySoundFile failed\\. Trying audioread instead\\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="librosa\\.core\\.audio\\.__audioread_load",
            category=FutureWarning,
        )
        audio_array, sampling_rate = librosa.load(
            str(Path(audio_path).expanduser()),
            sr=model.feature_extractor.sampling_rate,
            mono=True,
        )

    with quiet_transformers_logging():
        result = model(
            {
                "array": audio_array,
                "sampling_rate": sampling_rate,
            },
            generate_kwargs=generate_kwargs,
        )
    return SimpleNamespace(
        language=result.get("language", model_config["language"]),
        text=result["text"].strip(),
    )
