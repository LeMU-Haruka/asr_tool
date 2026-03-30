import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_qwen_asr_model_class():
    # Qwen ASR does not need torchvision, and masking it avoids a broken local torchvision install.
    sys.modules["torchvision"] = None

    search_paths = []
    for entry in sys.path:
        resolved = Path(entry or ".").resolve()
        if resolved != PROJECT_DIR:
            search_paths.append(entry)

    spec = importlib.machinery.PathFinder.find_spec("qwen_asr", search_paths)
    if spec is None or spec.loader is None:
        raise ImportError("Please install qwen-asr first: pip install -U qwen-asr")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen_asr"] = module
    spec.loader.exec_module(module)
    return module.Qwen3ASRModel


def load_model(model_config: dict):
    if str(model_config["device"]).startswith("cuda"):
        # This environment can use CUDA, but cuDNN initialization is unstable.
        torch.backends.cudnn.enabled = False

    qwen_asr_model = get_qwen_asr_model_class()
    return qwen_asr_model.from_pretrained(
        model_config["model_path"],
        dtype=DTYPE_MAP[model_config["dtype"]],
        device_map=model_config["device"],
        max_inference_batch_size=model_config["max_inference_batch_size"],
        max_new_tokens=model_config["max_new_tokens"],
    )


def transcribe(model, audio_path: str, model_config: dict):
    results = model.transcribe(
        audio=str(Path(audio_path).expanduser()),
        language=model_config["language"],
    )
    return results[0]
