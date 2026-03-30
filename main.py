import argparse

from config import CONFIG, get_model_config
from providers import qwen_asr as qwen_provider
from providers import whisper_asr as whisper_provider


PROVIDERS = {
    "qwen_asr": qwen_provider,
    "whisper_asr": whisper_provider,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", choices=CONFIG["models"].keys(), default=None)
    return parser.parse_args()


def print_result(active_model, result):
    print(f"model: {active_model}")
    if result.language is None:
        print("language: not returned by this provider")
    else:
        print(f"language: {result.language}")
    print("text:")
    print(result.text)


def main():
    args = parse_args()
    active_model = args.model or CONFIG["active_model"]
    model_config = get_model_config(active_model)
    provider = PROVIDERS[model_config["provider"]]
    model = provider.load_model(model_config)
    result = provider.transcribe(model, args.audio, model_config)
    print_result(active_model, result)


if __name__ == "__main__":
    main()
