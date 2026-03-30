import argparse
from config import CONFIG
from providers import qwen_asr as qwen_provider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = CONFIG["models"]["qwen_asr"]
    model = qwen_provider.load_model(model_config)
    result = qwen_provider.transcribe(model, args.audio, model_config)
    print(result.language)
    print(result.text)


if __name__ == "__main__":
    main()
