# ASR Tool

Local speech recognition tool with Qwen ASR and Whisper backends.

## Features

- Transcribe local audio files from the command line
- Switch between `qwen_asr` and `whisper_asr`
- Print recognized text and returned language info
- Load model weights from a project-local `models/` directory

## Model Weights

Create the `models/` directory manually in the project root, then place model weights like this:

```text
asr_tool/
├── models/
│   ├── Qwen3-ASR-1.7B/
│   └── whisper-large-v3/
```

Current expected locations:

- `models/Qwen3-ASR-1.7B/`
- `models/whisper-large-v3/`

The `models/` directory is ignored by Git and should stay local.

## Usage

Install dependencies:

```bash
uv sync
```

Run transcription:

```bash
uv run python main.py /path/to/audio.wav --model qwen_asr
uv run python main.py /path/to/audio.wav --model whisper_asr
```

If your environment is already activated, you can also use:

```bash
python main.py /path/to/audio.wav --model qwen_asr
```
