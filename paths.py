from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


def resolve_model_path(
    model_path: str | Path,
    fallback_paths: Iterable[str | Path] | None = None,
) -> str:
    requested_path = Path(model_path).expanduser()
    candidates: list[Path] = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(MODELS_DIR / requested_path)
        legacy_path = ROOT_DIR / requested_path
        if legacy_path != candidates[0]:
            candidates.append(legacy_path)

    for fallback_path in fallback_paths or ():
        candidate = Path(fallback_path).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])
