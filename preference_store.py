import json
import os
import tempfile
from pathlib import Path
from typing import Any


def empty_preference_store() -> dict[str, Any]:
    return {"schema_version": 1, "students": {}}


def load_preference_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_preference_store()
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load preferences from {path}: {error}") from error
    if (
        not isinstance(store, dict)
        or store.get("schema_version") != 1
        or not isinstance(store.get("students"), dict)
    ):
        raise ValueError(f"{path} does not match the student preference format.")
    for student_id, student_data in store["students"].items():
        if (
            not isinstance(student_id, str)
            or not isinstance(student_data, dict)
            or not isinstance(student_data.get("preferences"), list)
            or not isinstance(student_data.get("rounds"), list)
            or not isinstance(student_data.get("conversations", []), list)
        ):
            raise ValueError(f"{path} contains an invalid student preference record.")
    return store


def save_preference_store(store: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(store, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
