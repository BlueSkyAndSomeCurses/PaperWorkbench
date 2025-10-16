from pathlib import Path


def handle_file_reading_for_model(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as f:
        if file_path.suffix in (".csv", ".parquet"):
            return "".join(f.readline() for _ in range(6))

        return f.read()
