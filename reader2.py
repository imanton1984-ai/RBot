import os
from pathlib import Path

OUTPUT_FILE   = "ml_entry.txt"
TARGET_FOLDER = Path("/home/anton/Desktop/Rust_trader/strategies/ml_entry_strategy")
IGNORED_DIRS  = {"__pycache__", ".git", "node_modules", "target", "dist", "build", "venv", "third_party"}
IGNORED_FILES = set()
EXTENSIONS    = {".py", ".cu", ".ptx", ".rs", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".env", ".sql", "dockerfile", ".ts", ".tsx", ".css", ".html"}

def is_target(file: Path) -> bool:
    return (
        file.suffix in EXTENSIONS
        and file.name not in IGNORED_FILES
        and not any(part.startswith('.') for part in file.parts if part != '.')
    )

Path(OUTPUT_FILE).write_text("", encoding="utf-8")

with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for root, dirs, files in os.walk(TARGET_FOLDER):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for name in sorted(files):
            file = Path(root, name)
            if not is_target(file):
                continue

            rel_path = file.relative_to(TARGET_FOLDER)
            out.write(f"# {rel_path}\n\n")
            try:
                out.write(file.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                out.write(file.read_text(encoding="cp1251", errors="replace"))
            out.write("\n\n\n")
