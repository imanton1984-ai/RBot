import os
from pathlib import Path

OUTPUT_FILE   = "rust_bot.txt"
THIS_SCRIPT   = Path(__file__).resolve()
IGNORED_DIRS  = {".venv", "External Libraries", "Scratches and Consoles", "__pycache__", ".git", "node_modules", "target", "dist", "build", "venv"}
IGNORED_FILES = {"reverse.py"}  # ← исключаем конкретные файлы
EXTENSIONS    = {".py", ".cu", ".ptx", ".rs", ".sh", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini", ".env", ".sql"}

def is_target(file: Path) -> bool:
    return (
        file.suffix in EXTENSIONS
        and file.name not in IGNORED_FILES         # ← исключаем по имени
        and file.resolve() != THIS_SCRIPT
        and not any(part.startswith('.') for part in file.parts if part != '.')
    )

Path(OUTPUT_FILE).write_text("", encoding="utf-8")

with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for name in sorted(files):
            file = Path(root, name)
            if not is_target(file):
                continue

            rel_path = file.relative_to(".")
            out.write(f"# {rel_path}\n\n")
            try:
                out.write(file.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                out.write(file.read_text(encoding="cp1251", errors="replace"))
            out.write("\n\n\n")





