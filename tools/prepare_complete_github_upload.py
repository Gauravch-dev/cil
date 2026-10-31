"""Inventory every workspace file and preserve nested Git metadata for upload."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "repository_file_inventory.json"
ARCHIVE_DIR = ROOT / "repository_metadata"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    originals = []
    nested = []
    for base, dirs, files in os.walk(ROOT):
        base = Path(base)
        if base == ROOT:
            dirs[:] = [d for d in dirs if d not in (".git", "repository_metadata")]
        for dirname in dirs:
            if dirname == ".git":
                nested.append(base / dirname)
        for filename in files:
            path = base / filename
            if path != INVENTORY:
                originals.append(path)

    archives = {}
    if nested:
        ARCHIVE_DIR.mkdir(exist_ok=True)
    for gitdir in nested:
        rel = gitdir.relative_to(ROOT).as_posix()
        archive = ARCHIVE_DIR / ("__".join(gitdir.parent.relative_to(ROOT).parts) + "__git_metadata.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in sorted(gitdir.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as check:
            assert check.testzip() is None, "Nested repository archive failed verification"
        archives[rel] = archive.relative_to(ROOT).as_posix()

    rows = []
    staged_paths = []
    for number, path in enumerate(sorted(originals), 1):
        rel = path.relative_to(ROOT).as_posix()
        archive = next((value for key, value in archives.items() if rel.startswith(key + "/")), None)
        row = {"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)}
        if archive:
            row["preserved_in_archive"] = archive
        else:
            staged_paths.append(rel)
        rows.append(row)
        if number % 10000 == 0:
            print(f"Inventoried and hashed {number} files", flush=True)

    large = [r["path"] for r in rows if r["bytes"] >= 50 * 1024**2 and "preserved_in_archive" not in r]
    attrs = ROOT / ".gitattributes"
    attrs.write_text("# Preserve large files with Git LFS.\n" + "".join(
        f'"{path}" filter=lfs diff=lfs merge=lfs -text\n' for path in large
    ), encoding="utf-8")
    document = {
        "description": "Complete file inventory. Nested .git files are preserved in ZIP archives because Git cannot track .git directories as ordinary files.",
        "file_count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "nested_git_archives": archives,
        "files": rows,
    }
    INVENTORY.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    staged_paths.extend([".gitattributes", INVENTORY.name, *archives.values()])
    pathspec = ROOT / ".git" / "complete-upload-paths"
    pathspec.write_bytes(b"".join(path.encode("utf-8") + b"\0" for path in sorted(set(staged_paths))))
    print(json.dumps({"inventoried_files": len(rows), "staged_files": len(set(staged_paths)), "bytes": document["total_bytes"], "lfs_files": large, "archives": archives}), flush=True)


if __name__ == "__main__":
    main()
