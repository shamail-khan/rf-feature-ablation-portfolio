"""Check this code-only draft without reading any research dataset."""
import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "release_allowlist.json").read_text(encoding="utf-8"))
allowed = set(manifest["files"])
actual = set()
errors = []
for p in ROOT.rglob("*"):
    rel = p.relative_to(ROOT)
    if rel.parts[0] == ".git":
        continue
    if p.is_symlink():
        errors.append(f"Symlink is not allowed: {rel.as_posix()}")
        continue
    if p.is_file():
        actual.add(rel.as_posix())
for name in sorted(actual - allowed):
    errors.append(f"Unexpected file: {name}")
for name in sorted(allowed - actual):
    errors.append(f"Missing release file: {name}")
patterns = [
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    r"(?i)(?:password|api_key|access_token)\s*=\s*['\"][^'\"]{8,}['\"]",
]
parsed = 0
for name in sorted(actual & allowed):
    p = ROOT / name
    if p.stat().st_size > 200_000:
        errors.append(f"File exceeds code-only review size limit: {name}")
        continue
    try:
        content = p.read_text(encoding="utf-8-sig")
    except UnicodeError:
        errors.append(f"Non-text file: {name}")
        continue
    if "\x00" in content:
        errors.append(f"Binary marker: {name}")
    if any(re.search(pattern, content) for pattern in patterns):
        errors.append(f"Possible credential: {name}")
    if name.endswith(".py"):
        try:
            ast.parse(content, filename=name)
            parsed += 1
        except SyntaxError as exc:
            errors.append(f"Syntax error: {name}:{exc.lineno}: {exc.msg}")
        # Catch original machine locations without printing matched content.
        private_roots = [chr(68) + ":" + chr(92) + "sionna_grid_task", "/".join(["", "data", "beegfs", ""]), "/".join(["", "home", "mkhan", ""])]
        if any(value in content for value in private_roots):
            errors.append(f"Original private path: {name}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print(json.dumps({"status": "passed", "files": len(actual), "python_files_parsed": parsed,
                  "data_files": 0, "scope": "allowlist, text, size, syntax, credential patterns, known paths"}, indent=2))
