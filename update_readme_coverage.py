import json, re, pathlib, sys

cov_file = pathlib.Path("coverage.json")
if not cov_file.exists():
    print("coverage.json not found — run pytest with --cov=app --cov-report=json first")
    sys.exit(1)

data = json.loads(cov_file.read_text())

rows = []
for path, info in sorted(data["files"].items()):
    s = info["summary"]
    module = path.replace("\\", "/")
    rows.append(f"| `{module}` | {s['num_statements']} | {s['missing_lines']} | {s['percent_covered_display']} |")

t = data["totals"]
table = (
    "| Module | Stmts | Miss | Cover |\n"
    "|--------|------:|-----:|------:|\n"
    + "\n".join(rows)
    + f"\n| **TOTAL** | **{t['num_statements']}** | **{t['missing_lines']}** | **{t['percent_covered_display']}** |"
)

readme = pathlib.Path("README.md")
content = readme.read_text(encoding="utf-8")
updated = re.sub(
    r"<!-- COVERAGE_START -->.*?<!-- COVERAGE_END -->",
    f"<!-- COVERAGE_START -->\n{table}\n<!-- COVERAGE_END -->",
    content,
    flags=re.DOTALL,
)

readme.write_text(updated, encoding="utf-8")
print("README.md updated")
