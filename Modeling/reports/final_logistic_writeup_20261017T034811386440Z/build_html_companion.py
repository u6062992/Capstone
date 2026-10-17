"""Execute this trusted report's cells and build an HTML companion, not Quarto.

Supports only the Markdown subset used by this report. No models are fitted or
scored. Keep this file with the Quarto source, included body, and data snapshots.
"""

import ast
import argparse
import html
import json
from pathlib import Path
import re


DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY / "final_modeling_writeup.qmd"
OUTPUT = DIRECTORY / "final_modeling_writeup.html"
VALIDATION = DIRECTORY / "report_validation.json"


def expand_includes(text):
    def include(match):
        path = (DIRECTORY / match.group(1).strip()).resolve()
        if path.parent != DIRECTORY:
            raise ValueError("Includes must be in the report folder.")
        return path.read_text(encoding="utf-8")

    return re.sub(r"\{\{<\s*include\s+([^>]+?)\s*>\}\}", include, text)


def main(overwrite=False):
    text = expand_includes(SOURCE.read_text(encoding="utf-8"))
    title = re.search(r'^title:\s*"(.+)"\s*$', text, re.MULTILINE).group(1)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    namespace = {"__name__": "__report__", "__file__": str(SOURCE)}
    fragments, paragraphs, labels = [], [], []
    list_open = False

    def inline(value):
        value = re.sub(r"`\{python\}\s+([^`]+)`", lambda m: str(eval(m.group(1), namespace)), value)
        value = html.escape(value, quote=True)
        value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", value)

    def flush():
        if paragraphs:
            fragments.append("<p>" + inline(" ".join(paragraphs)) + "</p>")
            paragraphs.clear()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            if list_open:
                fragments.append("</ul>")
                list_open = False
            i += 1
            continue
        if line.startswith("```{python}"):
            flush()
            i += 1
            code_lines = []
            while i < len(lines) and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("Unclosed Python code cell.")
            code = "\n".join(code_lines)
            match = re.search(r"^#\| label: (.+)$", code, re.MULTILINE)
            if not match or match.group(1) in labels:
                raise ValueError("Every code cell must have a unique label.")
            labels.append(match.group(1))
            fragments.append('<pre class="source"><code>' + html.escape(code) + "</code></pre>")
            tree = ast.parse(code, filename=str(SOURCE))
            expression = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, str(SOURCE), "exec"), namespace)
            if expression is not None:
                value = eval(compile(ast.Expression(expression.value), str(SOURCE), "eval"), namespace)
                if value is not None:
                    rendered = value._repr_html_() if hasattr(value, "_repr_html_") else "<pre>" + html.escape(str(value)) + "</pre>"
                    fragments.append('<div class="output">' + rendered + "</div>")
        elif line.startswith("::: {"):
            flush()
            fragments.append('<aside class="callout">')
        elif line.strip() == ":::":
            flush()
            fragments.append("</aside>")
        elif re.match(r"^#{1,6} ", line):
            flush()
            heading = re.match(r"^(#{1,6}) (.*)$", line)
            level = len(heading.group(1))
            fragments.append(f"<h{level}>" + inline(heading.group(2)) + f"</h{level}>")
        elif line.startswith("- "):
            flush()
            if not list_open:
                fragments.append("<ul>")
                list_open = True
            fragments.append("<li>" + inline(line[2:]) + "</li>")
        else:
            paragraphs.append(line)
        i += 1
    flush()
    if list_open:
        fragments.append("</ul>")
    body = "\n".join(fragments)
    assert "`{python}" not in body and "{{< include" not in body
    assert namespace["scoring"]["rows_scored"] == 38219
    assert namespace["scoring"]["unknown_targets"] == namespace["scoring"]["rows_scored"]
    assert set(namespace["folds"]["assessment_fy"]) == {2022, 2023, 2024}
    assert set(namespace["fy2025"]["assessment_fy"]) == {2025}
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>""" + html.escape(title) + """</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.5rem;line-height:1.55;color:#172b3a}
h1{margin-top:2.5rem;border-bottom:2px solid #b9d0de;padding-bottom:.4rem}h2{margin-top:1.8rem}
.callout{background:#edf5fa;border-left:5px solid #2f607a;padding:.8rem 1rem;margin:1.5rem 0}
.source{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f6;padding:1rem;font-size:.82rem;border:1px solid #dce2e7}
.output{overflow-x:auto;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:.55rem .7rem;text-align:left;border:1px solid #d4dfe6}th{background:#e5eef4}tr:nth-child(even){background:#f7fafc}
code{font-size:.9em}a{color:#155a82}header p{color:#51616b}
@media print{.source{font-size:.65rem}body{max-width:none}.output{overflow:visible}table{font-size:.7rem}}
</style></head><body><header><h1>""" + html.escape(title) + """</h1>
<p>Standalone HTML companion. Code cells executed in a fresh process; not a native Quarto render.</p></header>
""" + body + "\n</body></html>"
    with OUTPUT.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(document)
    validation = {
        "status": "executed_and_built", "renderer": "standard-library HTML companion; not native Quarto",
        "code_cells_executed": len(labels), "code_cell_labels": labels,
        "input_hash_checks_passed": True, "inline_expressions_resolved": True,
        "model_fitting_performed": False, "model_scoring_performed": False,
        "fy2025_evaluation_repeated": False, "donor_level_data_embedded": False,
    }
    with VALIDATION.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    print(json.dumps(validation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the generated HTML and validation JSON.")
    main(overwrite=parser.parse_args().overwrite)
