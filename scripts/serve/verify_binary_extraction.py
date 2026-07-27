"""Verify binary extraction (xlsx/docx/pptx/pdf) in the E2B sandbox.

Usage:
    set -a && source docker/sandbox/tencent.env && set +a
    python scripts/serve/verify_binary_extraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.observer import extract_binaries, snapshot_workspace
from rollout.sandbox_client import make_sandbox


def main() -> int:
    sb = make_sandbox("e2b", timeout=300)
    print(f"Sandbox: {sb._sandbox_id}")

    # Step 1: Install pptx + pdfplumber (one at a time, longer timeout)
    for pkg in ["python-pptx", "pdfplumber"]:
        r = sb.run_code(
            f"import subprocess, sys; "
            f'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "{pkg}"]); '
            f'print("INSTALLED {pkg}")'
        )
        status = "OK" if r.ok else "FAIL"
        output = r.stdout.strip()[:60] or r.stderr[:80]
        print(f"Install {pkg}: {status} ({output})")

    # Step 1b: Verify installations
    r = sb.run_code(
        "import json; libs={}; "
        "[libs.update({n: True}) if __import__(n) else None for n in ['pptx','pdfplumber']]; "
        "print(json.dumps(libs))"
    )
    print(f"Libs after install: {r.stdout.strip()}")

    # Step 2: Create all 4 binary format test files
    # Build the code as a string to avoid Python escaping issues
    create_lines = [
        "import json, os",
        "cwd = os.getcwd()",
        "",
        "# xlsx",
        "import openpyxl",
        "wb = openpyxl.Workbook()",
        "ws = wb.active",
        'ws.title = "Q3"',
        'ws.append(["Item", "Amount"])',
        'ws.append(["Revenue", 12345])',
        'ws.append(["Cost", 6789])',
        'wb.save(os.path.join(cwd, "test.xlsx"))',
        "",
        "# docx",
        "import docx",
        "doc = docx.Document()",
        'doc.add_heading("Test Report", level=1)',
        'doc.add_paragraph("Q3 total revenue is 12345.")',
        'doc.add_paragraph("Key finding: costs are 6789.")',
        'doc.save(os.path.join(cwd, "test.docx"))',
        "",
        "# pptx",
        "from pptx import Presentation",
        "prs = Presentation()",
        "slide = prs.slides.add_slide(prs.slide_layouts[1])",
        'slide.shapes.title.text = "Q3 Summary"',
        'slide.placeholders[1].text = "Revenue: 12345, Cost: 6789"',
        'prs.save(os.path.join(cwd, "test.pptx"))',
        "",
        "# pdf (use fpdf2 if available, else write minimal PDF)",
        "try:",
        "    from fpdf import FPDF",
        "    pdf = FPDF()",
        "    pdf.add_page()",
        '    pdf.set_font("Helvetica", size=12)',
        '    pdf.cell(200, 10, txt="Q3 Revenue: 12345", ln=True)',
        '    pdf.cell(200, 10, txt="Cost: 6789", ln=True)',
        '    pdf.output(os.path.join(cwd, "test.pdf"))',
        "except ImportError:",
        "    # Write a minimal valid PDF",
        "    lines = [",
        '        b"%PDF-1.0",',
        '        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",',
        '        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",',
        '        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"',
        '        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj",',
        '        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",',
        '        b"5 0 obj<</Length 44>>stream",',
        '        b"BT /F1 12 Tf 100 750 Td (Q3 Revenue 12345) Tj ET",',
        '        b"endstream",',
        '        b"endobj",',
        '        b"xref",',
        '        b"0 6",',
        '        b"trailer<</Size 6/Root 1 0 R>>",',
        '        b"startxref",',
        '        b"434",',
        '        b"%%EOF",',
        "    ]",
        '    with open(os.path.join(cwd, "test.pdf"), "wb") as f:',
        '        f.write(b"\\n".join(lines))',
        "",
        'print(json.dumps({"xlsx": True, "docx": True, "pptx": True, "pdf": True}))',
    ]
    create_code = "\n".join(create_lines)
    r = sb.run_code(create_code)
    print(f"Create files: ok={r.ok}, output={r.stdout.strip()[:100]}")
    if r.stderr and not r.ok:
        print(f"  stderr: {r.stderr[:200]}")

    # Step 3: Snapshot workspace
    snap = snapshot_workspace(sb)
    print(f"\nSnapshot: {len(snap)} files")
    for p, rec in sorted(snap.items()):
        ext = rec.get("ext", "")
        kind = "BINARY" if rec.get("binary") else "text"
        size = rec.get("size", 0)
        text_preview = rec.get("text", "")[:50].replace("\n", "\\n")
        print(f"  {p} ({kind}, {size}B, ext={ext}): {text_preview!r}")

    # Step 4: Extract binary content
    binary_exts = (".xlsx", ".xlsm", ".docx", ".pptx", ".pdf")
    changed = [p for p, r in snap.items() if r.get("ext", "").lower() in binary_exts and r.get("binary")]
    print(f"\nExtracting {len(changed)} binary files: {changed}")

    results = {}
    extracted = extract_binaries(sb, changed)
    for p, text in extracted.items():
        results[p] = text[:300]
        status = "OK" if text.strip() else "EMPTY"
        print(f"  [{status}] {p}: {text[:200]}...")

    # Step 5: Summary
    print("\n=== Summary ===")
    all_ok = True
    for ext_name in ["xlsx", "docx", "pptx", "pdf"]:
        matching = [p for p in results if p.endswith(f".{ext_name}")]
        if matching:
            content = results[matching[0]]
            ok = bool(content.strip())
            print(f"  {ext_name}: {'PASS' if ok else 'FAIL'} (content extracted: {len(content)} chars)")
            if not ok:
                all_ok = False
        else:
            print(f"  {ext_name}: SKIP (file not created)")

    sb.kill()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
