import io
import json

import pytest

from app.ingestion.chunking import FixedWindowChunker, StructureChunker
from app.ingestion.dedup import LSHIndex, content_hash, estimated_jaccard, minhash_signature
from app.ingestion.loaders import DocumentParseError, UnsupportedFormatError, load_file, parse_frontmatter
from app.ingestion.metadata import build_metadata, extract_dates, extract_entities

# ------------------------------------------------------------------ loaders


def test_markdown_frontmatter_and_title():
    data = b"---\ntitle: Incident INC-1\ndoc_type: incident_report\ndate: 2026-03-12\n---\n\n# Heading\n\nBody text."
    loaded = load_file("inc.md", data)
    assert loaded.title == "Incident INC-1"
    assert loaded.metadata["doc_type"] == "incident_report"
    assert "Body text." in loaded.text and "---" not in loaded.text


def test_html_strips_scripts_and_keeps_headings():
    html = b"<html><head><title>T</title><meta name='date' content='2026-01-01'></head><body><nav>menu</nav>" \
           b"<script>alert(1)</script><h2>Section</h2><p>Hello <b>world</b></p><table><tr><td>a</td><td>b</td></tr></table></body></html>"
    loaded = load_file("page.html", html)
    assert "alert" not in loaded.text and "menu" not in loaded.text
    assert "## Section" in loaded.text and "Hello world" in loaded.text and "a | b" in loaded.text
    assert loaded.metadata["date"] == "2026-01-01"


def test_csv_rows_become_records():
    loaded = load_file("log.csv", b"id,service\nCHG-1,api-gateway\nCHG-2,wallet-service\n")
    assert loaded.text.split("\n\n") == ["id: CHG-1; service: api-gateway", "id: CHG-2; service: wallet-service"]
    assert loaded.metadata["record_based"] is True


def test_json_chat_export():
    data = json.dumps({"channel": "#inc", "messages": [{"ts": "10:00", "user": "ana", "text": "hello"}]}).encode()
    loaded = load_file("slack.json", data)
    assert loaded.title == "#inc"
    assert loaded.text == "[10:00] ana: hello"


def test_docx_headings_and_properties():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.core_properties.title = "Runbook"
    document.core_properties.category = "runbook"
    document.add_heading("Alerts", level=1)
    document.add_paragraph("Pool saturation alert.")
    buf = io.BytesIO()
    document.save(buf)
    loaded = load_file("rb.docx", buf.getvalue())
    assert loaded.title == "Runbook" and loaded.metadata["doc_type"] == "runbook"
    assert "# Alerts" in loaded.text


def test_pdf_roundtrip():
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, "1. Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 6, "The incident lasted 107 minutes and affected payment-gateway.", new_x="LMARGIN", new_y="NEXT")
    loaded = load_file("pm.pdf", bytes(pdf.output()))
    assert "107 minutes" in loaded.text and "## 1. Summary" in loaded.text


def test_unsupported_and_empty_files():
    with pytest.raises(UnsupportedFormatError):
        load_file("virus.exe", b"MZ")
    with pytest.raises(DocumentParseError):
        load_file("empty.txt", b"   \n  ")
    with pytest.raises(DocumentParseError):
        load_file("bad.json", b"{not json")


def test_parse_frontmatter_absent():
    meta, body = parse_frontmatter("no front matter")
    assert meta == {} and body == "no front matter"


# ------------------------------------------------------------------ metadata


def test_entities_and_dates():
    text = "CHG-1042 changed payment-gateway on 2026-03-09; INC-2041 began March 12, 2026 and 14 March 2026."
    assert extract_entities(text) == ["CHG-1042", "INC-2041", "payment-gateway"]
    assert extract_dates(text) == ["2026-03-09", "2026-03-12", "2026-03-14"]
    meta = build_metadata("PM-2041_postmortem.pdf", text, {})
    assert meta["doc_type"] == "postmortem" and meta["date"] == "2026-03-09"


# ------------------------------------------------------------------ chunking


def test_fixed_window_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    spans = FixedWindowChunker(max_words=100, overlap_words=20).split(text)
    assert [len(s.text.split()) for s in spans] == [100, 100, 90]
    assert spans[0].text.split()[-20:] == spans[1].text.split()[:20]


def test_structure_chunker_respects_sections():
    text = "## A\n\n" + "alpha " * 90 + "\n\n## B\n\n" + "beta " * 90
    spans = StructureChunker(max_words=200).split(text)
    assert [s.section for s in spans] == ["A", "B"]
    assert "beta" not in spans[0].text


def test_structure_chunker_merges_tiny_sections():
    text = "## Overview\n\nShort overview.\n\n## Escalation\n\nPage the on-call engineer."
    spans = StructureChunker(max_words=200).split(text)
    assert len(spans) == 1
    assert spans[0].section == "Overview" and "Escalation:" in spans[0].text


def test_structure_chunker_splits_long_paragraph_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))  # 300 words
    spans = StructureChunker(max_words=100).split(para)
    assert len(spans) == 3
    assert all(s.text.endswith(".") for s in spans)


def test_structure_chunker_records_are_never_split():
    records = "\n\n".join(f"id: CHG-{i}; service: s{i}; description: change number {i}" for i in range(30))
    spans = StructureChunker(max_words=50).split(records, record_based=True)
    for s in spans:
        for line in s.text.split("\n"):
            assert line.startswith("id: CHG-") and line.endswith(f"change number {line.split('CHG-')[1].split(';')[0]}")


# ------------------------------------------------------------------ dedup


def test_content_hash_ignores_formatting():
    assert content_hash("Hello,   World!\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_minhash_estimates_jaccard():
    base = " ".join(f"word{i}" for i in range(300))
    near = base.replace("word150", "changed")
    other = " ".join(f"other{i}" for i in range(300))
    s_base, s_near, s_other = minhash_signature(base), minhash_signature(near), minhash_signature(other)
    assert estimated_jaccard(s_base, s_near) > 0.9
    assert estimated_jaccard(s_base, s_other) < 0.1
    index = LSHIndex(threshold=0.8)
    index.add("base", s_base)
    assert [d for d, _ in index.query(s_near)] == ["base"]
    assert index.query(s_other) == []
