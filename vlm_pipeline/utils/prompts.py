"""Prompt templates for layout, OCR, table, and detect tasks."""

LAYOUT_PROMPT = """Analyze this document page layout.
Return ONLY valid JSON (no markdown fences) with this schema:
{
  "blocks": [
    {"label": "Title|Text|Table|Figure|List|Header|Footer", "bbox": [x1, y1, x2, y2], "reading_order": 1}
  ]
}
Use pixel coordinates from the image. Preserve reading order top-to-bottom."""

OCR_PROMPT = """OCR this document page.
Return clean reading-order text as HTML using <p> tags.
Preserve Hindi and English exactly as written. Do not summarize."""

TABLE_PROMPT = """Find all tables on this page.
Return ONLY valid JSON (no markdown fences):
{
  "tables": [
    {
      "bbox": [x1, y1, x2, y2],
      "rows": [["cell text", "..."]]
    }
  ]
}
If no table exists, return {"tables": []}."""

DETECT_PROMPT = """Read all visible text lines on this page.
Return ONLY valid JSON (no markdown fences):
{
  "text_lines": [
    {"text": "line text", "bbox": [x1, y1, x2, y2], "confidence": 0.95}
  ]
}
Use pixel coordinates."""

TASK_PROMPTS = {
    "layout": LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
    "table": TABLE_PROMPT,
    "detect": DETECT_PROMPT,
}
