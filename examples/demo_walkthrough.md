# Demo walkthrough

A 3-minute tour of Shelf Auditor MCP from an MCP client (Claude Desktop, Cline, …).

## 0. Setup

```bash
uv sync
uv run python scripts/download_models.py
```

Point your client at the server with `examples/claude_desktop_config.json`
(replace the path). Restart the client; you should see the `shelf-auditor`
tools appear.

## 1. Count products

> Prompt: **count-items** with a shelf photo, or just ask:
> "Count the products in this photo: `<path/URL/data-uri>`"

`count_products` returns `total`, per-label `counts`, every item's `bbox` /
`bbox_normalized` / `confidence`, and an annotated PNG (inline, or an
`annotated://{job_id}` reference for large images).

## 2. Find out-of-stock gaps

> "Where are the gaps on this shelf?"

`detect_gaps` clusters items into rows and reports each gap with a `row`,
a `width_ratio` (gap width ÷ mean product width in that row) and a `severity`.

## 3. Planogram check

Provide a spec — a list of slots with expected normalized regions:

```json
{
  "image": "<path>",
  "slots": [
    {"label": "cola-1.5L", "region": [0.00, 0.05, 0.18, 0.55], "order": 0},
    {"label": "cola-330ml", "region": [0.18, 0.05, 0.34, 0.55], "order": 1}
  ]
}
```

`check_planogram` lists `missing` / `misplaced` / `extra` / `wrong_order`
deviations and whether the shelf is `compliant`.

## 4. One-shot report

> "Give me a full shelf report for this photo."

`shelf_report` runs count + gaps (+ planogram if you pass `slots`) and returns
a combined JSON plus a markdown summary.

## Notes

- **Price tags:** `read_price_tags` is wired up but the OCR backend is a stub —
  it returns `status: "not_implemented"`. Implementing it (python-doctr) is the
  first post-MVP task.
- **Detector:** COCO-pretrained YOLO run class-agnostic — every box is
  `"product"`. Fine-tuning on SKU-110K gives per-SKU labels.
- **CPU:** with no CUDA, inference is seconds per image; the response `notes`
  say so.
