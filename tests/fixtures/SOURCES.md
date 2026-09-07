# Test fixture sources

Every file in `tests/fixtures/` must be listed here with its source and license.
The `.jpg` files are **not committed** — fetch them with
`uv run python tests/fixtures/download_fixtures.py` (see `manifest.json`).

| File | Source | License | Author | Use |
|------|--------|---------|--------|-----|
| `wine_shelf.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Alsatian_wines_in_a_supermarket.jpg) | CC BY 2.0 | francois (Flickr) | many wine bottles → COCO `bottle` class, numeric assertion |
| `bread_shelf.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2019-08-01_Krustenbrot_for_sale_at_supermarket.jpg) | CC BY-SA 4.0 | Maksym Kozlenko | full shelf of packaged goods (COCO detects ~none — motivates SKU fine-tune) |
| `empty_shelf.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Covid-19_pandemic_food_store_Lordship_Lane_Tottenham,_London,_England_2.jpg) | CC BY-SA 4.0 | Acabashi | visible out-of-stock gaps |

Synthetic images used by the core suite are generated in-test (`tests/conftest.py`)
and need no source entry.

## Guidance

- **SKU-110K** images (exact ground-truth box counts — best for numeric asserts):
  academic / non-commercial. Add them to `manifest.json` with an `expected_count`
  field; `test_fixtures.py` will assert against it. **Do not** commit the raw
  images or put them in the public README / demo.
- **Public-facing demo images** must be self-shot or CC0 / CC-BY from
  Unsplash / Pexels / Wikimedia (the three above are CC-BY, safe to show with
  attribution).
- Self-shot photos still worth adding: readable price tags, a clearly misplaced
  product, tilted / low-light.
