"""Fetch the real shelf photos listed in ``manifest.json``.

These are CC-licensed Wikimedia Commons images used by the integration tests
(``tests/test_fixtures.py``), which skip themselves when the files are absent.
The images are not committed — run this once locally / in CI.

    uv run python tests/fixtures/download_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
_API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent with a contact URL.
_UA = "shelf-auditor-mcp/0.1 (https://github.com/; tests) httpx"


def _thumb_url(client: httpx.Client, wikimedia_name: str) -> str:
    resp = client.get(
        _API,
        params={
            "action": "query",
            "titles": f"File:{wikimedia_name}",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "1600",
            "format": "json",
        },
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    info = next(iter(pages.values()))["imageinfo"][0]
    return info.get("thumburl") or info["url"]


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
    ok = True
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": _UA}) as client:
        for item in manifest["fixtures"]:
            dest = HERE / item["file"]
            if dest.is_file():
                print(f"have   {item['file']}")
                continue
            try:
                url = _thumb_url(client, item["wikimedia"])
                resp = client.get(url)
                resp.raise_for_status()
            except (httpx.HTTPError, KeyError, StopIteration) as exc:
                print(f"FAIL   {item['file']}: {exc}", file=sys.stderr)
                ok = False
                continue
            dest.write_bytes(resp.content)
            print(f"saved  {item['file']}  ({len(resp.content) // 1024} KB)  [{item['license']}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
