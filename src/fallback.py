from __future__ import annotations

from src.config import EXA_API_KEY

_exa = None


def _get_exa():
    global _exa
    if _exa is None:
        if not EXA_API_KEY:
            raise RuntimeError(
                "EXA_API_KEY not set. Add it to .env or skip fallback."
            )
        from exa_py import Exa

        _exa = Exa(api_key=EXA_API_KEY)
    return _exa


def web_search(query: str, num_results: int = 3) -> list[dict]:
    """Return a list of {title, url, text} snippets from Exa."""
    exa = _get_exa()
    results = exa.search_and_contents(
        query, num_results=num_results, text={"max_characters": 1000}
    )
    out = []
    for r in results.results:
        out.append(
            {
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", "") or "",
                "text": getattr(r, "text", "") or "",
            }
        )
    return out


def format_web_context(snippets: list[dict]) -> str:
    if not snippets:
        return ""
    blocks = []
    for i, s in enumerate(snippets, 1):
        blocks.append(
            f"[{i}] {s['title']}\n{s['url']}\n{s['text'][:800]}"
        )
    return "\n\n".join(blocks)
