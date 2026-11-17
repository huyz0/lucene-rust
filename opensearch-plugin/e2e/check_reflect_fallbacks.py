#!/usr/bin/env python3
"""Fails when the plugin node reports a fallback caused by a reflective read that found nothing.

The encoders read OpenSearch's and Lucene's private state by reflection; when a field or class
is gone (an OpenSearch bump moved it), the query is not refused: it falls back to OpenSearch's
own path with a `*_reflect` or `*_reflection` reason, results stay right, and the speed silently
drops to stock. After the e2e matrix every such counter in `GET _plugins/accelerated/stats`
must be 0.

  check_reflect_fallbacks.py http://localhost:9200
"""
import json
import sys
import urllib.request


def main() -> int:
    url = sys.argv[1].rstrip("/") + "/_plugins/accelerated/stats"
    with urllib.request.urlopen(url, timeout=60) as r:
        stats = json.load(r)
    fallbacks = stats.get("fallbacks", {})
    reflect = {k: v for k, v in fallbacks.items() if k.endswith(("_reflect", "_reflection")) and v}
    if reflect:
        print(f"check_reflect_fallbacks: reflective reads found nothing: {reflect}", file=sys.stderr)
        return 1
    print(f"check_reflect_fallbacks: no *_reflect fallbacks among {len(fallbacks)} fallback reasons seen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
