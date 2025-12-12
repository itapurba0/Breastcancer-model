#!/usr/bin/env python3
"""Send the same image to the backend /predict endpoint concurrently.

Usage:
  python run_predict_concurrent.py --file "/path/to/img.png" --url http://localhost:8001/predict \
    --count 100 --concurrency 10 --out results.json

The script uses a ThreadPoolExecutor and `requests` so it doesn't require
any extra async libraries.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from typing import Dict, Any

import requests
import mimetypes


def send_request(url: str, file_path: str, timeout: float = 10.0) -> Dict[str, Any]:
    try:
        with open(file_path, "rb") as fh:
            # try to set a sensible content-type so FastAPI's content check passes
            mime, _ = mimetypes.guess_type(file_path)
            if not mime:
                # default to png if unknown (most dataset images are png/jpg)
                mime = "image/png"
            files = {"file": (os.path.basename(file_path), fh, mime)}
            r = requests.post(url, files=files, timeout=timeout)
            # try to parse json safely
            try:
                body = r.json()
            except Exception:
                body = r.text
            return {"status_code": r.status_code, "body": body}
    except Exception as e:
        return {"error": str(e)}


def run_parallel(url: str, file_path: str, count: int, concurrency: int, timeout: float):
    results = []
    start = time.time()
    with requests.Session() as session:
        # We'll still use our send_request helper which opens the file per request.
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(send_request, url, file_path, timeout) for _ in range(count)]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                res = fut.result()
                results.append(res)
                # quick progress print
                if i % max(1, count // 20) == 0 or i == count:
                    print(f"Completed {i}/{count}")

    dur = time.time() - start
    print(f"All requests finished in {dur:.2f}s (avg {(dur/count):.3f}s per request)")
    return results


def summarize(results: list[Dict[str, Any]]):
    totals: Dict[str, int] = {}
    errors = 0
    ok = 0
    for r in results:
        if "error" in r:
            errors += 1
            totals.setdefault("error", 0)
            totals["error"] += 1
        else:
            code = str(r.get("status_code", "-"))
            totals.setdefault(code, 0)
            totals[code] += 1
            if r.get("status_code") == 200:
                ok += 1

    print("Summary:")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")
    print(f"  Success (200): {ok}")
    print(f"  Errors: {errors}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", required=False,
                   default=os.path.join(os.getcwd(), "classification_model", "data_prepared", "test", "malignant", "malignant (7).png"),
                   help="/home/apurba-roy/Developments/FinalYear-Project/Breastcancer model/backend/classification_model/data_prepared/test/malignant/malignant (7).png")
    p.add_argument("--url", "-u", default="http://localhost:8001/predict", help="Predict endpoint URL")
    p.add_argument("--count", "-n", type=int, default=100, help="Number of requests to send")
    p.add_argument("--concurrency", "-c", type=int, default=10, help="Number of concurrent workers")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds")
    p.add_argument("--out", "-o", help="Optional JSON file to write all results to")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print("File not found:", args.file)
        raise SystemExit(1)

    print(f"Sending {args.count} requests to {args.url} with concurrency={args.concurrency}")
    results = run_parallel(args.url, args.file, args.count, args.concurrency, args.timeout)
    summarize(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print("Wrote results to", args.out)


if __name__ == "__main__":
    main()
