#!/usr/bin/env python3
"""
smoke_test.py — Discern ONNX backend smoke test.

Loads the demo gallery and runs all demo probes through the model,
then prints known-accepted and look-alike-false-accept counts.

Usage:
    python smoke_test.py
    python smoke_test.py --host http://127.0.0.1:8000
"""
import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Discern smoke test")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    base = args.host.rstrip("/")

    print(f"[smoke_test] Connecting to {base} …")

    # 1. Check server health
    try:
        r = requests.get(f"{base}/", timeout=10)
        r.raise_for_status()
        info = r.json()
        print(f"[smoke_test] Server online: {info.get('model')} "
              f"(providers: {info.get('providers')})")
    except Exception as e:
        print(f"[smoke_test] FAIL: Cannot reach server at {base}. Start with: python -m uvicorn backend.app.main:app --port 8000")
        print(f"             Error: {e}")
        sys.exit(1)

    # 2. Load demo gallery
    print("[smoke_test] Loading demo gallery …")
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/demo/load", timeout=120)
    if not r.ok:
        print(f"[smoke_test] FAIL: demo load: {r.status_code} {r.text}")
        sys.exit(1)
    demo = r.json()
    print(f"[smoke_test] Demo loaded via '{demo['method']}': "
          f"{demo['enrolled_identities']} identities, "
          f"{demo['gallery_entries']} gallery / {demo['probe_entries']} probe entries "
          f"({time.perf_counter()-t0:.1f}s)")

    # 3. Run smoke test
    print("[smoke_test] Running all probe images …")
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/demo/smoke-test", timeout=300)
    if not r.ok:
        print(f"[smoke_test] FAIL: smoke-test: {r.status_code} {r.text}")
        sys.exit(1)
    result = r.json()
    elapsed = time.perf_counter() - t0

    ok   = result["known_accepted"]
    tot  = result["known_total"]
    fa   = result["lookalike_false_accepts"]
    unk  = result["lookalike_total"]

    print()
    print("=" * 60)
    print(f"  known probes correctly accepted : {ok}/{tot}  "
          f"({ok/tot*100:.1f}% TAR)" if tot > 0 else "  No enrolled probes found.")
    print(f"  look-alike strangers falsely accepted : {fa}/{unk}  "
          f"({fa/unk*100:.2f}% FAR)" if unk > 0 else "  No unknown probes found.")
    print(f"  total probe time : {elapsed:.1f}s  ({elapsed/max(tot+unk,1)*1000:.0f} ms/probe)")
    print("=" * 60)
    print()

    # Optional: show per-probe details for failures
    failures = [
        p for p in result["details"]
        if p["truth"] == "enrolled" and p["identity"] != p["expected"]
    ]
    if failures:
        print(f"[smoke_test] {len(failures)} enrolled probe failure(s):")
        for p in failures[:5]:
            print(f"  {p['file']}  expected={p['expected']}  got={p['identity']}  conf={p['confidence']:.3f}")
        if len(failures) > 5:
            print(f"  … and {len(failures)-5} more")
        print()

    fas = [p for p in result["details"] if p["truth"] != "enrolled" and p["decision"] == "accept"]
    if fas:
        print(f"[smoke_test] {len(fas)} look-alike false accept(s):")
        for p in fas[:5]:
            print(f"  {p['file']}  identity_returned={p['identity']}  conf={p['confidence']:.3f}")
        if len(fas) > 5:
            print(f"  … and {len(fas)-5} more")
        print()

    if ok == tot and fa == 0:
        print("[smoke_test] PASS — all enrolled accepted, zero false accepts.")
    else:
        print("[smoke_test] Results above (not necessarily a failure — depends on operating point).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
