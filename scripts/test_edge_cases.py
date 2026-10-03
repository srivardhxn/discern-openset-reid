import requests
import base64
import io
from PIL import Image
import numpy as np

API = "http://127.0.0.1:8000"

print("==================================================")
print("  DISCERN EDGE CASE & ENTERPRISE ROBUSTNESS SUITE ")
print("==================================================")

# 1. Base64 payload matching
print("\n[Edge 1] Base64 Image Matching via POST /match ...")
img = Image.open("data/sample_market1501/query/0026_c4s1_002604_00.jpg")
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

r = requests.post(f"{API}/match", json={"image_base64": b64_str})
assert r.status_code == 200, f"Base64 match failed: {r.text}"
res = r.json()
assert res["decision"] == "ACCEPTED"
print("[OK] Base64 match succeeded:", res["decision"], res["predicted_name"], "confidence:", res["calibrated_confidence"])

# 2. Extreme Tiny & Large Image Dimensions
print("\n[Edge 2] Tiny Image (16x16) & High-Resolution Image (1600x1200) ...")
tiny_img = Image.new("RGB", (16, 16), color=(120, 80, 50))
tiny_buf = io.BytesIO()
tiny_img.save(tiny_buf, format="JPEG")
r = requests.post(f"{API}/match", files={"file": ("tiny.jpg", tiny_buf.getvalue(), "image/jpeg")})
assert r.status_code == 200, f"Tiny image failed: {r.text}"
print("[OK] Tiny 16x16 handled smoothly:", r.json()["decision"], r.json()["reason_code"])

large_img = Image.new("RGB", (1600, 1200), color=(50, 100, 150))
large_buf = io.BytesIO()
large_img.save(large_buf, format="JPEG")
r = requests.post(f"{API}/match", files={"file": ("large.jpg", large_buf.getvalue(), "image/jpeg")})
assert r.status_code == 200, f"Large image failed: {r.text}"
print("[OK] Large 1600x1200 handled smoothly:", r.json()["decision"], r.json()["reason_code"])

# 3. Pure Black / Solid White (Zero-Variance Images)
print("\n[Edge 3] Pure Zero-Variance (Solid Black & Solid White) ...")
black_img = Image.new("RGB", (128, 256), color=(0, 0, 0))
buf = io.BytesIO()
black_img.save(buf, format="JPEG")
r = requests.post(f"{API}/match", files={"file": ("black.jpg", buf.getvalue(), "image/jpeg")})
assert r.status_code == 400
print("[OK] Solid Black cleanly rejected for zero variance:", r.json()["detail"])

white_img = Image.new("RGB", (128, 256), color=(255, 255, 255))
buf = io.BytesIO()
white_img.save(buf, format="JPEG")
r = requests.post(f"{API}/match", files={"file": ("white.jpg", buf.getvalue(), "image/jpeg")})
assert r.status_code == 400
print("[OK] Solid White cleanly rejected for zero variance:", r.json()["detail"])

# 4. Non-Existent Identity Deletion
print("\n[Edge 4] Deleting Non-Existent Identity (ID 999999) ...")
r = requests.delete(f"{API}/identity/999999")
assert r.status_code == 404, f"Expected 404, got {r.status_code}"
print("[OK] Correctly returned 404:", r.json()["detail"])

# 5. Case-Insensitive Duplicate Name Rejection
print("\n[Edge 5] Case-Insensitive Duplicate Name Check ...")
gal_resp = requests.get(f"{API}/gallery")
assert gal_resp.status_code == 200 and len(gal_resp.json()) > 0
existing_gal_name = gal_resp.json()[0]["name"]
r = requests.post(f"{API}/enroll", data={"name": existing_gal_name.lower()}, files={"files": ("test.jpg", tiny_buf.getvalue(), "image/jpeg")})
assert r.status_code == 400
print(f"[OK] Correctly rejected duplicate name '{existing_gal_name.lower()}' vs '{existing_gal_name}':", r.json()["detail"])

# 6. Extreme Operating Point Alpha Boundaries
print("\n[Edge 6] Extreme Operating Point Alpha Boundaries (0.0001 and 0.50) ...")
r = requests.put(f"{API}/operating-point", json={"alpha": 0.0001})
assert r.status_code == 200
print("[OK] Ultra-strict alpha 0.0001 (0.01% FAR): tau =", r.json()["operating_point"]["threshold_tau"], "delta =", r.json()["operating_point"]["margin_delta"])

r = requests.put(f"{API}/operating-point", json={"alpha": 0.50})
assert r.status_code == 200
print("[OK] Permissive alpha 0.50 (50% FAR): tau =", r.json()["operating_point"]["threshold_tau"], "delta =", r.json()["operating_point"]["margin_delta"])

# Reset to 0.01
requests.put(f"{API}/operating-point", json={"alpha": 0.01})

# 7. Oversized file (> 10MB)
print("\n[Edge 7] Oversized File Upload (> 10MB) ...")
huge_bytes = b"0" * (11 * 1024 * 1024)
r = requests.post(f"{API}/enroll", data={"name": "Oversized User"}, files={"files": ("huge.jpg", huge_bytes, "image/jpeg")})
assert r.status_code == 400
print("[OK] Oversized payload cleanly rejected:", r.json()["detail"])

print("\n==================================================")
print("  ALL 7 ADVANCED EDGE CASES VERIFIED & PASSED!    ")
print("==================================================")
