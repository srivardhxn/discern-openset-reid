import requests
import json
import time

API = "http://127.0.0.1:8000"

print("--- 1. Health & Root ---")
r = requests.get(f"{API}/")
assert r.status_code == 200
print("Root:", r.json())

print("\n--- 2. Gallery Listing ---")
r = requests.get(f"{API}/gallery")
assert r.status_code == 200
gallery = r.json()
print(f"Gallery count: {len(gallery)}")
assert len(gallery) > 0

print("\n--- 3. Demo Scenario (3 consecutive runs) ---")
for i in range(1, 4):
    r = requests.post(f"{API}/demo/scenario")
    assert r.status_code == 200, f"Demo failed run {i}"
    d = r.json()
    steps = d.get("steps", [])
    print(f"Run {i}: {len(steps)} steps")
    for s in steps:
        step_num = s["step"]
        exp = s["expected"]
        discern_v = s["discern_verdict"]
        base_v = s["baseline_verdict"]
        prevented = s["is_false_accept_prevented"]
        print(f"  Step {step_num}: Expected={exp} | Discern={discern_v} | Baseline={base_v} | FalseAcceptPrevented={prevented}")
        assert discern_v == exp, f"Step {step_num} mismatch!"

print("\n--- 4. Negative / Bad Input Testing ---")
# Empty name
r = requests.post(f"{API}/enroll", data={"name": "   "}, files={"files": ("test.jpg", b"fake", "image/jpeg")})
assert r.status_code == 400, f"Expected 400, got {r.status_code}"
print("Empty name rejected:", r.status_code, r.json().get("detail"))

# Non-image file
r = requests.post(f"{API}/enroll", data={"name": "Test Non-Image"}, files={"files": ("doc.txt", b"hello text", "text/plain")})
assert r.status_code == 400, f"Expected 400, got {r.status_code}"
print("Non-image file rejected:", r.status_code, r.json().get("detail"))

# Corrupt image bytes
r = requests.post(f"{API}/enroll", data={"name": "Test Corrupt"}, files={"files": ("bad.jpg", b"not an image at all", "image/jpeg")})
assert r.status_code == 400, f"Expected 400, got {r.status_code}"
print("Corrupt image rejected:", r.status_code, r.json().get("detail"))

print("\n--- 5. Positive Enrollment & Deletion ---")
# Enroll a valid test identity with real market1501 images
with open("data/sample_market1501/bounding_box_test/0004_c1s1_000401_00.jpg", "rb") as f1, \
     open("data/sample_market1501/bounding_box_test/0004_c2s1_000402_00.jpg", "rb") as f2:
    sample_imgs = [
        ("files", ("img1.jpg", f1.read(), "image/jpeg")),
        ("files", ("img2.jpg", f2.read(), "image/jpeg"))
    ]
r = requests.post(f"{API}/enroll", data={"name": "Audited Staff Officer"}, files=sample_imgs)
assert r.status_code == 200, f"Enroll failed: {r.text}"
enrolled_info = r.json()
new_id = enrolled_info["identity"]["identity_id"]
print(f"Successfully enrolled identity: {enrolled_info['identity']['name']} (ID: {new_id})")

# Verify gallery increased
r = requests.get(f"{API}/gallery")
gallery_after = r.json()
assert len(gallery_after) == len(gallery) + 1
print(f"Gallery size updated from {len(gallery)} to {len(gallery_after)}")

# Duplicate name rejection
with open("data/sample_market1501/bounding_box_test/0004_c1s1_000401_00.jpg", "rb") as f1:
    sample_imgs_dup = [("files", ("img1.jpg", f1.read(), "image/jpeg"))]
r = requests.post(f"{API}/enroll", data={"name": "Audited Staff Officer"}, files=sample_imgs_dup)
assert r.status_code == 400
print("Duplicate name rejected as expected:", r.status_code, r.json().get("detail"))

# Now delete the enrolled test identity
r = requests.delete(f"{API}/identity/{new_id}")
assert r.status_code == 200, f"Delete failed: {r.text}"
del_res = r.json()
print(f"Successfully deleted identity {new_id}: remaining size = {del_res['remaining_gallery_size']}")

# Verify gallery reverted
r = requests.get(f"{API}/gallery")
gallery_final = r.json()
assert len(gallery_final) == len(gallery)
print("Gallery successfully reverted to original count:", len(gallery_final))

print("\n--- 6. Operating Point & Analysis Endpoints ---")
# Test FAR slider
r = requests.put(f"{API}/operating-point", json={"alpha": 0.005})
assert r.status_code == 200
print("Updated operating point alpha=0.005:", r.json())

# Reset to 0.01
r = requests.put(f"{API}/operating-point", json={"alpha": 0.01})
assert r.status_code == 200

# Other endpoints
for ep in ["metrics", "roc", "ablation", "lookalikes"]:
    r = requests.get(f"{API}/{ep}")
    assert r.status_code == 200, f"{ep} returned {r.status_code}"
    print(f"GET /{ep} OK: status {r.status_code}")

print("\n[ALL CHECKS PASSED PERFECTLY!]")
