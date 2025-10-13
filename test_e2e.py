#!/usr/bin/env python3
"""
End-to-end test for Face Recognition API using local sample data.
"""
import requests
from pathlib import Path
import time

BASE = "http://localhost:9001/api"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CLASS_ID = "E2E_TEST"


def post(url, **kwargs):
    r = requests.post(url, timeout=30, **kwargs)
    print("POST", url, r.status_code)
    if r.text:
        print(r.text[:500])
    return r


def get(url, **kwargs):
    r = requests.get(url, timeout=30, **kwargs)
    print("GET", url, r.status_code)
    return r


def delete(url, **kwargs):
    r = requests.delete(url, timeout=30, **kwargs)
    print("DELETE", url, r.status_code)
    if r.text:
        print(r.text[:500])
    return r


def main():
    # 1) Create class
    r = post(f"{BASE}/classes", json={"class_id": CLASS_ID})
    assert r.status_code == 200 or "already exists" in r.text

    # 2) Register two students with local images
    samples = [
        ("trump", [
            DATA_DIR / "trump/Donald_Trump_official_portrait.jpg",
            DATA_DIR / "trump/test.jpg",
            DATA_DIR / "trump/images.jpeg",
        ]),
        ("elon", [
            DATA_DIR / "musk/Elon_Musk_(54816836217)_(cropped).jpg",
            DATA_DIR / "musk/elon_musk_royal_society.jpg",
            DATA_DIR / "musk/images.jpeg",
        ]),
    ]

    for sid, imgs in samples:
        files = []
        for i, p in enumerate(imgs):
            if p.exists():
                ctype = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else (
                    "image/png" if p.suffix.lower() == ".png" else None
                )
                if ctype:
                    files.append(("images", (p.name, p.read_bytes(), ctype)))
        if not files:
            print(f"No images found for {sid}, skipping register")
            continue
        r = post(
            f"{BASE}/face-recognition/students/register",
            files=files,
            data={"student_id": sid, "class_id": CLASS_ID},
        )
        assert r.status_code == 200, "register failed"

    time.sleep(1)

    # 3) List students
    r = get(f"{BASE}/face-recognition/students?class_id={CLASS_ID}")
    assert r.status_code == 200
    print(r.json())

    # 4) Attendance with one of the images (if exists)
    test_image = Path("data/trump/Donald_Trump_official_portrait.jpg")
    if test_image.exists():
        r = post(
            f"{BASE}/face-recognition/attendance",
            files={"image": test_image.read_bytes()},
            data={"class_id": CLASS_ID},
        )
        assert r.status_code == 200
        print(r.json())
    else:
        print("No test attendance image found, skipping")

    # 5) Stats
    r = get(f"{BASE}/face-recognition/stats")
    assert r.status_code == 200
    print(r.json())

    # 6) Cleanup: delete class (and all students)
    r = delete(f"{BASE}/classes/{CLASS_ID}")
    assert r.status_code == 200
    print("Cleanup done")


if __name__ == "__main__":
    main()


