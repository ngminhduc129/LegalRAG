"""Compute Character Error Rate (CER) between ground truth and OCR output."""
import json
import os
import sys
import types
from pathlib import Path

GT_DIR = Path("tests/data/cer_dev/gt")
OCR_TESS_DIR = Path("data/ocr")
OCR_VIETOCR_OUT = Path("tests/data/cer_dev/ocr/vietocr")
IMG_DIR = Path("tests/data/cer_dev/images")

# mapping: gt_name -> (pdf_path, page_number)
PAGES = [
    ("nd_145_2020_p01", "data/raw/Ngh\u1ecb-\u0111\u1ecbnh-145-2020-N\u0110-CP.pdf", 1),
    ("nd_145_2020_p30", "data/raw/Ngh\u1ecb-\u0111\u1ecbnh-145-2020-N\u0110-CP.pdf", 30),
    ("nd_12_2022_p05", "data/raw/Ngh\u1ecb-\u0111\u1ecbnh-12-2022-N\u0110-CP.pdf", 5),
]


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (DP, O(n*m))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def normalize(text: str) -> str:
    """Keep text as-is (raw CER). Spaces/newlines count as chars."""
    return text


def read_gt(name: str) -> str:
    p = GT_DIR / f"{name}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def read_tesseract(doc: str, page: int) -> str:
    """Read Tesseract cache output."""
    cache_dir = OCR_TESS_DIR / doc / "7667ff02239e"
    p = cache_dir / f"page-{page:03d}.json"
    if not p.exists():
        return ""
    payload = json.loads(p.read_text(encoding="utf-8"))
    text = payload.get("text") or ""
    if not text:
        lines = payload.get("lines") or []
        text = "\n".join(lines)
    return text


def segment_lines(pil_image, min_height=8, pad=4):
    """Chia ảnh thành các dòng theo hình chiếu ngang (horizontal projection)."""
    import numpy as np

    arr = np.array(pil_image.convert("L"))
    binary = (arr < 128).astype(np.uint8)
    proj = binary.sum(axis=1)
    text_rows = proj > 0
    boxes = []
    in_line = False
    start = 0
    for i, is_text in enumerate(text_rows):
        if is_text and not in_line:
            start = i
            in_line = True
        elif not is_text and in_line:
            if i - start >= min_height:
                boxes.append((start, i))
            in_line = False
    if in_line:
        boxes.append((start, len(text_rows)))
    h = arr.shape[0]
    crops = []
    for (top, bottom) in boxes:
        t = max(0, top - pad)
        b = min(h, bottom + pad)
        if b - t < min_height:
            continue
        crop = pil_image.crop((0, t, pil_image.width, b)).convert("RGB")
        crops.append(crop)
    return crops


def apply_pkg_resources_shim():
    if "pkg_resources" in sys.modules:
        return
    mod = types.ModuleType("pkg_resources")

    class _Dist:
        version = "0.0.0"

        def requires(self):
            return []

    def get_distribution(name):
        return _Dist()

    mod.get_distribution = get_distribution
    mod.working_set = []
    sys.modules["pkg_resources"] = mod


def run_vietocr_on_images() -> dict:
    """Run VietOcrEngine (with post-processing) on each dev image, save outputs, return {name: text}."""
    from legal_rag.corpus.ocr import VietOcrEngine

    engine = VietOcrEngine()  # loads model once

    OCR_VIETOCR_OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, pdf_path, page in PAGES:
        img_path = IMG_DIR / f"{name}.png"
        if not img_path.exists():
            results[name] = ""
            continue
        png_bytes = img_path.read_bytes()
        lines = engine.ocr_image(png_bytes)
        text = "\n".join(lines)
        out_path = OCR_VIETOCR_OUT / f"{name}.txt"
        out_path.write_text(text, encoding="utf-8")
        results[name] = text
        print(f"  VietOCR {name}: {len(lines)} lines, {len(text)} chars")
    return results


def main():
    print("=== Tesseract CER ===")
    tess_results = {}
    for name, pdf_path, page in PAGES:
        doc = name.rsplit("_p", 1)[0]
        t = read_tesseract(doc, page)
        tess_results[name] = t
        ref = read_gt(name)
        c = cer(ref, t)
        print(f"  {name}: CER={c:.4f} (ref={len(ref)} hyp={len(t)})")

    print()
    print("=== VietOCR CER ===")
    viet_results = run_vietocr_on_images()
    for name, _, _ in PAGES:
        hyp = viet_results.get(name, "")
        ref = read_gt(name)
        c = cer(ref, hyp)
        print(f"  {name}: CER={c:.4f} (ref={len(ref)} hyp={len(hyp)})")


if __name__ == "__main__":
    main()
