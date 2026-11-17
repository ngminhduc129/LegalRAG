"""So sánh CER: Tesseract (từ cache) vs PaddleOCR (mới) trên 3 trang dev GT."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("LEGALRAG_ROOT", "E:\\LegalRAG"))

GT_DIR = Path("tests/data/cer_dev/gt")
TESS_CACHE = Path("data/ocr")
PADDLE_OUT = Path("tests/data/cer_dev/ocr/paddleocr")
IMG_DIR = Path("tests/data/cer_dev/images")

PAGES = [
    ("nd_145_2020_p01", "nd_145_2020", 1),
    ("nd_145_2020_p30", "nd_145_2020", 30),
    ("nd_12_2022_p05", "nd_12_2022", 5),
]


def levenshtein(a, b):
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


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def read_gt(name):
    p = GT_DIR / f"{name}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_tesseract(doc, page):
    p = TESS_CACHE / doc / "7667ff02239e" / f"page-{page:03d}.json"
    if not p.exists():
        return ""
    payload = json.loads(p.read_text(encoding="utf-8"))
    text = payload.get("text") or ""
    if not text:
        text = "\n".join(payload.get("lines") or [])
    return text


def run_paddleocr() -> dict:
    """Chạy PaddleOCR 2.x trên 3 ảnh dev, lưu ra paddleocr/, trả về {name: text}."""
    # Kích hoạt venv paddleocr qua PATH đã được set trước khi gọi script này.
    from paddleocr import PaddleOCR
    from PIL import Image
    import numpy as np

    engine = PaddleOCR(lang="vi", use_angle_cls=True, show_log=False)
    PADDLE_OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, doc, page in PAGES:
        img_path = IMG_DIR / f"{name}.png"
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        result = engine.ocr(arr, cls=True)
        lines = []
        if result and result[0]:
            items = []
            for box, payload in result[0]:
                text = str(payload[0])
                score = float(payload[1]) if len(payload) > 1 else 1.0
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                y_center = sum(ys) / 4.0
                x_min = min(xs)
                items.append((y_center, x_min, text, score))
            items.sort(key=lambda t: t[0])
            groups = []
            for y_c, x_m, t, s in items:
                placed = False
                for g in groups:
                    if abs(y_c - g[0][0]) <= 20:
                        g.append((x_m, t, s))
                        placed = True
                        break
                if not placed:
                    groups.append([(x_m, t, s)])
            for g in groups:
                g.sort(key=lambda t: t[0])
                lines.extend(t for _, t, _ in g)
        text = "\n".join(lines)
        (PADDLE_OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        results[name] = text
        print(f"  PaddleOCR {name}: {len(lines)} lines, {len(text)} chars")
    return results


def main():
    paddle_results = run_paddleocr()
    print()
    print(f"{'trang':<16}{'GT':>6}{'Tess CER':>10}{'Paddle CER':>12}")
    for name, doc, page in PAGES:
        ref = read_gt(name)
        tess = read_tesseract(doc, page)
        padd = paddle_results.get(name, "")
        print(
            f"{name:<16}{len(ref):>6}{cer(ref, tess):>10.4f}{cer(ref, padd):>12.4f}"
        )


if __name__ == "__main__":
    main()
