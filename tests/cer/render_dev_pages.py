"""Render dev-set pages as PNG for ground truth transcription."""
import fitz, os

pages = [
    ("data/raw/Ngh\u1ecb-\u0111\u1ecbnh-145-2020-N\u0110-CP.pdf", 1, "nd_145_2020_p01"),
    ("data/raw/Ngh\u1ecb-\u0111\u1ecbnh-145-2020-N\u0110-CP.pdf", 30, "nd_145_2020_p30"),
    ("data/raw/Ngh\u1ecb-\u0111\u1ecbnh-12-2022-N\u0110-CP.pdf", 5, "nd_12_2022_p05"),
    ("data/raw/Th\u00f4ng-t\u01b0-08-2026-TT-BNV.pdf", 6, "tt_08_2026_p06"),
    ("data/raw/Ngh\u1ecb-\u0111\u1ecbnh-10-2024-N\u0110-CP.pdf", 10, "nd_10_2024_p10"),
]
outdir = "tests/data/cer_dev/images"
os.makedirs(outdir, exist_ok=True)
for pdf, pno, name in pages:
    doc = fitz.open(pdf)
    pix = doc[pno - 1].get_pixmap(dpi=300, alpha=False)
    out = os.path.join(outdir, name + ".png")
    pix.save(out)
    doc.close()
    print("saved", out, pix.width, "x", pix.height)
