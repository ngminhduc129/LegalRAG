DOCUMENT_NAME = "Bộ luật Lao động"
DOCUMENT_NUMBER = "125/VBHN-VPQH"

# Normalized markers of the trailing signature/authentication block of a VBHN
# document (matched with startswith). Once a line matches, it and all
# following lines are dropped from chunk content.
SIGNATURE_MARKERS = (
    "vanphong",   # VAN PHONG QUOC HQI / VĂN PHÒNG QUỐC HỘI
    "xacthuc",    # XAC THUC / XÁC THỰC VĂN BẢN HỢP NHẤT
    "noinhan",    # Noi nhan: / Nơi nhận:
)

# First tokens (normalized as above) accepted as fuzzy variants of "Điều"
# for corrupted headings, e.g. text-layer "Di~u 220" or OCR "Điu 220".
FUZZY_ARTICLE_TOKENS = (
    "dieu",
    "diu",
)