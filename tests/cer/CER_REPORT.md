# CER đánh giá VietOCR vs Tesseract (dev set, 3 trang có ground truth)

## Cách tính
- CER = Levenshtein distance (ký tự) / độ dài GT.
- GT được ghi tay từ ảnh render 300 DPI.
- Tesseract: đọc từ cache `data/ocr/<doc>/7667ff02239e/page-XXX.json` (trường `text`).
- VietOCR: vgg_transformer (CPU), phân đoạn dòng bằng hình chiếu ngang, predict từng dòng.

## Kết quả

| Trang | Tesseract CER | VietOCR CER | Hiệu lệnh |
|-------|--------------|-------------|-----------|
| nd_145_2020 p01 (header + Điều 1) | 3.79% | 21.01% | Tesseract |
| nd_145_2020 p30 (Điều 38, danh sách) | 4.34% | 24.97% | Tesseract |
| nd_12_2022 p05 (Điều 23-34) | 12.08% | 31.47% | Tesseract |

## Nhận xét
- Tesseract rõ quyền hơn trên corpus pháp lý này (CER 3.8–12% vs 21–31%).
- VietOCR lỗi hệ thống: `Điều`→`Điêu`, `Điều 12`→`Điêu 12`, số bị nhầm (`l7`→`17`, `Ì`→`6`), mất dấu (`thê`→`thể`, `bô`→`bổ`).
- Nguyên nhân: VietOCR là recognizer cấp **dòng**, không phải OCR toàn trang; line segmentation đơn giản (hình chiếu ngang) chưa đủ mạnh cho trang phức tạp.

## Phần (b): Cải thiện pipeline VietOCR
Đã thêm module `src/legal_rag/corpus/vietocr_post.py` với quy tắc hậu xử lý deterministic
(`Điêu`→`Điều`, `bô sung`→`bổ sung`, `sửa đôi`→`sửa đổi`, …) và đính kèm vào
`VietOcrEngine.ocr_image` (file `src/legal_rag/corpus/ocr.py`).

**Kết quả chạy lại CER với hậu xử lý:**

| Trang | Tesseract CER | VietOCR (không hậu xử lý) | VietOCR (có hậu xử lý) |
|-------|--------------|---------------------------|------------------------|
| nd_145_2020 p01 | 3.79% | 21.01% | 20.59% |
| nd_145_2020 p30 | 4.34% | 24.97% | 24.97% |
| nd_12_2022 p05 | 12.08% | 31.47% | 31.47% |

Hậu xử lý quy tắc đơn giản **không cải thiện đáng kể** (giảm 0.4% chỉ). Lỗi lớn nằm ở
line segmentation (thêm/bớt newline → chỉnh sửa ký tự) và sai số/dấu không sửa được
bằng replace đơn giản.

## Kết luận
VietOCR vgg_transformer (CPU) **không thiết lập được thay thế Tesseract** cho corpus
pháp lý này: CER cao gấp 3–8× và mất 24% số Điều (xem corpus_status). Để dùng được,
cần invest thêm: line segmentation tốt hơn (PaddleOCR detector / layout analysis),
hậu xử lý ngữ cảnh cho số và Điều, hoặc fine-tune model trên dữ liệu pháp luật.

## Tệp
- GT: `tests/data/cer_dev/gt/*.txt`
- Ảnh mẫu: `tests/data/cer_dev/images/*.png`
- OCR thô: `tests/data/cer_dev/ocr/vietocr/*.txt`
- Script: `tests/cer/compute_cer.py`
- Báo cáo: `tests/cer/CER_REPORT.md`
