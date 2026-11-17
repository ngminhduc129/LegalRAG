# CER đánh giá VietOCR vs Tesseract (dev set, 3 trang có ground truth)

## Cách tính
- CER = Levenshtein distance (ký tự) / độ dài GT.
- GT được ghi tay từ ảnh render 300 DPI.
- Tesseract: đọc từ cache `data/ocr/<doc>/7667ff02239e/page-XXX.json` (trường `text`).
- VietOCR: vgg_transformer (CPU), phân đoạn dòng bằng hình chiếu ngang, predict từng dòng.

## Kết quả

| Trang | Tesseract CER | VietOCR CER | PaddleOCR CER |
|-------|--------------|-------------|---------------|
| nd_145_2020 p01 (header + Điều 1) | 3.79% | 21.01% | 26.42% |
| nd_145_2020 p30 (Điều 38, danh sách) | 4.34% | 24.97% | 36.08% |
| nd_12_2022 p05 (Điều 23-34) | 12.08% | 31.47% | 38.35% |

## Nhận xét
- Tesseract rõ quyền hơn trên corpus pháp lý này (CER 3.8–12%).
- VietOCR lỗi hệ thống: `Điều`→`Điêu`, số bị nhầm, mất dấu. Hậu xử lý quy tắc đơn giản không giúp ích.
- PaddleOCR 2.x (`paddlepaddle==2.6.2` + `paddleocr==2.10.0`, CPU) chạy được trên Windows (~6s/trang) nhưng CER rất cao (26–38%), do nhận dạng chữ kém và tách dòng vỡ (ví dụ `nd_12_2022` p05: 13 dòng GT → 36 dòng OCR).
- Nguyên nhân căn bản: cả VietOCR và PaddleOCR là recognizer đang chưa đủ mạnh cho văn bản pháp lý scan (cột, chữ số, dấu, header/footer phức tạp).

## Kết luận cuối cùng
**Giữ Tesseract làm engine OCR mặc định cho corpus.** 

VietOCR và PaddleOCR được giữ như engine tuỳ chọn (đã có adapter trong `ocr.py`), nhưng **không chạy full corpus** cho chúng vì CER cao gấp 3–8× và mất nhiều Điều (VietOCR mất 24% số Điều so với Tesseract).

Để dùng VietOCR/PaddleOCR sau này, cần invest thêm: line segmentation tốt hơn (Paddle det / DocLayout), hậu xử lý ngữ cảnh cho số và Điều, hoặc fine-tune model trên dữ liệu pháp lý.

## Tệp
- GT: `tests/data/cer_dev/gt/*.txt`
- Ảnh mẫu: `tests/data/cer_dev/images/*.png`
- OCR thô: `tests/data/cer_dev/ocr/{vietocr,paddleocr}/*.txt`
- Script: `tests/cer/compute_cer.py`, `tests/cer/compare_paddle.py`
- Báo cáo: `tests/cer/CER_REPORT.md`
