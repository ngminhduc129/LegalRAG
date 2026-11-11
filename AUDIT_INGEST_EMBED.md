# AUDIT_INGEST_EMBED

Tóm tắt kiểm tra quy trình ingest → chunk → embed cho Corpus Lao động 2019 (`labor_code_2019`).

## Nguyên liệu
- PDF gốc: `data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf` (86 trang)
- Chunks hiện có: `data/chunks/test_reg_labor/all_legal_aware.jsonl` (982 chunk)
- Patch có sẵn: `data/ocr/patches/labor_code_2019_page86.json` (sửa trang 86)

## Tóm tắt các vấn đề (F-01 → F-05 + phát hiện mới)

### F-01 — Kích thước chunk (giảm xuống Low)
- 3/982 chunk > 256 token
- 53/982 chunk > 128 token
- Nhất: `labor_code_2019_điều_219_1_e_point` (532 token) — đoạn văn bị hỏng OCR trang 86
- Xử lý: giảm mức xuống **Low**, không chặn pipeline

### F-02 — `clean_chunk()` không hiệu chỉnh gì (đóng / Low)
- Chạy `clean_chunk()` lên toàn bộ 982 chunk: **0 chunk thay đổi content**
- Không có dòng nào bị xóa (số trang, chú thích, v.v.)
- Không có nội dung luật hợp lệ bị xóa nhầm (số thứ tự, tiêu đề, năm)
- `clean_chunk()` là no-op trên tập này → không xử lý được văn bản rác trang 86, không xử lý chunk kết thúc giữa câu
- Xử lý: hạ **Low / đóng**

### Phát hiện mới — Text layer OCR hỏng trang 86, patch không áp dụng trong `data/chunks` hiện có
- Trang 86 là trang quét (image_coverage = 1.0), text layer là OCR bị hỏng (mất dấu, gãy chữ): `lao d{>ng`, `Ban tr9ng tai`, `khong duqc`, `quy8t dinh`, `£IJ`, `Le Quang Tung`
- `data/ocr/patches/labor_code_2019_page86.json` có sẵn nội dung thay thế đúng cho trang 86, nhưng **không được áp dụng** khi sinh `data/chunks/test_reg_labor` (run_manifest ghi `a6be9f6`, chưa truyền `--patches`)
- Kết quả: chunk `điều_220_3_none_clause` chứa văn bản rác; 4 chunk (`điều_219_1_a/c/4_b_point`, `điều_219_2_a_point__occ4`) kết thúc bằng `“Điều 54/50/55/32…` (mở ngoặc kép không khích, nội dung ادامه sang trang sau)
- Khắc phục: rebuild với `--patches data/ocr/patches/labor_code_2019_page86.json` → 983 chunk, không còn ký tự rác, `điều_219_1_e_point` = `"Bãi bỏ khoản 2 Điều 32."` (đầy đủ, kết thúc bằng dấu chấm), `điều_220_3_none_clause` bị thay bằng `điều_220_2_none_clause` (nội dung sạch)
- Cảnh báo chất lượng đã được thêm vào extractor/pipeline: `PAGE_QUALITY` được ghi vào `parse_report` và `run_manifest` (cạnh `empty_pages`)

### F-04 — Mở
- Chưa xử lý. Vẫn cần theo dõi.

### F-05 — Trang rỗng đã xử lý, còn thiếu trang văn bản hỏng
- `empty_pages` đã được ghi vào parse_report và manifest
- Trang văn bản hỏng (như trang 86: có text layer nhưng rác) **chưa được xử lý** trước khi thêm cảnh báo PAGE_QUALITY

## Chi tiết trang bị gắn cỡ (PAGE_QUALITY)

Ngưỡng (được hiệu chỉnh từ thống kê 86 trang):
- Strange chars `{}=<>~£|^` > 0.5%
- Mixed alphanum từ (chữ+số như `tr9ng`) > 1.5%
- Chữ có dấu tiếng Việt < 5.0% trong tổng chữ cái

| Trang | Strange% | Mixed% | ViDiac% | Chú thích |
|-------|----------|--------|---------|-----------|
| 1–85 | ~0 | ~0 | ~12.7% | Sạch |
| 86 | 2.11% | 3.16% | 0.00% | **BỊ CỘ** — OCR hỏng, toàn bộ trang |

Chỉ **trang 86** bị gắn cờ. Trang 1 có mixed ≈ 1.33% (số hiệu văn bản `45/2019/QH14`) nhưng dưới ngưỡng 1.5% → không cờ.

## Commit gốc
- Chunks `data/chunks/test_reg_labor/` được commit tại **fc1f962** (message "hi", đồng thời viết lại `src/legal_rag/chunking/*`)
- `run_manifest.json` trong run đó ghi `git_commit: a6be9f6` (commit build chạy, khác commit chứa file)
- Nên ghi **fc1f962** là commit gốc chứa `data/chunks`, kèm lưu ý `run_manifest` ghi `a6be9f6`

## Lưu ý số dòng
Các commit sau fc1f962 đã sửa `src/legal_rag/chunking/*`, `src/legal_rag/parsing/*`, `src/legal_rag/corpus/*` → số dòng trong báo cáo trước đó có thể đã lệch so với HEAD.
