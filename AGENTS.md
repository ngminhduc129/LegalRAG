# AGENTS.md — Vietnamese Labor Law Legal RAG

Chỉ gồm quy tắc bắt buộc và các điểm phải kiểm tra. Kế hoạch chi tiết nằm ở file Excel/MD riêng.

## Quy tắc không được vi phạm

1. **Không bịa số liệu.** Mọi con số trong bảng, biểu, bài viết phải đến từ lần chạy thực tế có log và config. Chưa có kết quả thì ghi `TBD`. Các "target" trong kế hoạch không phải kết quả.
2. **Không bịa trích dẫn.** Không tự tạo tên bài báo, tác giả, năm, DOI. Thiếu thông tin thì ghi `[CẦN XÁC MINH]`.
3. **Không tune trên test set.** Test set đã đóng băng thì không sửa. Chọn tham số bằng dev set.
4. **Không sửa `data/raw/`.** Mọi biến đổi ghi ra thư mục khác bằng script chạy lại được.
5. **Không ghi đè kết quả cũ.** Mỗi lần chạy có `run_id` riêng.
6. **Không phá cấu trúc pháp lý** khi chuẩn hóa: giữ nguyên đánh số Chương/Mục/Điều/Khoản/Điểm.
7. **Không để mất truy vết nguồn.** Mỗi chunk phải truy ngược được về `document_id`, trang gốc, Điều/Khoản/Điểm.
8. **Không trích dẫn điều không có trong ngữ cảnh đã retrieve.** Câu trả lời chỉ được dẫn Điều/Khoản có trong context.
9. **Không bỏ qua hiệu lực văn bản.** Không trả về điều khoản hết hiệu lực mà không cảnh báo.
10. **Không đổi giữa chừng** định nghĩa metric, cách tính "hit", tập test, hay baseline giữa các lần so sánh.
11. **Không hard-code** API key, đường dẫn, tham số. Không commit `.env` hoặc dữ liệu nhạy cảm.
12. **Không đoán** dữ kiện pháp lý hoặc học thuật khi không chắc. Nói rõ là không chắc.
13. **Mọi thay đổi code/mới (feature, fix, refactor, config, test) phải commit và push lên git ngay sau khi hoàn thành.** Không để thay đổi local chưa commit. Commit message rõ ràng, tiếng Anh, format: `type(scope): short description`.

## Phải kiểm tra trước khi báo hoàn thành

### So sánh công bằng
- [ ] Các hệ thống so sánh dùng cùng corpus, cùng tập câu hỏi, cùng retriever/LLM/prompt (chỉ đổi đúng biến đang thử), cùng k, cùng seed.
- [ ] Cách tính "hit" (đúng Điều, hay đúng cả Khoản) thống nhất cho mọi chiến lược chunking.
- [ ] Fixed-size được thử nhiều cỡ (300/500/800), không chỉ một cấu hình.

### Rò rỉ dữ liệu
- [ ] Gold label, gold article ID, reference answer không lọt vào index, prompt hay text của chunk.
- [ ] Metadata nối vào text (số Điều, số hiệu) không làm BM25 khớp câu hỏi chỉ nhờ số Điều. Câu hỏi có và không nhắc số Điều được đánh giá riêng.
- [ ] Không trùng lặp câu hỏi giữa các split.

### Chunking
- [ ] Không trùng nội dung, không trùng structure key, không thiếu phân cấp.
- [ ] Tham chiếu kiểu "khoản 1 Điều 35" vẫn truy được sau khi chunk.
- [ ] Đã xem phân bố độ dài chunk; chunk không vượt giới hạn token của embedding model (nếu vượt thì tách theo Khoản).
- [ ] Metadata nằm trong text embed hay chỉ ở trường riêng: đã chọn rõ và ghi lại.

### Retrieval
- [ ] Tokenizer tiếng Việt cho BM25 đã ghi rõ (tên, phiên bản).
- [ ] Văn bản tiếng Việt đã chuẩn hóa Unicode NFC trước khi so khớp hoặc tokenize.
- [ ] Embedding model hỗ trợ tiếng Việt; dùng đúng tiền tố query/passage theo model card; ghi tên và phiên bản.
- [ ] RRF: ghi rõ `k` (mặc định 60) và số ứng viên lấy từ mỗi retriever; "Hybrid" và "Hybrid + RRF" định nghĩa khác nhau.

### Sinh câu trả lời và đánh giá
- [ ] Citation trong câu trả lời thực sự hỗ trợ claim, không chỉ tồn tại trong context.
- [ ] Hallucination rate = claim không có căn cứ / tổng claim pháp lý, theo đúng quy tắc annotation đã định.
- [ ] Có tỉ lệ trả về điều khoản hết hiệu lực, không chỉ Recall@k.

### Tái lập
- [ ] Lưu config snapshot, git commit, seed, phiên bản model/embedding/LLM, `metrics.json`.
- [ ] Bảng và biểu sinh tự động từ `metrics.json`, không nhập tay.
- [ ] Chạy lại cho cùng kết quả.

### Kết quả bất thường
- [ ] Metric quá cao, trùng split, hoặc kết quả ngược giả thuyết: **dừng lại và báo cáo**, không tiếp tục như bình thường. Kết quả không như giả thuyết vẫn phải được báo cáo trung thực.

### Bài viết
- [ ] Mọi nhận định có dẫn chứng từ kết quả thật; không overclaim.
- [ ] Có phần hạn chế và disclaimer: hệ thống không thay thế tư vấn pháp lý.
- [ ] Từng RQ được trả lời tường minh.

## Quy ước tối thiểu

- Giao tiếp bằng tiếng Việt; code, tên biến, docstring, commit bằng tiếng Anh.
- File text đọc/ghi với `encoding="utf-8"`.
- Chạy test liên quan (parser, chunker, metric, RRF) trước khi báo xong.
