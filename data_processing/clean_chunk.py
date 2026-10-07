import json
import re
from pathlib import Path


# =========================================================
# 1. CẤU HÌNH
# =========================================================

INPUT_PATH = Path(
    r"E:\LegalRAG\output\chunks.jsonl"
)

OUTPUT_PATH = Path(
    r"E:\LegalRAG\output\chunks.jsonl"
)

# File chứa các chunk cần kiểm tra thủ công
REVIEW_PATH = Path(
    r"E:\LegalRAG\output\chunks_review.jsonl"
)


# =========================================================
# 2. CÁC HÀM NHẬN DIỆN
# =========================================================

def is_page_number(line: str) -> bool:
    """
    Kiểm tra một dòng có phải chỉ chứa số trang hay không.

    Ví dụ:
        "3"   -> True
        "10"  -> True
        "123" -> True

    Không xử lý:
        "Khoản 3"
        "Điều 123"
        "30 ngày"
    """

    line = line.strip()

    return bool(
        re.fullmatch(r"\d{1,3}", line)
    )


def is_superscript_footnote(text: str) -> bool:
    """
    Kiểm tra số chú thích dạng superscript Unicode.

    Ví dụ:
        "THI HÀNH³"
        "lao động²"
    """

    return bool(
        re.search(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", text)
    )


def remove_superscript_footnote(text: str):
    """
    Xóa số chú thích dạng superscript ở cuối chuỗi.

    Ví dụ:
        THI HÀNH³
        ->
        THI HÀNH
    """

    new_text = re.sub(
        r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$",
        "",
        text
    ).rstrip()

    return new_text


def remove_known_footnote_number(text: str):
    """
    Xử lý trường hợp PDF biến footnote superscript thành
    số bình thường.

    Ví dụ:

        ĐIỀU KHOẢN THI HÀNH3
        ->
        ĐIỀU KHOẢN THI HÀNH

    CHỈ áp dụng cho các heading đã biết.

    Không xóa số ở cuối mọi câu vì có thể làm mất
    nội dung pháp luật thực tế.
    """

    known_headings = [
        "ĐIỀU KHOẢN THI HÀNH",
        "ĐIỀU KHOẢN THI HÀNH.",
        "ĐIỀU KHOẢN CHUYỂN TIẾP",
    ]

    stripped = text.strip()

    for heading in known_headings:

        # Ví dụ:
        # ĐIỀU KHOẢN THI HÀNH3
        pattern = (
            r"^"
            + re.escape(heading.rstrip("."))
            + r"\s*\d+$"
        )

        if re.fullmatch(
            pattern,
            stripped,
            flags=re.IGNORECASE
        ):
            return heading.rstrip(".")

    return text


def clean_metadata_field(text: str):
    """Làm sạch metadata field (chapter_title, article_title, v.v.)"""
    if not text:
        return text
    return remove_known_footnote_number(text)


# =========================================================
# 3. CLEAN CONTENT
# =========================================================

def is_suspicious_ending(content: str) -> bool:
    """
    Kiem tra xem dong cuoi cung co chi chua so khong.
    Chi flag neu dong cuoi HOAN TOAN la so (1-3 chu so) -
    tuc la so trang hoac footnote roi rac.
    Khong flag cac truong hop hop le nhu:
    - "khoan 2", "Dieu 36", "02 thang", "15 ngay", "2026"
    """
    lines = content.strip().splitlines()
    if not lines:
        return False
    last_line = lines[-1].strip()
    return bool(re.fullmatch(r"\d{1,3}", last_line))


def clean_content(content: str):
    """
    Làm sạch nội dung của một chunk.

    Trả về:
        cleaned_content
        actions
        suspicious
    """

    actions = []
    suspicious = []

    if not content:
        return "", actions, suspicious

    # -----------------------------------------------------
    # Chuẩn hóa line ending
    # -----------------------------------------------------

    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # -----------------------------------------------------
    # Tách dòng
    # -----------------------------------------------------

    raw_lines = content.split("\n")

    cleaned_lines = []

    for line in raw_lines:

        line = line.strip()

        # Bỏ dòng trống
        if not line:
            continue

        # -------------------------------------------------
        # 1. SỐ TRANG
        # -------------------------------------------------

        if is_page_number(line):

            actions.append(
                f"REMOVE_PAGE_NUMBER:{line}"
            )

            continue

        # -------------------------------------------------
        # 2. FOOTNOTE SUPERSCRIPT
        # -------------------------------------------------

        if is_superscript_footnote(line):

            new_line = remove_superscript_footnote(line)

            if new_line != line:

                actions.append(
                    f"REMOVE_SUPERSCRIPT_FOOTNOTE:"
                    f"{line} -> {new_line}"
                )

                line = new_line

        # -------------------------------------------------
        # 3. FOOTNOTE DẠNG SỐ THƯỜNG
        # -------------------------------------------------

        new_line = remove_known_footnote_number(line)

        if new_line != line:

            actions.append(
                f"REMOVE_HEADING_FOOTNOTE:"
                f"{line} -> {new_line}"
            )

            line = new_line

        # -------------------------------------------------
        # 4. KHÔNG ĐỂ DÒNG RỖNG
        # -------------------------------------------------

        if line:
            cleaned_lines.append(line)

    # -----------------------------------------------------
    # GHÉP LẠI
    # -----------------------------------------------------

    cleaned = "\n".join(cleaned_lines)

    # -----------------------------------------------------
    # CHUẨN HÓA SPACE
    # -----------------------------------------------------

    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned
    )

    # Không để quá nhiều newline
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned
    )

    cleaned = cleaned.strip()

    # =====================================================
    # KIỂM TRA NGHI VẤN
    # =====================================================

    # Nếu vẫn còn dòng chỉ chứa số
    for line in cleaned.split("\n"):

        if is_page_number(line):
            suspicious.append(
                f"PAGE_NUMBER_REMAIN:{line}"
            )

    # -----------------------------------------------------
    # Chunk ket thuc bang so trang / footnote (dong cuoi chi chua so)
    # -----------------------------------------------------

    if is_suspicious_ending(cleaned):

        suspicious.append(
            "ENDS_WITH_NUMBER"
        )

    # -----------------------------------------------------
    # Footnote superscript còn sót
    # -----------------------------------------------------

    if re.search(
        r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$",
        cleaned
    ):
        suspicious.append(
            "POSSIBLE_FOOTNOTE"
        )

    return cleaned, actions, suspicious


# =========================================================
# 4. CLEAN MỘT CHUNK
# =========================================================

def clean_chunk(chunk: dict):

    new_chunk = dict(chunk)

    # Clean metadata fields
    for field in ["chapter_title", "article_title"]:
        if new_chunk.get(field):
            new_chunk[field] = clean_metadata_field(new_chunk[field])

    original_content = chunk.get(
        "content",
        ""
    )

    cleaned_content, actions, suspicious = (
        clean_content(original_content)
    )

    # Nội dung sau khi clean
    new_chunk["content"] = cleaned_content

    # -----------------------------------------------------
    # Lưu log để kiểm tra
    # -----------------------------------------------------

    if actions:
        new_chunk["cleaned"] = True
        new_chunk["clean_actions"] = actions
    else:
        new_chunk["cleaned"] = False
        new_chunk["clean_actions"] = []

    # -----------------------------------------------------
    # Đánh dấu cần review
    # -----------------------------------------------------

    if suspicious:
        new_chunk["needs_review"] = True
        new_chunk["review_reasons"] = suspicious
    else:
        new_chunk["needs_review"] = False
        new_chunk["review_reasons"] = []

    return new_chunk


# =========================================================
# 5. ĐỌC JSONL
# =========================================================

def load_chunks(path: Path):

    chunks = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                chunk = json.loads(line)

                chunks.append(chunk)

            except json.JSONDecodeError as e:

                print(
                    f"[ERROR] JSON lỗi ở dòng "
                    f"{line_number}: {e}"
                )

    return chunks


# =========================================================
# 6. SAVE JSONL
# =========================================================

def save_chunks(
    chunks,
    path: Path
):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


# =========================================================
# 7. MAIN
# =========================================================

def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy file:\n"
            f"{INPUT_PATH}"
        )

    print("=" * 70)
    print("CLEAN LEGAL RAG CHUNKS")
    print("=" * 70)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print("\n[1] Đọc chunks...")

    chunks = load_chunks(
        INPUT_PATH
    )

    print(
        f"Đã đọc: {len(chunks)} chunks"
    )

    # -----------------------------------------------------
    # Clean
    # -----------------------------------------------------

    print("\n[2] Đang clean...")

    cleaned_chunks = []

    total_cleaned = 0
    total_review = 0

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        cleaned_chunk = clean_chunk(
            chunk
        )

        cleaned_chunks.append(
            cleaned_chunk
        )

        if cleaned_chunk["cleaned"]:

            total_cleaned += 1

        if cleaned_chunk["needs_review"]:

            total_review += 1

    # -----------------------------------------------------
    # Save cleaned
    # -----------------------------------------------------

    print("\n[3] Lưu chunks sạch...")

    save_chunks(
        cleaned_chunks,
        OUTPUT_PATH
    )

    # -----------------------------------------------------
    # Tạo file review
    # -----------------------------------------------------

    review_chunks = [
        chunk
        for chunk in cleaned_chunks
        if chunk["needs_review"]
    ]

    if review_chunks:

        save_chunks(
            review_chunks,
            REVIEW_PATH
        )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)

    print(
        f"Tổng chunks       : {len(chunks)}"
    )

    print(
        f"Chunks đã clean   : {total_cleaned}"
    )

    print(
        f"Chunks cần review : {total_review}"
    )

    print(
        f"\nFile sạch:"
        f"\n{OUTPUT_PATH}"
    )

    if review_chunks:

        print(
            f"\nFile cần kiểm tra:"
            f"\n{REVIEW_PATH}"
        )

    print("\nHoàn thành!")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()