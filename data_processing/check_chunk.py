
import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

# =====================================================
# CAU HINH
# =====================================================

INPUT_FILE = Path(r"E:\LegalRAG\output\chunks.jsonl")
REPORT_DIR = Path(r"E:\LegalRAG\output\check_report")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

ISSUES_CSV = REPORT_DIR / "issues.csv"
SUMMARY_TXT = REPORT_DIR / "summary.txt"
CHUNKS_CSV = REPORT_DIR / "checked_chunks.csv"

# Bộ luật Lao động có 17 chương, 220 điều theo cấu trúc
# của Bộ luật số 45/2019/QH14 ban đầu.
# Văn bản hợp nhất có thể có nội dung sửa đổi nên chỉ dùng
# ngưỡng này để cảnh báo, không dùng làm kết luận pháp lý.
EXPECTED_ARTICLE_COUNT = 220

# Điều chỉnh nếu PDF của bạn có trang bìa/phụ lục khác.
MIN_CONTENT_LENGTH = 5

# =====================================================
# REGEX
# =====================================================

CHAPTER_RE = re.compile(
    r"^Chương\s+([IVXLCDM]+|\d+)$",
    re.IGNORECASE
)

ARTICLE_RE = re.compile(
    r"^Điều\s+(\d+)$",
    re.IGNORECASE
)

CLAUSE_RE = re.compile(r"^\d+$")
POINT_RE = re.compile(r"^[a-zđ]$", re.IGNORECASE)

# Dấu hiệu tiêu đề pháp luật bị lẫn vào nội dung
STRUCTURE_LINE_RE = re.compile(
    r"^\s*(Chương\s+[IVXLCDM\d]+|Điều\s+\d+)\b",
    re.IGNORECASE
)

# Các dấu hiệu OCR hoặc trích xuất có thể có vấn đề.
SUSPICIOUS_PATTERNS = [
    (re.compile(r"\ufffd"), "Có ký tự thay thế �"),
    (re.compile(r"[A-Za-z]\?{2,}"), "Có chuỗi dấu hỏi bất thường"),
    (re.compile(r"\b(?:CNG|BO LUT|LAO DONG)\b", re.I),
     "Có thể mất dấu tiếng Việt"),
]


# =====================================================
# DOC JSONL
# =====================================================

def load_chunks(path):
    chunks = []

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            "Hãy chạy chunk_labor_code.py trước."
        )

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
                chunk["_line_number"] = line_number
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                chunks.append({
                    "_line_number": line_number,
                    "_json_error": str(e),
                    "content": line
                })

    return chunks


# =====================================================
# KIEM TRA MOT CHUNK
# =====================================================

def check_one_chunk(chunk):
    issues = []

    line_no = chunk.get("_line_number", "?")

    if "_json_error" in chunk:
        issues.append((
            "ERROR",
            "INVALID_JSON",
            f"Dòng JSONL không hợp lệ: {chunk['_json_error']}"
        ))
        return issues

    content = str(chunk.get("content") or "").strip()

    # 1. Nội dung
    if not content:
        issues.append((
            "ERROR", "EMPTY_CONTENT", "Nội dung chunk trống"
        ))
        return issues

    if len(content) < MIN_CONTENT_LENGTH:
        issues.append((
            "WARNING",
            "VERY_SHORT_CONTENT",
            f"Chunk chỉ có {len(content)} ký tự"
        ))

    # 2. Metadata bắt buộc
    required_fields = [
        "document_name",
        "document_number",
        "article",
        "chunk_type",
        "page_start",
        "page_end",
    ]

    for field in required_fields:
        if chunk.get(field) is None or chunk.get(field) == "":
            issues.append((
                "ERROR",
                "MISSING_METADATA",
                f"Thiếu metadata: {field}"
            ))

    # 3. Số điều
    article = chunk.get("article")

    if article is not None:
        match = ARTICLE_RE.fullmatch(str(article).strip())

        if not match:
            issues.append((
                "ERROR",
                "INVALID_ARTICLE_FORMAT",
                f"Định dạng article không hợp lệ: {article}"
            ))

    # 4. Khoản
    clause = chunk.get("clause")

    if clause is not None:
        if not CLAUSE_RE.fullmatch(str(clause).strip()):
            issues.append((
                "WARNING",
                "INVALID_CLAUSE_FORMAT",
                f"Khoản không phải số nguyên: {clause}"
            ))

    # 5. Điểm phải có khoản cha
    point = chunk.get("point")

    if point is not None:
        if not POINT_RE.fullmatch(str(point).strip()):
            issues.append((
                "WARNING",
                "INVALID_POINT_FORMAT",
                f"Ký hiệu điểm bất thường: {point}"
            ))

        if clause is None:
            issues.append((
                "ERROR",
                "POINT_WITHOUT_CLAUSE",
                "Có điểm nhưng không có khoản cha"
            ))

    # 6. Loại chunk
    valid_types = {
        "article_intro",
        "clause",
        "clause_intro",
        "point",
    }

    if chunk.get("chunk_type") not in valid_types:
        issues.append((
            "WARNING",
            "UNKNOWN_CHUNK_TYPE",
            f"Loại chunk không nằm trong danh sách dự kiến: "
            f"{chunk.get('chunk_type')}"
        ))

    # 7. Số trang
    try:
        page_start = int(chunk["page_start"])
        page_end = int(chunk["page_end"])

        if page_start < 1 or page_end < page_start:
            issues.append((
                "ERROR",
                "INVALID_PAGE_RANGE",
                f"Khoảng trang không hợp lệ: "
                f"{page_start}-{page_end}"
            ))
    except (KeyError, TypeError, ValueError):
        issues.append((
            "ERROR",
            "INVALID_PAGE_METADATA",
            "page_start/page_end không phải số hợp lệ"
        ))

    # 8. Cảnh báo nội dung có thể chứa tiêu đề cấu trúc
    # Đây là cảnh báo, vì văn bản pháp luật có thể trích dẫn
    # Điều hoặc Chương khác trong nội dung.
    for line in content.splitlines():
        if STRUCTURE_LINE_RE.match(line):
            issues.append((
                "WARNING",
                "POSSIBLE_MERGED_STRUCTURE",
                f"Nội dung có dòng giống tiêu đề Chương/Điều: "
                f"{line[:100]}"
            ))
            break

    # 9. Dấu hiệu lỗi ký tự
    for pattern, message in SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            issues.append((
                "WARNING",
                "SUSPICIOUS_TEXT",
                message
            ))

    # 10. Chunk quá dài: cần xem xét lại chiến lược chia
    if len(content) > 5000:
        issues.append((
            "WARNING",
            "VERY_LONG_CHUNK",
            f"Chunk có {len(content)} ký tự"
        ))

    return issues


# =====================================================
# KIEM TRA TOAN BO
# =====================================================

def run_checks(chunks):
    all_issues = []
    article_numbers = set()
    article_to_chunks = defaultdict(list)
    exact_content_counter = Counter()

    for index, chunk in enumerate(chunks):
        chunk_id = index + 1
        chunk["_chunk_id"] = chunk_id

        for severity, code, message in check_one_chunk(chunk):
            all_issues.append({
                "chunk_id": chunk_id,
                "source_line": chunk.get("_line_number"),
                "severity": severity,
                "issue_code": code,
                "message": message,
                "article": chunk.get("article"),
                "clause": chunk.get("clause"),
                "point": chunk.get("point"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "content_preview": str(
                    chunk.get("content", "")
                )[:250],
            })

        article = chunk.get("article")
        if article:
            match = ARTICLE_RE.fullmatch(str(article).strip())
            if match:
                article_numbers.add(int(match.group(1)))
                article_to_chunks[article].append(chunk)

        normalized_content = re.sub(
            r"\s+", " ",
            str(chunk.get("content") or "").strip()
        )

        if normalized_content:
            exact_content_counter[normalized_content] += 1

    # 11. Chunk trùng nội dung
    for chunk in chunks:
        content = re.sub(
            r"\s+", " ",
            str(chunk.get("content") or "").strip()
        )

        if content and exact_content_counter[content] > 1:
            all_issues.append({
                "chunk_id": chunk.get("_chunk_id"),
                "source_line": chunk.get("_line_number"),
                "severity": "WARNING",
                "issue_code": "DUPLICATE_CONTENT",
                "message": (
                    "Nội dung trùng với ít nhất một chunk khác"
                ),
                "article": chunk.get("article"),
                "clause": chunk.get("clause"),
                "point": chunk.get("point"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "content_preview": content[:250],
            })

    # 12. Điều bị chia thành quá nhiều chunk trùng metadata
    # Không khẳng định sai, chỉ đưa ra để kiểm tra thủ công.
    for article, article_chunks in article_to_chunks.items():
        seen = set()

        for chunk in article_chunks:
            key = (
                chunk.get("clause"),
                chunk.get("point"),
                chunk.get("chunk_type")
            )

            if key in seen:
                all_issues.append({
                    "chunk_id": chunk.get("_chunk_id"),
                    "source_line": chunk.get("_line_number"),
                    "severity": "WARNING",
                    "issue_code": "REPEATED_STRUCTURE_KEY",
                    "message": (
                        f"{article} có nhiều chunk cùng "
                        f"(clause, point, chunk_type) = {key}"
                    ),
                    "article": article,
                    "clause": chunk.get("clause"),
                    "point": chunk.get("point"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "content_preview": str(
                        chunk.get("content", "")
                    )[:250],
                })

            seen.add(key)

    return all_issues, article_numbers


# =====================================================
# XUAT BAO CAO
# =====================================================

def save_issues_csv(issues):
    fields = [
        "chunk_id",
        "source_line",
        "severity",
        "issue_code",
        "message",
        "article",
        "clause",
        "point",
        "page_start",
        "page_end",
        "content_preview",
    ]

    with ISSUES_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)


def save_checked_chunks(chunks):
    if not chunks:
        return

    # Không xuất các trường nội bộ bắt đầu bằng "_".
    fields = [
        key for key in chunks[0].keys()
        if not key.startswith("_")
    ]

    with CHUNKS_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(chunks)


def main():
    chunks = load_chunks(INPUT_FILE)

    if not chunks:
        print("File không có chunk nào.")
        return

    issues, article_numbers = run_checks(chunks)

    errors = sum(
        1 for issue in issues
        if issue["severity"] == "ERROR"
    )
    warnings = sum(
        1 for issue in issues
        if issue["severity"] == "WARNING"
    )

    issue_counts = Counter(
        issue["issue_code"] for issue in issues
    )

    save_issues_csv(issues)
    save_checked_chunks(chunks)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("BAO CAO KIEM TRA CHUNK LEGAL RAG\n")
        f.write("=" * 45 + "\n")
        f.write(f"Tổng số chunk: {len(chunks)}\n")
        f.write(f"Số điều khác nhau: {len(article_numbers)}\n")
        f.write(f"Số lỗi: {errors}\n")
        f.write(f"Số cảnh báo: {warnings}\n\n")
        f.write("Chi tiết theo loại vấn đề:\n")

        for code, count in issue_counts.most_common():
            f.write(f"- {code}: {count}\n")

        f.write(
            "\nLưu ý: Đây là kiểm tra heuristic, "
            "không thay thế việc đối chiếu PDF gốc.\n"
        )

    print("\n========== KET QUA KIEM TRA ==========")
    print(f"Tổng chunk: {len(chunks)}")
    print(f"Số điều khác nhau: {len(article_numbers)}")
    print(f"Số lỗi: {errors}")
    print(f"Số cảnh báo: {warnings}")

    print("\nCác loại vấn đề:")
    for code, count in issue_counts.most_common():
        print(f"  {code}: {count}")

    print("\nFile báo cáo:")
    print(f"- {ISSUES_CSV}")
    print(f"- {SUMMARY_TXT}")
    print(f"- {CHUNKS_CSV}")


if __name__ == "__main__":
    main()