
import re
import json
import csv
from pathlib import Path

import fitz  # PyMuPDF


# =========================================================
# 1. CAU HINH
# =========================================================

PDF_PATH = Path(
    r"E:\LegalRAG\data\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf"
)

OUTPUT_DIR = Path(r"E:\LegalRAG\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUTPUT_DIR / "chunks.jsonl"
CSV_PATH = OUTPUT_DIR / "chunks_preview.csv"

DOCUMENT_NAME = "Bộ luật Lao động"
DOCUMENT_NUMBER = "125/VBHN-VPQH"

DEBUG = True


# =========================================================
# 2. REGEX NHAN DIEN CAU TRUC PHAP LUAT
# =========================================================

CHAPTER_RE = re.compile(
    r"^\s*Chương\s+([IVXLCDM]+|\d+)\s*$",
    re.IGNORECASE
)

ARTICLE_RE = re.compile(
    r"^\s*Điều\s+(\d+)\s*\.?\s*(.*?)\s*$",
    re.IGNORECASE
)

CLAUSE_RE = re.compile(
    r"^\s*(\d+)\s*\.\s+(.+?)\s*$"
)

POINT_RE = re.compile(
    r"^\s*([a-zđ])\s*\)\s*(.*?)\s*$",
    re.IGNORECASE
)


# =========================================================
# 3. CAC HAM HO TRO
# =========================================================

def clean_line(text: str) -> str:
    """Chuan hoa khoang trang, khong sua noi dung phap ly."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    # Chuan hoa tab va nhieu khoang trang lien tiep
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def is_page_number(text: str) -> bool:
    """
    Nhan dien dong chi chua so trang.
    Vi du: 1, 2, 3, 10.
    """

    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def extract_pdf_lines(pdf_path: Path):
    """
    Trich xuat van ban bang PyMuPDF.
    Loai dong so trang dung rieng.
    Luu so trang PDF cho tung dong.
    """

    lines = []

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1

            page_text = page.get_text("text", sort=True)

            for raw_line in page_text.splitlines():
                line = clean_line(raw_line)

                if not line:
                    continue

                # Bo qua dong chi chua so trang
                if is_page_number(line):
                    continue

                lines.append({
                    "text": line,
                    "page": page_num
                })

    return lines


def join_text(parts):
    """
    Ghep cac dong thanh noi dung chunk.
    """

    return "\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    ).strip()


def make_chunk(
    chapter,
    chapter_title,
    article,
    article_title,
    clause,
    point,
    content,
    page_start,
    page_end,
    chunk_type
):
    """Tao mot ban ghi chunk thong nhat."""

    return {
        "document_name": DOCUMENT_NAME,
        "document_number": DOCUMENT_NUMBER,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "article": article,
        "article_title": article_title,
        "clause": clause,
        "point": point,
        "chunk_type": chunk_type,
        "content": content.strip(),
        "page_start": page_start,
        "page_end": page_end,
    }


# =========================================================
# 4. PARSE CHUONG, DIEU, KHOAN, DIEM
# =========================================================

def parse_legal_structure(lines):
    """
    Chien luoc chunking:

    - Chương: cap nhat metadata chuong.
    - Dieu: bat dau mot dieu moi.
    - Khoan: bat dau mot khoan moi.
    - Diem: bat dau mot diem thuoc khoan hien tai.
    - Dong tiep theo duoc noi vao don vi dang mo.
    - Noi dung qua trang van duoc noi tiep.
    - Tieu de Dieu lap lai khong tao Dieu moi neu
      van la cung mot Dieu va cung tieu de.
    """

    chunks = []

    chapter = None
    chapter_title = None

    article = None
    article_title = None
    article_page = None

    article_intro = []
    article_intro_pages = []

    clause = None
    clause_parts = []
    clause_pages = []

    clause_intro = []
    clause_intro_pages = []

    point = None
    point_parts = []
    point_pages = []

    clause_has_points = False

    # -----------------------------------------------------
    # XUAT ARTICLE INTRO
    # -----------------------------------------------------

    def emit_article_intro():
        nonlocal article_intro, article_intro_pages

        content = join_text(article_intro)

        if content and article is not None:
            chunks.append(
                make_chunk(
                    chapter=chapter,
                    chapter_title=chapter_title,
                    article=article,
                    article_title=article_title,
                    clause=None,
                    point=None,
                    content=content,
                    page_start=min(article_intro_pages),
                    page_end=max(article_intro_pages),
                    chunk_type="article_intro"
                )
            )

        article_intro = []
        article_intro_pages = []

    # -----------------------------------------------------
    # XUAT POINT
    # -----------------------------------------------------

    def emit_point():
        nonlocal point, point_parts, point_pages

        content = join_text(point_parts)

        if content and article is not None:
            chunks.append(
                make_chunk(
                    chapter=chapter,
                    chapter_title=chapter_title,
                    article=article,
                    article_title=article_title,
                    clause=clause,
                    point=point,
                    content=content,
                    page_start=min(point_pages),
                    page_end=max(point_pages),
                    chunk_type="point"
                )
            )

        point = None
        point_parts = []
        point_pages = []

    # -----------------------------------------------------
    # XUAT CLAUSE INTRO
    # -----------------------------------------------------

    def emit_clause_intro():
        nonlocal clause_intro, clause_intro_pages

        content = join_text(clause_intro)

        if content and article is not None:
            chunks.append(
                make_chunk(
                    chapter=chapter,
                    chapter_title=chapter_title,
                    article=article,
                    article_title=article_title,
                    clause=clause,
                    point=None,
                    content=content,
                    page_start=min(clause_intro_pages),
                    page_end=max(clause_intro_pages),
                    chunk_type="clause_intro"
                )
            )

        clause_intro = []
        clause_intro_pages = []

    # -----------------------------------------------------
    # XUAT CLAUSE
    # -----------------------------------------------------

    def emit_clause():
        nonlocal clause, clause_parts, clause_pages
        nonlocal clause_has_points

        if clause is not None and not clause_has_points:
            content = join_text(clause_parts)

            if content:
                chunks.append(
                    make_chunk(
                        chapter=chapter,
                        chapter_title=chapter_title,
                        article=article,
                        article_title=article_title,
                        clause=clause,
                        point=None,
                        content=content,
                        page_start=min(clause_pages),
                        page_end=max(clause_pages),
                        chunk_type="clause"
                    )
                )

        elif clause is not None and clause_has_points:
            emit_clause_intro()

        clause = None
        clause_parts = []
        clause_pages = []
        clause_has_points = False

    # -----------------------------------------------------
    # DONG DIEU HIEN TAI
    # -----------------------------------------------------

    def close_article():
        nonlocal article, article_title, article_page

        emit_point()
        emit_clause()
        emit_article_intro()

        article = None
        article_title = None
        article_page = None

    # =====================================================
    # VONG LAP PARSE
    # =====================================================

    i = 0

    while i < len(lines):
        item = lines[i]

        text = item["text"]
        page = item["page"]

        # -------------------------------------------------
        # CHUONG
        # -------------------------------------------------

        chapter_match = CHAPTER_RE.match(text)

        if chapter_match:
            close_article()

            chapter = f"Chương {chapter_match.group(1)}"
            chapter_title = None

            # Thu lay ten chuong tu dong tiep theo
            if i + 1 < len(lines):
                next_text = lines[i + 1]["text"]

                if (
                    not CHAPTER_RE.match(next_text)
                    and not ARTICLE_RE.match(next_text)
                    and not CLAUSE_RE.match(next_text)
                ):
                    chapter_title = next_text
                    i += 1

            i += 1
            continue

        # -------------------------------------------------
        # DIEU
        # -------------------------------------------------

        article_match = ARTICLE_RE.match(text)

        if article_match:
            article_number = article_match.group(1)
            article_title_text = article_match.group(2).strip()

            candidate_article = f"Điều {article_number}"

            # Xu ly tieu de Dieu bi lap o dau trang.
            # Neu cung so Dieu va cung tieu de thi bo qua
            # dong lap, khong dong chunk dang mo.
            same_article = (
                article is not None
                and article == candidate_article
                and (
                    article_title is None
                    or not article_title_text
                    or article_title.strip().casefold()
                    == article_title_text.strip().casefold()
                )
            )

            if same_article:
                i += 1
                continue

            close_article()

            article = candidate_article
            article_title = article_title_text or None
            article_page = page

            i += 1
            continue

        # -------------------------------------------------
        # BO QUA NOI DUNG TRUOC DIEU DAU TIEN
        # -------------------------------------------------

        if article is None:
            i += 1
            continue

        # -------------------------------------------------
        # KHOAN
        # -------------------------------------------------

        clause_match = CLAUSE_RE.match(text)

        if clause_match:
            emit_point()
            emit_clause()

            clause = clause_match.group(1)
            clause_parts = [clause_match.group(2)]
            clause_pages = [page]

            clause_has_points = False

            i += 1
            continue

        # -------------------------------------------------
        # DIEM
        # -------------------------------------------------

        point_match = POINT_RE.match(text)

        if point_match and clause is not None:
            if not clause_has_points:
                # Chuyen phan dan cua khoan sang clause_intro
                clause_intro = list(clause_parts)
                clause_intro_pages = list(clause_pages)

                clause_parts = []
                clause_pages = []

                clause_has_points = True

            emit_point()

            point = point_match.group(1).lower()
            point_parts = [point_match.group(2)]
            point_pages = [page]

            i += 1
            continue

        # -------------------------------------------------
        # NOI DUNG THUONG / DONG TIEP THEO
        # -------------------------------------------------

        if point is not None:
            point_parts.append(text)
            point_pages.append(page)

        elif clause is not None:
            clause_parts.append(text)
            clause_pages.append(page)

        else:
            article_intro.append(text)
            article_intro_pages.append(page)

        i += 1

    # Dong cau truc cuoi cung
    close_article()

    return chunks


# =========================================================
# 5. GHI JSONL VA CSV
# =========================================================

def save_jsonl(chunks, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )


def save_csv(chunks, output_path):
    if not chunks:
        print("Khong co chunk de ghi CSV.")
        return

    fieldnames = list(chunks[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(chunks)


# =========================================================
# 6. KIEM TRA CHUNK
# =========================================================

def validate_chunks(chunks):
    """
    Kiem tra mot so loi co the gap:
    - Chunk rong.
    - Noi dung chi chua so trang.
    - Chunk co noi dung bat thuong.
    """

    empty_count = 0
    page_number_count = 0

    for index, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "").strip()

        if not content:
            empty_count += 1
            print(f"[CAN KIEM TRA] Chunk {index} rong.")

        if is_page_number(content):
            page_number_count += 1
            print(
                f"[CAN KIEM TRA] Chunk {index} "
                f"chi chua so: {content}"
            )

    print("\n===== BAO CAO KIEM TRA =====")
    print(f"Tong so chunk: {len(chunks)}")
    print(f"Chunk rong: {empty_count}")
    print(f"Chunk chi chua so: {page_number_count}")


# =========================================================
# 7. CHAY CHUONG TRINH
# =========================================================

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file PDF: {PDF_PATH}"
        )

    print("Dang trich xuat PDF...")

    lines = extract_pdf_lines(PDF_PATH)

    print(f"So dong trich xuat: {len(lines)}")

    print("Dang phan tich cau truc phap luat...")

    chunks = parse_legal_structure(lines)

    print(f"Tong so chunk: {len(chunks)}")

    validate_chunks(chunks)

    save_jsonl(chunks, JSONL_PATH)
    save_csv(chunks, CSV_PATH)

    print("\nDa luu ket qua:")
    print(f"JSONL: {JSONL_PATH}")
    print(f"CSV:   {CSV_PATH}")

    if DEBUG:
        print("\n===== 5 CHUNK DAU TIEN =====")

        for chunk in chunks[:5]:
            print("-" * 70)

            print(
                f"{chunk['article']} | "
                f"Khoan {chunk['clause']} | "
                f"Diem {chunk['point']}"
            )

            print("Loai:", chunk["chunk_type"])

            print(
                "Trang:",
                chunk["page_start"],
                "-",
                chunk["page_end"]
            )

            print("Noi dung:", chunk["content"][:300])


if __name__ == "__main__":
    main()