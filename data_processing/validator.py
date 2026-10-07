import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter, defaultdict


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    chunk_id: int = -1
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    @property
    def infos(self) -> int:
        return sum(1 for i in self.issues if i.severity == "INFO")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Total chunks: {self.stats.get('total_chunks', 0)}",
            f"Articles found: {self.stats.get('articles_found', 0)}",
            f"Expected articles: {self.stats.get('expected_articles', 0)}",
            f"Missing articles: {len(self.stats.get('missing_articles', []))}",
            f"Errors: {self.errors}",
            f"Warnings: {self.warnings}",
            f"Infos: {self.infos}",
            "",
            "Issues by type:"
        ]

        by_code = Counter(i.code for i in self.issues)
        for code, count in by_code.most_common():
            lines.append(f"  {code}: {count}")

        if self.stats.get("missing_articles"):
            lines.append("")
            lines.append("Missing articles:")
            for art in sorted(self.stats["missing_articles"]):
                lines.append(f"  {art}")

        return "\n".join(lines)


def is_suspicious_ending(content: str) -> bool:
    lines = content.strip().splitlines()
    if not lines:
        return False
    last_line = lines[-1].strip()
    return bool(re.fullmatch(r"\d{1,3}", last_line))


def find_legal_references(text: str) -> List[str]:
    refs = []
    patterns = [
        r"Điều\s+\d+(?:\s+của\s+\w+)?",
        r"khoản\s+\d+\s+Điều\s+\d+",
        r"theo\s+quy\s+định\s+tại\s+[Điều|khoản]\s+\d+",
        r"quy\s+định\s+tại\s+[Điều|khoản]\s+\d+",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        refs.extend(matches)
    return refs


def ends_mid_sentence(content: str) -> bool:
    content = content.strip()
    if not content:
        return False
    last_char = content[-1]
    if last_char in ".!?":
        return False
    if re.fullmatch(r"\d{1,3}", content.strip().splitlines()[-1].strip()):
        return False
    return True


def is_short_title(title: str) -> bool:
    if not title:
        return True
    return len(title.strip()) < 5


def is_truncated_title(title: str) -> bool:
    if not title:
        return False
    title = title.strip()
    truncated_endings = (
        "của", "tại", "theo", "quy định", "khoản", "điều",
        "mục", "chương", "và", "hoặc", "các", "những",
        "trong", "theo", "theo quy", "căn cứ", "dựa trên"
    )
    lower = title.lower()
    return any(lower.endswith(ending) for ending in truncated_endings)


class PostParseValidator:
    EXPECTED_ARTICLE_COUNT = 220

    def __init__(self):
        self.issues = []
        self.stats = {}

    def validate(self, chunks: List[Dict[str, Any]]) -> ValidationReport:
        self.issues = []
        self.stats = {
            "total_chunks": len(chunks),
            "articles_found": 0,
            "expected_articles": self.EXPECTED_ARTICLE_COUNT,
            "missing_articles": [],
            "chapters": set(),
            "sections": set(),
            "clauses": 0,
            "points": 0,
        }

        article_numbers = set()
        article_to_chunks = defaultdict(list)
        seen_structure = set()

        for idx, chunk in enumerate(chunks):
            chunk["_chunk_id"] = idx + 1
            self._validate_chunk(chunk, idx + 1)

            article = chunk.get("article")
            if article:
                match = re.match(r"Điều\s+(\d+)", article)
                if match:
                    num = int(match.group(1))
                    article_numbers.add(num)
                    article_to_chunks[article].append(chunk)

            chapter = chunk.get("chapter")
            if chapter:
                self.stats["chapters"].add(chapter)

            section = chunk.get("section")
            if section:
                self.stats["sections"].add(section)

            if chunk.get("clause"):
                self.stats["clauses"] += 1
            if chunk.get("point"):
                self.stats["points"] += 1

            struct_key = (
                chunk.get("article"),
                chunk.get("clause"),
                chunk.get("point"),
                chunk.get("chunk_type")
            )
            if struct_key in seen_structure:
                self.issues.append(ValidationIssue(
                    severity="WARNING",
                    code="DUPLICATE_STRUCTURE_KEY",
                    message=f"Duplicate structure key: {struct_key}",
                    chunk_id=idx + 1,
                    article=article,
                    clause=chunk.get("clause"),
                    point=chunk.get("point"),
                ))
            seen_structure.add(struct_key)

        expected = set(range(1, self.EXPECTED_ARTICLE_COUNT + 1))
        missing = expected - article_numbers
        self.stats["missing_articles"] = [f"Điều {n}" for n in sorted(missing)]
        self.stats["articles_found"] = len(article_numbers)

        self._cross_validate_chunks(chunks, article_to_chunks)

        return ValidationReport(
            issues=self.issues,
            stats={k: (sorted(v) if isinstance(v, set) else v) for k, v in self.stats.items()}
        )

    def _validate_chunk(self, chunk: Dict[str, Any], chunk_id: int):
        article = chunk.get("article")
        article_title = chunk.get("article_title")
        content = chunk.get("content", "")
        chunk_type = chunk.get("chunk_type")

        if is_short_title(article_title):
            self.issues.append(ValidationIssue(
                severity="WARNING",
                code="SHORT_ARTICLE_TITLE",
                message=f"Article title too short: '{article_title}'",
                chunk_id=chunk_id,
                article=article,
            ))

        if is_truncated_title(article_title):
            self.issues.append(ValidationIssue(
                severity="WARNING",
                code="TRUNCATED_ARTICLE_TITLE",
                message=f"Article title appears truncated: '{article_title}'",
                chunk_id=chunk_id,
                article=article,
            ))

        if ends_mid_sentence(content):
            self.issues.append(ValidationIssue(
                severity="WARNING",
                code="CONTENT_ENDS_MID_SENTENCE",
                message="Content may end mid-sentence",
                chunk_id=chunk_id,
                article=article,
                clause=chunk.get("clause"),
                point=chunk.get("point"),
            ))

        if is_suspicious_ending(content):
            self.issues.append(ValidationIssue(
                severity="WARNING",
                code="SUSPICIOUS_NUMBER_ENDING",
                message="Content ends with standalone number (possible page number)",
                chunk_id=chunk_id,
                article=article,
                clause=chunk.get("clause"),
                point=chunk.get("point"),
            ))

        refs = find_legal_references(content)
        if refs:
            chunk["has_legal_refs"] = True
            chunk["legal_refs"] = refs

        required = ["document_name", "document_number", "article", "chunk_type", "page_start", "page_end"]
        for field in required:
            if chunk.get(field) is None or chunk.get(field) == "":
                self.issues.append(ValidationIssue(
                    severity="ERROR",
                    code="MISSING_METADATA",
                    message=f"Missing required field: {field}",
                    chunk_id=chunk_id,
                    article=article,
                ))

        try:
            ps = int(chunk["page_start"])
            pe = int(chunk["page_end"])
            if ps < 1 or pe < ps:
                self.issues.append(ValidationIssue(
                    severity="ERROR",
                    code="INVALID_PAGE_RANGE",
                    message=f"Invalid page range: {ps}-{pe}",
                    chunk_id=chunk_id,
                    article=article,
                ))
        except (KeyError, TypeError, ValueError):
            self.issues.append(ValidationIssue(
                severity="ERROR",
                code="INVALID_PAGE_METADATA",
                message="page_start/page_end not valid",
                chunk_id=chunk_id,
                article=article,
            ))

    def _cross_validate_chunks(self, chunks: List[Dict], article_to_chunks: Dict):
        for article, article_chunks in article_to_chunks.items():
            has_intro = any(c["chunk_type"] == "article_intro" for c in article_chunks)
            has_clauses = any(c.get("clause") for c in article_chunks)

            if not has_intro and not has_clauses:
                self.issues.append(ValidationIssue(
                    severity="INFO",
                    code="ARTICLE_NO_CONTENT",
                    message=f"Article has no intro or clauses",
                    chunk_id=article_chunks[0]["_chunk_id"],
                    article=article,
                ))

            clause_nums = []
            for c in article_chunks:
                if c.get("clause"):
                    try:
                        clause_nums.append(int(c["clause"]))
                    except ValueError:
                        pass

            if clause_nums:
                clause_nums.sort()
                for i, n in enumerate(clause_nums):
                    if i > 0 and n != clause_nums[i-1] + 1:
                        self.issues.append(ValidationIssue(
                            severity="WARNING",
                            code="CLAUSE_NUMBER_GAP",
                            message=f"Clause number gap: {clause_nums[i-1]} -> {n}",
                            chunk_id=c["_chunk_id"],
                            article=article,
                            clause=str(n),
                        ))


def validate_chunks(chunks: List[Dict[str, Any]]) -> ValidationReport:
    validator = PostParseValidator()
    return validator.validate(chunks)