"""Optional OCR adapters for scanned PDF pages.

OCR is an optional dependency.  The module never pretends OCR succeeded: if no
engine is installed it raises :class:`OcrUnavailable` so the batch build can
mark the document ``ocr_required`` instead of silently emitting empty chunks.

Page results are cached as ``<cache_root>/<document_id>/page-XXX.json`` so a
re-run does not pay the OCR cost twice.  The cache key includes engine, language
and DPI; a different configuration is a different cache entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import fitz  # PyMuPDF

DEFAULT_DPI = 300
CACHE_ROOT = Path("data/ocr")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TESSDATA_DIR = REPO_ROOT / "data" / "ocr" / "tessdata"


class OcrUnavailable(RuntimeError):
    """Raised when the requested OCR engine is not installed."""


class OcrError(RuntimeError):
    """Raised when OCR execution fails for a page."""


class OcrEngine(Protocol):
    name: str
    lang: str

    def ocr_image(self, png_bytes: bytes) -> List[str]:
        """Return recognized text lines for one rendered page image."""


@dataclass
class TesseractEngine:
    lang: str = "vie"
    name: str = "tesseract"
    tesseract_cmd: Optional[str] = None
    tessdata_dir: Optional[Path] = None
    psm: int = 6

    def __post_init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "pytesseract is not installed; run `pip install pytesseract` "
                "and install the Tesseract binary with the `vie` language data"
            ) from exc

        import os
        import shutil

        cmd = self.tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if not cmd:
            cmd = shutil.which("tesseract")
        if not cmd:
            for candidate in (
                Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            ):
                if candidate.exists():
                    cmd = str(candidate)
                    break
        if not cmd:
            raise OcrUnavailable(
                "Tesseract binary not found; install Tesseract or set TESSERACT_CMD"
            )
        pytesseract.pytesseract.tesseract_cmd = str(cmd)

        data_dir = self.tessdata_dir
        if data_dir is None:
            env_dir = os.environ.get("TESSDATA_DIR")
            data_dir = Path(env_dir) if env_dir else DEFAULT_TESSDATA_DIR
        if self.lang != "eng" and data_dir is not None:
            if not (Path(data_dir) / f"{self.lang}.traineddata").exists():
                raise OcrUnavailable(
                    f"missing {self.lang}.traineddata in {data_dir}; download it "
                    "from https://github.com/tesseract-ocr/tessdata_best"
                )
        self._tessdata_dir = Path(data_dir) if data_dir is not None else None
        self._config = f"--psm {self.psm}"
        if self._tessdata_dir is not None:
            # Tesseract reads TESSDATA_PREFIX; passing --tessdata-dir with
            # quotes through pytesseract breaks on Windows paths.
            os.environ["TESSDATA_PREFIX"] = str(self._tessdata_dir)

    def cache_params(self) -> Dict[str, object]:
        """Engine settings that change the OCR output for a page."""
        return {"psm": self.psm}

    def ocr_image(self, png_bytes: bytes) -> List[str]:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(png_bytes))
        text = pytesseract.image_to_string(
            image, lang=self.lang, config=self._config
        )
        return [line for line in text.splitlines() if line.strip()]


@dataclass
class EasyOcrEngine:
    """EasyOCR adapter (pip-installable, supports Vietnamese ``vi``)."""

    lang: str = "vi"
    name: str = "easyocr"
    gpu: bool = False

    def __post_init__(self) -> None:
        try:
            import easyocr  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "easyocr is not installed; run `pip install easyocr`"
            ) from exc
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(
                [self.lang], gpu=self.gpu, verbose=False
            )
        return self._reader

    def ocr_image(self, png_bytes: bytes) -> List[str]:
        import io

        import numpy as np
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        results = self._get_reader().readtext(image, detail=0, paragraph=False)
        return [str(line) for line in results if str(line).strip()]


@dataclass
class PaddleEngine:
    lang: str = "vi"
    name: str = "paddleocr"

    def __post_init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "paddleocr is not installed; run `pip install paddleocr paddlepaddle`"
            ) from exc
        self._ocr = None

    def _engine(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True, show_log=False)
        return self._ocr

    def ocr_image(self, png_bytes: bytes) -> List[str]:
        import io

        import numpy as np
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        result = self._engine().ocr(image, cls=True)
        lines: List[str] = []
        for page in result or []:
            for item in page or []:
                # PaddleOCR returns [box, (text, confidence)]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    payload = item[1]
                    if isinstance(payload, (list, tuple)) and payload:
                        lines.append(str(payload[0]))
                    else:
                        lines.append(str(payload))
        return [line for line in lines if line.strip()]


def get_engine(name: str = "auto", lang: Optional[str] = None) -> OcrEngine:
    """Instantiate an OCR engine or raise :class:`OcrUnavailable`."""
    if name == "auto":
        for candidate in ("tesseract", "easyocr", "paddleocr"):
            try:
                return get_engine(candidate, lang=lang)
            except OcrUnavailable:
                continue
        raise OcrUnavailable(
            "no OCR engine available; install pytesseract+Tesseract, easyocr, "
            "or paddleocr"
        )
    if name == "tesseract":
        return TesseractEngine(lang=lang or "vie")
    if name == "easyocr":
        return EasyOcrEngine(lang=lang or "vi")
    if name == "paddleocr":
        return PaddleEngine(lang=lang or "vi")
    raise ValueError(
        f"unknown OCR engine: {name!r} (expected tesseract/easyocr/paddleocr/auto)"
    )


def render_page(pdf_path: Path | str, page: int, dpi: int = DEFAULT_DPI) -> bytes:
    """Render a 1-based PDF page to PNG bytes."""
    with fitz.open(pdf_path) as pdf:
        if page < 1 or page > pdf.page_count:
            raise OcrError(f"page {page} out of range 1..{pdf.page_count}")
        pixmap = pdf[page - 1].get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")


def _cache_path(cache_root: Path, document_id: str, page: int, engine: OcrEngine, dpi: int) -> Path:
    return Path(cache_root) / document_id / f"page-{page:03d}.json"


def _cache_params(
    engine: OcrEngine, dpi: int, pdf_sha256: Optional[str] = None
) -> Dict[str, object]:
    """Everything that must match for a cached page to be reusable."""
    params: Dict[str, object] = {
        "engine": engine.name,
        "lang": engine.lang,
        "dpi": dpi,
    }
    extra = getattr(engine, "cache_params", None)
    if callable(extra):
        params.update(extra())
        # The identity fields always win over engine-reported values.
        params["engine"] = engine.name
        params["lang"] = engine.lang
        params["dpi"] = dpi
    if pdf_sha256:
        params["pdf_sha256"] = pdf_sha256
    return params


def pdf_fingerprint(pdf_path: Path | str) -> str:
    """Short content hash of the PDF, so a replaced file invalidates its cache."""
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _config_hash(params: Dict[str, object]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _cache_matches(payload: Dict[str, object], params: Dict[str, object]) -> bool:
    """True when a cached page was produced with the same OCR settings.

    Entries written before an engine reported a setting (e.g. ``psm``) are
    accepted: those settings were constants of the code that produced them, so
    a missing field is not a different configuration.
    """
    for key, value in params.items():
        if key not in payload:
            if key in {"psm", "pdf_sha256"}:
                continue
            return False
        if payload[key] != value:
            return False
    return True


def ocr_page(
    pdf_path: Path | str,
    page: int,
    engine: OcrEngine,
    cache_root: Path = CACHE_ROOT,
    document_id: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    force: bool = False,
    pdf_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """OCR one page with an on-disk cache keyed by the OCR configuration."""
    document_id = document_id or Path(pdf_path).stem
    cache_file = _cache_path(Path(cache_root), document_id, page, engine, dpi)
    params = _cache_params(engine, dpi, pdf_sha256 or pdf_fingerprint(pdf_path))
    config_hash = _config_hash(params)

    if cache_file.exists() and not force:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if _cache_matches(cached, params):
            return cached

    png = render_page(pdf_path, page, dpi=dpi)
    lines = engine.ocr_image(png)
    payload: Dict[str, object] = {
        "document_id": document_id,
        "page": page,
        **params,
        "config_hash": config_hash,
        "lines": lines,
        "text": "\n".join(lines),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_cached_pages(
    cache_root: Path,
    document_id: str,
    engine: OcrEngine,
    dpi: int = DEFAULT_DPI,
) -> Dict[int, List[str]]:
    """Load every cached page for a document/config pair."""
    params = _cache_params(engine, dpi)
    directory = Path(cache_root) / document_id
    pages: Dict[int, List[str]] = {}
    if not directory.exists():
        return pages
    for path in sorted(directory.glob("page-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _cache_matches(payload, params):
            continue
        page = payload.get("page")
        lines = payload.get("lines")
        if isinstance(page, int) and isinstance(lines, list):
            pages[page] = [str(line) for line in lines]
    return pages


def ocr_document(
    pdf_path: Path | str,
    engine: OcrEngine,
    document_id: Optional[str] = None,
    pages: Optional[Iterable[int]] = None,
    cache_root: Path = CACHE_ROOT,
    dpi: int = DEFAULT_DPI,
    force: bool = False,
    workers: int = 1,
) -> Dict[int, List[str]]:
    """OCR selected pages (default: every page) and return page -> lines.

    ``workers`` > 1 runs pages in parallel threads.  Each page is an
    independent Tesseract subprocess, so the GIL is not a bottleneck; the
    per-process OpenMP thread count is pinned to 1 to avoid oversubscription.
    """
    document_id = document_id or Path(pdf_path).stem
    with fitz.open(pdf_path) as pdf:
        page_count = pdf.page_count
    selected: Sequence[int] = (
        list(pages) if pages is not None else list(range(1, page_count + 1))
    )
    pdf_sha256 = pdf_fingerprint(pdf_path)

    def _one(page: int) -> tuple[int, List[str]]:
        payload = ocr_page(
            pdf_path,
            page,
            engine,
            cache_root=cache_root,
            document_id=document_id,
            dpi=dpi,
            force=force,
            pdf_sha256=pdf_sha256,
        )
        return page, [str(line) for line in payload.get("lines", [])]  # type: ignore[arg-type]

    result: Dict[int, List[str]] = {}
    if workers and workers > 1:
        import os
        from concurrent.futures import ThreadPoolExecutor

        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for page, lines in pool.map(_one, selected):
                result[page] = lines
        return result

    for page in selected:
        page, lines = _one(page)
        result[page] = lines
    return result
