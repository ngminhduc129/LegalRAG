"""Optional OCR adapters for scanned PDF pages.

OCR is an optional dependency.  The module never pretends OCR succeeded: if no
engine is installed it raises :class:`OcrUnavailable` so the batch build can
mark the document ``ocr_required`` instead of silently emitting empty chunks.

Page results are cached as
``<cache_root>/<document_id>/<config_hash>/page-XXX.json``.  ``config_hash``
covers engine, language, DPI and engine-specific settings (e.g. Tesseract
``psm``), so a different configuration is a different cache entry.  The PDF
content hash is stored in each page file and must match, so a replaced PDF
invalidates its cached pages.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import fitz  # PyMuPDF

DEFAULT_DPI = 300
REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / "data" / "ocr"
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


# ---------------------------------------------------------------------------
# Language code handling
# ---------------------------------------------------------------------------

_TESSERACT_LANGS = {"vi": "vie", "en": "eng"}
_OTHER_LANGS = {"vie": "vi", "eng": "en"}


def _normalize_lang(engine_name: str, lang: Optional[str]) -> Optional[str]:
    """Map ``vi``/``vie`` style codes to what each engine expects."""
    if not lang:
        return None
    mapping = _TESSERACT_LANGS if engine_name == "tesseract" else _OTHER_LANGS
    if engine_name == "tesseract":
        return "+".join(mapping.get(part, part) for part in lang.split("+"))
    return mapping.get(lang, lang)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


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

        langs = [part for part in self.lang.split("+") if part]

        def _missing(directory: Path) -> List[str]:
            return [
                part
                for part in langs
                if not (directory / f"{part}.traineddata").exists()
            ]

        # Explicit setting (argument or env) must be valid; the repo default is
        # only used when it actually has the needed language data, otherwise we
        # fall back to the system Tesseract installation.
        explicit = self.tessdata_dir
        if explicit is None and os.environ.get("TESSDATA_DIR"):
            explicit = Path(os.environ["TESSDATA_DIR"])

        data_dir: Optional[Path] = None
        if explicit is not None:
            data_dir = Path(explicit)
            missing = _missing(data_dir)
            if missing:
                raise OcrUnavailable(
                    f"missing {', '.join(m + '.traineddata' for m in missing)} "
                    f"in {data_dir}; download from "
                    "https://github.com/tesseract-ocr/tessdata_best"
                )
        elif DEFAULT_TESSDATA_DIR.is_dir() and not _missing(DEFAULT_TESSDATA_DIR):
            data_dir = DEFAULT_TESSDATA_DIR

        self._tessdata_dir = data_dir
        self._config = f"--psm {self.psm}"
        if data_dir is not None:
            # Tesseract reads TESSDATA_PREFIX; passing --tessdata-dir with
            # quotes through pytesseract breaks on Windows paths.  Note: this
            # is a process-wide side effect.
            os.environ["TESSDATA_PREFIX"] = str(data_dir)

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
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            import easyocr  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "easyocr is not installed; run `pip install easyocr`"
            ) from exc
        self._reader = None

    def cache_params(self) -> Dict[str, object]:
        return {"gpu": self.gpu}

    def _get_reader(self):
        with self._lock:
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
    """PaddleOCR adapter supporting both the 2.x and 3.x APIs.

    The 3.x branch could not be exercised against a real install; verify it on
    your pinned version before relying on it.
    """

    lang: str = "vi"
    name: str = "paddleocr"
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "paddleocr is not installed; run `pip install paddleocr paddlepaddle`"
            ) from exc
        self._ocr = None
        self._new_api = False

    def _engine(self):
        with self._lock:
            if self._ocr is None:
                from paddleocr import PaddleOCR

                try:  # PaddleOCR 2.x
                    self._ocr = PaddleOCR(
                        lang=self.lang, use_angle_cls=True, show_log=False
                    )
                    self._new_api = False
                except (TypeError, ValueError):  # PaddleOCR 3.x
                    self._ocr = PaddleOCR(
                        lang=self.lang, use_textline_orientation=True
                    )
                    self._new_api = True
            return self._ocr

    @staticmethod
    def _texts_from_v3(result) -> List[str]:
        lines: List[str] = []
        for item in result or []:
            data = item
            if not isinstance(data, dict) and hasattr(data, "res"):
                data = data.res
            texts = None
            if isinstance(data, dict):
                texts = data.get("rec_texts")
            elif hasattr(data, "get"):
                texts = data.get("rec_texts")
            for text in texts or []:
                lines.append(str(text))
        return lines

    def ocr_image(self, png_bytes: bytes) -> List[str]:
        import io

        import numpy as np
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        engine = self._engine()
        lines: List[str] = []
        if self._new_api:
            lines = self._texts_from_v3(engine.predict(image))
        else:
            result = engine.ocr(image, cls=True)
            for page in result or []:
                for item in page or []:
                    # PaddleOCR 2.x returns [box, (text, confidence)]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        payload = item[1]
                        if isinstance(payload, (list, tuple)) and payload:
                            lines.append(str(payload[0]))
                        else:
                            lines.append(str(payload))
        return [line for line in lines if line.strip()]


def get_engine(name: str = "auto", lang: Optional[str] = None) -> OcrEngine:
    """Instantiate an OCR engine or raise :class:`OcrUnavailable`.

    ``lang`` may be given as ``vi`` or ``vie``; it is mapped to the code each
    engine expects (Tesseract also accepts ``vie+eng``).
    """
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
        return TesseractEngine(lang=_normalize_lang(name, lang) or "vie")
    if name == "easyocr":
        return EasyOcrEngine(lang=_normalize_lang(name, lang) or "vi")
    if name == "paddleocr":
        return PaddleEngine(lang=_normalize_lang(name, lang) or "vi")
    raise ValueError(
        f"unknown OCR engine: {name!r} (expected tesseract/easyocr/paddleocr/auto)"
    )


# ---------------------------------------------------------------------------
# Rendering and cache
# ---------------------------------------------------------------------------


def render_page(pdf_path: Path | str, page: int, dpi: int = DEFAULT_DPI) -> bytes:
    """Render a 1-based PDF page to PNG bytes."""
    with fitz.open(pdf_path) as pdf:
        if page < 1 or page > pdf.page_count:
            raise OcrError(f"page {page} out of range 1..{pdf.page_count}")
        pixmap = pdf[page - 1].get_pixmap(dpi=dpi, alpha=False)
        return pixmap.tobytes("png")


def pdf_fingerprint(pdf_path: Path | str) -> str:
    """Short content hash of the PDF, so a replaced file invalidates its cache."""
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _config_params(engine: OcrEngine, dpi: int) -> Dict[str, object]:
    """OCR settings that define a cache entry (independent of the PDF)."""
    params: Dict[str, object] = {}
    extra = getattr(engine, "cache_params", None)
    if callable(extra):
        params.update(extra())
    # The identity fields always win over engine-reported values.
    params["engine"] = engine.name
    params["lang"] = engine.lang
    params["dpi"] = dpi
    return params


def _config_hash(params: Dict[str, object]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _cache_dir(cache_root: Path, document_id: str, config_hash: str) -> Path:
    return Path(cache_root) / document_id / config_hash


def _cache_path(cache_dir: Path, page: int) -> Path:
    return cache_dir / f"page-{page:03d}.json"


def _cache_matches(payload: Dict[str, object], params: Dict[str, object]) -> bool:
    """True when a cached page was produced with exactly these settings."""
    for key, value in params.items():
        if key not in payload or payload[key] != value:
            return False
    return True


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


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
    config = _config_params(engine, dpi)
    config_hash = _config_hash(config)
    params = {**config, "pdf_sha256": pdf_sha256 or pdf_fingerprint(pdf_path)}
    cache_file = _cache_path(
        _cache_dir(Path(cache_root), document_id, config_hash), page
    )

    if cache_file.exists() and not force:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None  # corrupt or partial file: re-run OCR
        if isinstance(cached, dict) and _cache_matches(cached, params):
            return cached

    png = render_page(pdf_path, page, dpi=dpi)
    try:
        lines = engine.ocr_image(png)
    except (OcrError, OcrUnavailable):
        raise
    except Exception as exc:
        raise OcrError(
            f"{engine.name} failed on {document_id} page {page}: {exc}"
        ) from exc

    payload: Dict[str, object] = {
        "document_id": document_id,
        "page": page,
        **params,
        "config_hash": config_hash,
        "lines": lines,
        "text": "\n".join(lines),
    }
    _atomic_write_json(cache_file, payload)
    return payload


def load_cached_pages(
    cache_root: Path,
    document_id: str,
    engine: OcrEngine,
    dpi: int = DEFAULT_DPI,
    pdf_path: Optional[Path | str] = None,
) -> Dict[int, List[str]]:
    """Load every cached page for a document/config pair.

    Pass ``pdf_path`` to also reject pages cached from a different version of
    the PDF.
    """
    config = _config_params(engine, dpi)
    params = dict(config)
    if pdf_path is not None:
        params["pdf_sha256"] = pdf_fingerprint(pdf_path)
    directory = _cache_dir(Path(cache_root), document_id, _config_hash(config))
    pages: Dict[int, List[str]] = {}
    if not directory.exists():
        return pages
    for path in sorted(directory.glob("page-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _cache_matches(payload, params):
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

    ``workers`` > 1 runs pages in parallel threads.  Each Tesseract page is an
    independent subprocess, so the GIL is not a bottleneck; the per-process
    OpenMP thread count is pinned to 1 to avoid oversubscription.  EasyOCR and
    PaddleOCR share one in-process model, so prefer ``workers=1`` for them.
    """
    document_id = document_id or Path(pdf_path).stem
    with fitz.open(pdf_path) as pdf:
        page_count = pdf.page_count
    selected: Sequence[int] = (
        list(dict.fromkeys(pages)) if pages is not None else list(range(1, page_count + 1))
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