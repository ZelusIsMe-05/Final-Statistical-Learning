import re
from typing import List, Optional

import fitz

PYMUPDF_AVAILABLE = True

MAX_CHUNK_TOKENS = 700
MAX_CHUNK_CHARS = 3000   # tăng từ 2000 → chunk giàu nội dung hơn, ít chunk hơn
MIN_CHUNK_CHARS = 100
OVERLAP_CHARS = 200      # tăng từ 150 → đảm bảo liên tục giữa các chunk

HEADING_PATTERNS = [
    r"^(Chương|Chapter)\s+[\dIVXivx]+[\.:]?\s+.+",
    r"^(Phần|Part)\s+[\dIVXivx]+[\.:]?\s+.+",
    r"^(Mục|Section)\s+[\d\.]+[\.:]?\s+.+",
    r"^\d+(?:\.\d+)*[\.\s].+",
    r"^[IVX]+\.\s+.+",
    r"^[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẶẦẨẬẮẴẰẸỀỂỆỈỊỌỒỔỘỜỞỢỤỪỬỰỲỴ\s]{3,}:?\s*$",
]

HEADING_RE = [re.compile(pattern, re.MULTILINE) for pattern in HEADING_PATTERNS]
BULLET_PREFIX_RE = re.compile(
    r"^\s*("
    r"[\u2022\u2023\u25e6\u2219\-\*\u2013\u2014]"
    r"|"
    r"\d+[\.\)]"
    r"|"
    r"[A-Za-z][\.\)]"
    r")\s+"
)


def is_heading(line: str) -> bool:
    """Return whether a line looks like a document heading."""

    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in HEADING_RE)


def table_to_text(rows: List[List[str]], headers: Optional[List[str]] = None) -> str:
    """Convert a table into Vietnamese descriptive sentences."""

    if not rows:
        return ""

    col_count = max(len(row) for row in rows)
    if not headers or len(headers) < col_count:
        headers = [f"Cột {i + 1}" for i in range(col_count)]

    sentences = []
    for row in rows:
        parts = []
        for i, cell in enumerate(row):
            cell = str(cell).strip()
            if not cell or cell in ("-", "N/A", "n/a"):
                continue
            label = headers[i] if i < len(headers) else f"Cột {i + 1}"
            parts.append(f"{label} là {cell}")
        if parts:
            sentences.append(", ".join(parts) + ".")

    return " ".join(sentences)


def _parse_markdown_table(block: str) -> str:
    """Convert a Markdown table block into prose."""

    lines = [line.strip() for line in block.strip().splitlines()]
    separator_re = re.compile(r"^[\|\-:\s]+$")
    data_lines = [
        line for line in lines
        if line.startswith("|") and not separator_re.match(line)
    ]

    if not data_lines:
        return block

    def split_row(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    headers = split_row(data_lines[0])
    rows = [split_row(line) for line in data_lines[1:]]
    return table_to_text(rows, headers)


def convert_text_tables(text: str) -> str:
    """Convert Markdown and simple ASCII tables into prose before chunking."""

    lines = text.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^\s*\|", line):
            table_block = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                table_block.append(lines[i])
                i += 1
            result.append(_parse_markdown_table("\n".join(table_block)))
            continue

        if re.match(r"^[\+\|][-=\+\|]+[\+\|]\s*$", line.strip()):
            i += 1
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


def _normalize_repeated_pdf_line(text: str) -> str:
    """Normalize a PDF block so repeated headers and footers can be detected."""

    text = " ".join(text.split()).lower()
    text = re.sub(r"\d+", "#", text)
    return text.strip()


def _is_page_number(text: str) -> bool:
    """Detect simple footer page-number blocks."""

    return bool(re.fullmatch(r"(page\s*)?\d+(\s*/\s*\d+)?", text.strip(), flags=re.IGNORECASE))


def _find_repeated_margin_text(doc) -> set:
    """Find repeated text in page margins, which is usually header/footer text."""

    page_count = len(doc)
    if page_count < 2:
        return set()

    threshold = max(2, int(page_count * 0.5))
    occurrences = {}

    for page in doc:
        page_height = page.rect.height
        top_limit = page_height * 0.12
        bottom_limit = page_height * 0.88
        seen_on_page = set()

        for block in page.get_text("blocks", sort=True):
            if block[6] != 0:
                continue
            _, by0, _, by1 = block[:4]
            if by1 > top_limit and by0 < bottom_limit:
                continue
            normalized = _normalize_repeated_pdf_line(block[4])
            if normalized:
                seen_on_page.add(normalized)

        for normalized in seen_on_page:
            occurrences[normalized] = occurrences.get(normalized, 0) + 1

    return {text for text, count in occurrences.items() if count >= threshold}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract readable text from a PDF while skipping repeated margins."""

    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF chưa được cài. Chạy: pip install pymupdf")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = []
    repeated_margin_text = _find_repeated_margin_text(doc)

    for page in doc:
        parts = []
        page_height = page.rect.height
        top_limit = page_height * 0.12
        bottom_limit = page_height * 0.88
        table_rects = []

        try:
            tables = page.find_tables()
            for table in tables:
                rows = table.extract()
                if not rows:
                    continue

                headers = [
                    str(cell).strip() if cell else f"Cột {i + 1}"
                    for i, cell in enumerate(rows[0])
                ]
                data_rows = rows[1:] if len(rows) > 1 else rows
                converted = table_to_text(
                    [[str(cell) if cell is not None else "" for cell in row] for row in data_rows],
                    headers,
                )
                if converted:
                    parts.append(converted)
                table_rects.append(table.bbox)
        except Exception:
            pass

        for block in page.get_text("blocks", sort=True):
            if block[6] != 0:
                continue

            bx0, by0, bx1, by1 = block[:4]
            in_table = any(
                bx0 >= tx0 - 2 and by0 >= ty0 - 2 and bx1 <= tx1 + 2 and by1 <= ty1 + 2
                for tx0, ty0, tx1, ty1 in table_rects
            )
            if in_table:
                continue

            text = block[4].strip()
            normalized = _normalize_repeated_pdf_line(text)
            is_margin_block = by1 <= top_limit or by0 >= bottom_limit
            if is_margin_block and (normalized in repeated_margin_text or _is_page_number(text)):
                continue
            if text:
                parts.append(text)

        if parts:
            page_texts.append("\n".join(parts))

    doc.close()
    return "\n".join(page_texts)


def clean_text(text: str) -> str:
    """Clean extracted text while preserving paragraph and bullet boundaries."""

    text = convert_text_tables(text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    raw_lines = text.split("\n")
    merged: List[str] = []
    pending_bullet = ""

    for line in raw_lines:
        stripped = line.strip()
        if re.fullmatch(r"[\u2022\u2023\u25e6\-\*\u2013\u2014\u2219]+", stripped):
            pending_bullet = stripped + " "
            continue
        if stripped:
            merged.append(pending_bullet + stripped)
            pending_bullet = ""
        elif not pending_bullet:
            merged.append("")

    text = "\n".join(merged)

    sentence_end = re.compile(r"[.!?:;]\s*$")
    bullet_start_re = re.compile(r"^[\u2022\u2023\u25e6\-\*\u2013\u2014\u2219]\s|^\d+[\.\)]\s")
    heading_start_re = re.compile("|".join(HEADING_PATTERNS))
    lines = text.split("\n")

    pass1: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if (
            cur
            and not sentence_end.search(cur)
            and not heading_start_re.match(cur)
            and not bullet_start_re.match(cur)
            and i + 1 < len(lines)
        ):
            nxt = lines[i + 1].strip()
            if nxt and not bullet_start_re.match(nxt) and not heading_start_re.match(nxt):
                pass1.append(cur + " " + nxt)
                i += 2
                continue
        pass1.append(lines[i])
        i += 1

    pass2: List[str] = []
    i = 0
    while i < len(pass1):
        cur = pass1[i].strip()
        if cur and not sentence_end.search(cur) and not heading_start_re.match(cur):
            j = i + 1
            blanks = 0
            while j < len(pass1) and not pass1[j].strip():
                blanks += 1
                j += 1
            if j < len(pass1):
                nxt = pass1[j].strip()
                is_new_bullet = bool(bullet_start_re.match(nxt))
                if blanks <= 2 and nxt and not is_new_bullet and not heading_start_re.match(nxt):
                    pass2.append(cur + " " + nxt)
                    i = j + 1
                    continue
        pass2.append(pass1[i])
        i += 1

    text = "\n".join(pass2)
    text = text.replace("_", " ")
    text = re.sub(r"\s+([,\.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.split("\n"):
        cleaned_lines.append(" ".join(line.split()) if line.strip() else "")

    return "\n".join(cleaned_lines).strip()


def chunk_by_headings(text: str) -> List[dict]:
    """Split cleaned text into heading-scoped sections."""

    sections = []
    lines = text.split("\n")
    current_heading = "Mở đầu"
    current_content: List[str] = []

    for line in lines:
        if is_heading(line):
            content = "\n".join(current_content).strip()
            if content:
                sections.append({"heading": current_heading, "content": content})
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line.rstrip())

    content = "\n".join(current_content).strip()
    if content:
        sections.append({"heading": current_heading, "content": content})

    return sections


def _split_section_into_paragraphs(full_section: str) -> List[str]:
    """Split a heading section into paragraph and bullet chunks."""

    if not full_section.strip():
        return []

    bullet_pattern = re.compile(
        r"^([\u2022\u2023\u25e6\u2219\-\*\u2013\u2014]\s+|\d+[\.\)]\s+)",
        re.MULTILINE,
    )
    sentence_end = re.compile(r"[.!?]\s*$")
    chunks: List[str] = []
    current_chunk: List[str] = []
    is_bullet_group = False

    def flush_current() -> None:
        nonlocal current_chunk, is_bullet_group
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(" ".join(chunk_text.split()))
        current_chunk = []
        is_bullet_group = False

    for line in full_section.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current_chunk:
                flush_current()
            continue

        is_bullet = bool(bullet_pattern.match(stripped))
        if is_bullet:
            if current_chunk:
                flush_current()
            current_chunk = [stripped]
            is_bullet_group = True
            continue

        if is_bullet_group:
            if current_chunk and sentence_end.search(current_chunk[-1]):
                flush_current()
                current_chunk = [stripped]
            else:
                current_chunk.append(stripped)
            continue

        if current_chunk and sentence_end.search(current_chunk[-1]):
            flush_current()
            current_chunk = [stripped]
        elif current_chunk:
            current_chunk.append(stripped)
        else:
            current_chunk = [stripped]

    if current_chunk:
        flush_current()

    return chunks


def extract_bullet_prefix(text: str) -> str:
    """Return the leading bullet or list marker for a chunk."""

    stripped = text.strip()
    first_line = stripped.split("\n", 1)[0] if stripped else ""
    match = BULLET_PREFIX_RE.match(first_line)
    return match.group(1) if match else ""


def chunk_text_smart_with_headings(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> tuple:
    """Split text into chunks and keep heading/bullet metadata aligned."""

    del overlap

    if not text:
        return [], [], []

    sections = chunk_by_headings(text)
    chunks: List[str] = []
    headings: List[str] = []
    bullets: List[str] = []

    for section in sections:
        heading = section["heading"]
        paragraphs = _split_section_into_paragraphs(section["content"])

        for paragraph in paragraphs:
            para_clean = paragraph.strip()
            bullet = extract_bullet_prefix(para_clean)
            if para_clean and (len(para_clean) >= MIN_CHUNK_CHARS or bullet):
                chunks.append(para_clean)
                headings.append(heading)
                bullets.append(bullet)

    if not chunks:
        fallback = text[:max_chars]
        return [fallback], ["Nội dung"], [extract_bullet_prefix(fallback)]

    return chunks, headings, bullets


def process_document(
    content: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    max_chars: int = MAX_CHUNK_CHARS,
) -> dict:
    """Process text or PDF input into cleaned text, chunks, and metadata."""

    raw_text = ""
    source = "text"

    if pdf_bytes:
        raw_text = extract_text_from_pdf(pdf_bytes)
        source = "pdf"
    elif content:
        raw_text = content
    else:
        return {
            "full_text": "",
            "chunks": [],
            "chunk_headings": [],
            "chunk_bullets": [],
            "chunk_count": 0,
            "source": source,
            "error": "Không có nội dung đầu vào",
        }

    clean = clean_text(raw_text)
    chunks, headings, bullets = chunk_text_smart_with_headings(clean, max_chars)

    return {
        "full_text": clean,
        "chunks": chunks,
        "chunk_headings": headings,
        "chunk_bullets": bullets,
        "chunk_count": len(chunks),
        "source": source,
    }
