"""Đọc tài liệu (PDF/TXT/MD), làm sạch, chia chunk -> data/chunks.jsonl"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
import yaml

def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text)

def read_pdf(path: Path):
    import fitz  # pymupdf
    pages = []
    with fitz.open(path) as doc:
        for p in doc:
            pages.append(norm(p.get_text()))
    # bỏ dòng lặp ở nhiều trang (đầu trang/chân trang)
    cnt = Counter(l.strip() for t in pages for l in set(t.splitlines()) if l.strip())
    repeated = {l for l, c in cnt.items() if len(pages) >= 4 and c > 0.5 * len(pages)}
    return ["\n".join(l for l in t.splitlines() if l.strip() not in repeated) for t in pages]

def read_docx(path: Path):
    """Đọc .docx theo đúng thứ tự đoạn văn và bảng. Word không có số trang cố định nên trả về một 'trang' duy nhất."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(path)
    lines = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = Paragraph(child, doc).text.strip()
            if t:
                lines.append(t)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = []
                for c in row.cells:  # ô gộp có thể lặp lại, bỏ ô trùng liên tiếp
                    t = c.text.strip().replace("\n", " ")
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    lines.append(" | ".join(cells))
    return [norm("\n".join(lines))]

def read_any(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return [norm(path.read_text(encoding="utf-8"))]

def chunk_pages(pages, size, overlap):
    """Chia theo số từ (âm tiết tiếng Việt cách nhau bằng khoảng trắng), giữ số trang bắt đầu."""
    words, page_of = [], []
    for i, t in enumerate(pages, 1):
        for w in t.split():
            words.append(w); page_of.append(i)
    step = max(1, size - overlap)
    out = []
    for s in range(0, len(words), step):
        piece = words[s:s + size]
        if len(piece) < size // 4 and out:  # đoạn đuôi quá ngắn thì bỏ
            break
        out.append({"text": " ".join(piece), "page": page_of[s]})
        if s + size >= len(words):
            break
    return out

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    raw = Path(cfg["raw_dir"])
    files = sorted(p for p in raw.glob("*")
                   if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"} and not p.name.startswith("~$"))
    old_doc = [p.name for p in raw.glob("*.doc")]
    if old_doc:
        print("Bỏ qua file .doc (định dạng cũ), hãy lưu thành .docx hoặc PDF:", ", ".join(old_doc))
    if not files:
        sys.exit(f"Chưa có tài liệu trong {raw}")
    n = 0
    with open(cfg["chunks_path"], "w", encoding="utf-8") as f:
        for p in files:
            for j, c in enumerate(chunk_pages(read_any(p), cfg["chunk"]["size_words"], cfg["chunk"]["overlap_words"])):
                rec = {"id": f"{p.stem}#{j}", "doc": p.name, "page": c["page"], "text": c["text"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"{len(files)} tài liệu -> {n} chunk -> {cfg['chunks_path']}")

if __name__ == "__main__":
    main(*sys.argv[1:])
