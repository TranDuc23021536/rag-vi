# rag-vi — Thí nghiệm RAG cho tài liệu tiếng Việt

So sánh ba cách truy xuất (vector, BM25, hybrid-RRF) trong hệ thống hỏi đáp tài liệu tiếng Việt.

## Cài đặt
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Quy trình
1. Bỏ tài liệu (PDF/TXT/MD) vào `data/raw/`.
2. `python src/ingest.py` → tạo `data/chunks.jsonl`. **Mở file này ra đọc vài dòng** để kiểm tra mất dấu, lẫn đầu/chân trang.
3. Soạn `data/questions.csv` theo `data/questions_template.csv`:
   - `evidence` = đoạn **nguyên văn** trong tài liệu chứa đáp án, ngắn (8–30 từ) để ít bị cắt giữa hai chunk.
   - `answerable=0` cho câu hỏi tài liệu không trả lời được (dùng đo hành vi từ chối ở bước sinh).
4. `python src/eval_retrieval.py` → `results/retrieval.json` (Recall@1/3/5, MRR cho bm25, vector, hybrid).
   Chạy riêng một phương pháp: `python src/eval_retrieval.py config.yaml bm25`.
5. Đổi `chunk.size_words` (100/150/250) rồi chạy lại để so sánh cỡ chunk. Chạy lại `ingest.py` mỗi lần đổi.

## Ghi chú thiết kế
- Mọi tham số nằm trong `config.yaml` để tái lập thí nghiệm. Cố định `generation.model` cho mọi lần chạy.
- Embedding được cache vào `data/emb_cache/`, đổi chunk hay mô hình thì tự tính lại.
- Chunk chia theo số từ có overlap; bằng chứng nằm vắt ngang hai chunk sẽ bị tính là trượt. Hãy chọn evidence ngắn và ghi nhận hạn chế này trong báo cáo.
- `src/generate.py` chưa có script chạy toàn bộ: sẽ bổ sung ở bước đánh giá câu trả lời (RQ2, RQ3).
