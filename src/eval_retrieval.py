"""Đánh giá truy xuất (RQ1). Chunk đúng = chunk chứa nguyên văn đoạn bằng chứng (evidence) trong questions.csv.
Cột: id, question, answer, evidence, answerable (1/0). Câu answerable=0 bị bỏ qua ở bước này."""
import json, sys, re, unicodedata
from pathlib import Path
import pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import Retriever, load_chunks

def squash(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"\s+", " ", s).strip()

def evaluate(cfg, methods=("bm25", "vector", "hybrid"), ks=(1, 3, 5)):
    chunks = load_chunks(cfg["chunks_path"])
    R = Retriever(chunks, cfg)
    qs = pd.read_csv(cfg["questions_path"])
    qs = qs[qs["answerable"] == 1]
    kmax = max(ks); out = {}
    for m in methods:
        hits = {k: 0 for k in ks}; rr = 0.0; per_q = []
        for _, row in qs.iterrows():
            ev = squash(row["evidence"])
            ranked = R.search(m, row["question"], kmax)
            pos = next((i for i, (c, _) in enumerate(ranked, 1) if ev in squash(c["text"])), None)
            for k in ks:
                hits[k] += int(pos is not None and pos <= k)
            rr += 1.0 / pos if pos else 0.0
            per_q.append({"id": row["id"], "rank_of_evidence": pos})
        n = len(qs)
        out[m] = {**{f"Recall@{k}": hits[k] / n for k in ks}, "MRR": rr / n, "n": n, "per_question": per_q}
        print(m, {k: round(v, 3) for k, v in out[m].items() if k != "per_question"})
    return out

if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml", encoding="utf-8"))
    methods = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("bm25", "vector", "hybrid")
    res = evaluate(cfg, methods)
    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    json.dump(res, open(Path(cfg["results_dir"]) / "retrieval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
