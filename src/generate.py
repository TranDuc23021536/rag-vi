"""Sinh câu trả lời từ các đoạn truy xuất. Prompt ép trả lời theo tài liệu, trích nguồn, và từ chối khi thiếu bằng chứng."""
import os

REFUSAL = "Tài liệu không đủ thông tin để trả lời câu hỏi này."

SYSTEM = (
    "Bạn là trợ lý hỏi đáp tài liệu. Chỉ trả lời dựa trên các đoạn tài liệu được cung cấp. "
    "Sau mỗi ý, ghi nguồn dạng [số đoạn]. Nếu các đoạn không chứa đủ thông tin để trả lời, "
    f"hãy trả lời đúng một câu: \"{REFUSAL}\" Không dùng kiến thức bên ngoài."
)

def build_prompt(question, chunks):
    ctx = "\n\n".join(f"[{i}] ({c['doc']}, tr.{c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return f"Các đoạn tài liệu:\n{ctx}\n\nCâu hỏi: {question}\n\nTrả lời:"

def generate(question, chunks, gcfg):
    if not gcfg.get("model"):
        raise ValueError("Chưa điền generation.model trong config.yaml")
    prompt = build_prompt(question, chunks)
    if gcfg["provider"] == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # đọc ANTHROPIC_API_KEY từ biến môi trường
        r = client.messages.create(model=gcfg["model"], max_tokens=gcfg["max_tokens"],
                                   temperature=gcfg["temperature"], system=SYSTEM,
                                   messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import OpenAI
    client = OpenAI(base_url=gcfg["base_url"] or None, api_key=os.environ.get("OPENAI_API_KEY", "none"))
    r = client.chat.completions.create(model=gcfg["model"], temperature=gcfg["temperature"],
                                       max_tokens=gcfg["max_tokens"],
                                       messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": prompt}])
    return r.choices[0].message.content
