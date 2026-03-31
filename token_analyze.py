import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 模型路径 (确保该文件夹下有 tokenizer.json 等文件)
model_path = "models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

files = [
    "mix_general_llama3_tokenized_v5.1/fineweb-edu-dedup.jsonl",
    "mix_general_llama3_tokenized_v5.1/hermes.jsonl"
]

def count_tokens_in_file(file_path):
    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {file_path.split('/')[-1]}"):
            try:
                data = json.loads(line)
                # 假设每行有一个 'text' 字段包含文本
                # 如果数据格式是已经 tokenized 好的列表，请按需修改代码
                text = data.get("text", "")
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
            except Exception as e:
                continue
    return total_tokens

def format_tokens(count):
    """支持 P/T/B/M/K 的表示法"""
    if count >= 1_000_000_000_000_000:
        return f"{count / 1_000_000_000_000_000:.2f} P"
    elif count >= 1_000_000_000_000:
        return f"{count / 1_000_000_000_000:.2f} T"
    elif count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f} B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f} K"
    else:
        return f"{count} "

# 执行统计
grand_total = 0
for file in files:
    count = count_tokens_in_file(file)
    print(f"{file}: {format_tokens(count)} tokens")
    grand_total += count

print("-" * 30)
print(f"Total Tokens: {format_tokens(grand_total)}")