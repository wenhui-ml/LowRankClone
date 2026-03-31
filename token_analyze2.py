import json
from tqdm import tqdm

files = [
    "mix_general_llama3_tokenized_v5.1/fineweb-edu-dedup.jsonl",
    "mix_general_llama3_tokenized_v5.1/hermes.jsonl"
]

def format_tokens(count):
    """支持 P/T/B/M/K 的表示法"""
    if count >= 1_000_000_000_000_000:
        return f"{count / 1_000_000_000_000_000:.3f} P"
    elif count >= 1_000_000_000_000:
        return f"{count / 1_000_000_000_000:.3f} T"
    elif count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.3f} B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.3f} M"
    elif count >= 1_000:
        return f"{count / 1_000:.3f} K"
    return f"{count} "

def count_tokens_in_files(file_list):
    grand_total = 0
    results = {}
    
    for file_path in file_list:
        file_tokens = 0
        print(f"\n开始统计: {file_path}")
        
        # 使用 open 对大文件更友好，且逐行处理不会爆内存
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"处理中"):
                try:
                    data = json.loads(line)
                    # 直接获取预处理好的 input_ids 长度
                    file_tokens += len(data["input_ids"])
                except (json.JSONDecodeError, KeyError):
                    continue
        
        results[file_path] = file_tokens
        grand_total += file_tokens
        print(f"{file_path} 统计完成: {format_tokens(file_tokens)} ({file_tokens:,} tokens)")
        
    print("\n" + "="*40)
    print("统计汇总：")
    for path, count in results.items():
        print(f" - {path.split('/')[-1]}: {format_tokens(count)}")
    
    print("-" * 40)
    print(f"总计 Token 量: {format_tokens(grand_total)} ({grand_total:,} tokens)")
    print("="*40)

if __name__ == "__main__":
    count_tokens_in_files(files)