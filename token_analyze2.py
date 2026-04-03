import os
import json
from multiprocessing import Pool
from tqdm import tqdm

files = [
    "mix_general_llama3_tokenized_v5.1_10b/train.jsonl"
]

NUM_WORKERS = 128  # 指定进程数

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

def process_chunk(args):
    """
    工作进程函数：处理文件的一个物理字节块
    """
    file_path, start_byte, end_byte = args
    tokens = 0
    
    # 注意：以二进制模式('rb')打开，保证 seek() 字节定位的绝对精确
    with open(file_path, 'rb') as f:
        if start_byte > 0:
            f.seek(start_byte)
            # 如果不是从文件头开始，必然会落在一行文字的中间。
            # 直接废弃当前的不完整行，跳到下一个换行符开始读取，
            # 因为前一个进程的 end_byte 会负责读完这一行。
            f.readline()
            
        while True:
            pos = f.tell()
            if pos >= end_byte:
                break  # 已经超出了分配给该进程的字节块范围，停止读取
            
            line = f.readline()
            if not line:
                break  # 文件结尾
                
            try:
                # 将 bytes 解码为字符串并解析 JSON
                data = json.loads(line.decode('utf-8'))
                tokens += len(data.get("input_ids", []))
            except Exception:
                continue
                
    return tokens

def count_tokens_in_files(file_list, num_workers=128):
    grand_total = 0
    results = {}
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件未找到，跳过: {file_path}")
            continue
            
        print(f"\n🚀 开始多进程统计: {file_path}")
        file_size = os.path.getsize(file_path)
        
        if file_size == 0:
            results[file_path] = 0
            continue
            
        # 根据文件总字节数，均匀切割给 128 个进程
        chunk_size = file_size // num_workers
        args_list = []
        for i in range(num_workers):
            start = i * chunk_size
            # 最后一个进程负责包揽到文件末尾
            end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
            args_list.append((file_path, start, end))
            
        file_tokens = 0
        
        # 启动进程池
        with Pool(processes=num_workers) as pool:
            # imap_unordered 可以让我们在进程完成时实时更新 tqdm 进度条
            for count in tqdm(pool.imap_unordered(process_chunk, args_list), 
                              total=num_workers, 
                              desc=f"处理进度 (共 {num_workers} 个区块)"):
                file_tokens += count
                
        results[file_path] = file_tokens
        grand_total += file_tokens
        print(f"✅ {file_path} 统计完成: {format_tokens(file_tokens)} ({file_tokens:,} tokens)")
        
    print("\n" + "="*40)
    print("📊 统计汇总：")
    for path, count in results.items():
        print(f" - {path.split('/')[-1]}: {format_tokens(count)}")
    
    print("-" * 40)
    print(f"🎯 总计 Token 量: {format_tokens(grand_total)} ({grand_total:,} tokens)")
    print("="*40)

if __name__ == "__main__":
    count_tokens_in_files(files, num_workers=NUM_WORKERS)