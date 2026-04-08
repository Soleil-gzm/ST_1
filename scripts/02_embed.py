#!/usr/bin/env python3
"""
使用 Sentence-Transformers 生成向量
输入：outputs/{timestamp}/cleaned_intents.txt
输出：outputs/{timestamp}/embeddings.npy
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# ========== 配置 ==========
# 设置为 None 表示自动使用最新的时间戳目录
# 如果你想固定使用某个时间戳，直接写字符串，例如 "20260407_161619"
TARGET_TIMESTAMP = None
MODEL_NAME = "./models/all-MiniLM-L6-v2"    # 改为本地解压后的模型目录
USE_TFIDF_FALLBACK = True
# ========================

def get_latest_timestamp(output_dir="outputs"):
    """获取 outputs 目录下最新的时间戳子目录"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir()]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main(timestamp):
    input_dir = Path("outputs") / timestamp
    txt_path = input_dir / "cleaned_intents.txt"
    if not txt_path.exists():
        print(f"错误：{txt_path} 不存在，请先运行清洗脚本")
        return False

    output_path = input_dir / "embeddings.npy"

    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"加载 {len(texts)} 条意图")

    # model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model_name = MODEL_NAME
    print(f"加载模型 {model_name} ...")
    model = SentenceTransformer(model_name)

    batch_size = 512
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    np.save(output_path, embeddings)
    print(f"向量已保存，形状 {embeddings.shape}，路径 {output_path}")
    return True

if __name__ == "__main__":
    if TARGET_TIMESTAMP is None:
        timestamp = get_latest_timestamp()
        if timestamp is None:
            print("错误：未找到任何时间戳目录，请先运行清洗脚本")
            exit(1)
        print(f"自动使用最新时间戳: {timestamp}")
    else:
        timestamp = TARGET_TIMESTAMP
        print(f"使用固定时间戳: {timestamp}")
    main(timestamp)