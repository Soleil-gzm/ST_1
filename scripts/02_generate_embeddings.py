#!/usr/bin/env python3
"""
生成文本向量并进行 PCA 降维（只运行一次）
输入：outputs/{input_timestamp}/cleaned_intents.txt
输出：outputs/{output_timestamp}_base_embeddings/embeddings.npy
      outputs/{output_timestamp}_base_embeddings/embeddings_pca.npy
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# ========== 配置 ==========
INPUT_TIMESTAMP = None          # None=自动使用最新（不含任何后缀）
OUTPUT_SUFFIX = "_base_embeddings"   # 输出目录后缀
MODEL_NAME = "./models/text2vec-base-chinese"
PCA_COMPONENTS = 100
# ========================

def get_latest_input_timestamp(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    # 排除包含 '_base_embeddings' 或 '_text2vec_' 或 '_cluster' 的目录
    timestamps = [d.name for d in output_path.iterdir()
                  if d.is_dir() and '_base_embeddings' not in d.name 
                  and '_text2vec_' not in d.name and '_cluster' not in d.name]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main():
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_input_timestamp()
        if input_ts is None:
            print("错误：未找到任何输入时间戳目录，请先运行清洗脚本")
            sys.exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP
        print(f"使用指定输入时间戳: {input_ts}")

    input_dir = Path("outputs") / input_ts
    txt_path = input_dir / "cleaned_intents.txt"
    if not txt_path.exists():
        print(f"错误：{txt_path} 不存在")
        return

    # 读取文本
    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"加载 {len(texts)} 条意图")

    # 加载模型并生成向量
    print("加载 text2vec-base-chinese 模型...")
    model = SentenceTransformer(MODEL_NAME)
    print("生成文本向量...")
    batch_size = 256
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"原始向量形状: {embeddings.shape}")

    # PCA 降维
    print(f"使用 PCA 降维到 {PCA_COMPONENTS} 维...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"降维后向量形状: {embeddings_pca.shape}")

    # 创建输出目录
    output_ts = input_ts + OUTPUT_SUFFIX
    output_dir = Path("outputs") / output_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存向量
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "embeddings_pca.npy", embeddings_pca)
    print(f"向量已保存到 {output_dir}")

if __name__ == "__main__":
    main()