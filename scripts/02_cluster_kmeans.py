#!/usr/bin/env python3
"""
使用 text2vec-base-chinese 模型生成向量，并进行 PCA 降维 + K-means 聚类。
输入：outputs/{input_timestamp}/cleaned_intents.txt
输出：outputs/{output_timestamp}/embeddings.npy (原始768维)
      outputs/{output_timestamp}/embeddings_pca.npy (降维后)
      outputs/{output_timestamp}/cluster_labels.npy
      outputs/{output_timestamp}/clustered_intents.csv
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# ========== 配置 ==========
INPUT_TIMESTAMP = None          # None=自动使用最新（不含 _cluster 后缀）
OUTPUT_SUFFIX = "_text2vec_kmeans"   # 输出目录后缀
N_CLUSTERS = 3000               # K-means 簇数量
MODEL_NAME = "./models/text2vec-base-chinese"   # 本地模型路径（必须已下载）
PCA_COMPONENTS = 100            # PCA 降维后的维度（建议 50~200）
# ========================

def get_latest_input_timestamp(output_dir="outputs"):
    """获取最新的原始数据目录（不含 _cluster 后缀）"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir() and '_cluster' not in d.name]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main(input_ts, output_ts, n_clusters):
    input_dir = Path("outputs") / input_ts
    txt_path = input_dir / "cleaned_intents.txt"
    if not txt_path.exists():
        print(f"错误：{txt_path} 不存在，请先运行清洗脚本")
        return

    # 读取文本
    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"加载 {len(texts)} 条意图")

    # 加载模型并生成向量
    print("加载 text2vec-base-chinese 模型...")
    model = SentenceTransformer(MODEL_NAME)
    print("生成文本向量...")
    # 分批编码，避免内存溢出（batch_size 可根据显存/内存调整）
    batch_size = 256
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"原始向量形状: {embeddings.shape}")

    # PCA 降维
    print(f"使用 PCA 降维到 {PCA_COMPONENTS} 维...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"降维后向量形状: {embeddings_pca.shape}")

    # K-means 聚类（使用降维后的向量，速度更快）
    print(f"执行 K-means 聚类，簇数 = {n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
    labels = kmeans.fit_predict(embeddings_pca)
    print("聚类完成")

    # 创建输出目录
    output_dir = Path("outputs") / output_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始向量、降维后向量和标签
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "embeddings_pca.npy", embeddings_pca)
    np.save(output_dir / "cluster_labels.npy", labels)
    print(f"向量和标签已保存到 {output_dir}")

    # 保存带标签的 CSV
    df = pd.DataFrame({"intent": texts, "cluster_id": labels})
    csv_path = output_dir / "clustered_intents.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {csv_path}")

    # 简单统计
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== 聚类统计 ===")
    print(f"总样本数: {len(texts)}")
    print(f"实际簇数量: {len(unique)}")
    print(f"最大簇大小: {counts.max()}")
    print(f"最小簇大小: {counts.min()}")
    print(f"平均簇大小: {counts.mean():.1f}")

if __name__ == "__main__":
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_input_timestamp()
        if input_ts is None:
            print("错误：未找到任何输入时间戳目录，请先运行清洗脚本")
            sys.exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP
        print(f"使用指定输入时间戳: {input_ts}")

    output_ts = input_ts + OUTPUT_SUFFIX
    print(f"输出目录: {output_ts}")

    main(input_ts, output_ts, N_CLUSTERS)