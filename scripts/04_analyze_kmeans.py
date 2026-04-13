#!/usr/bin/env python3
"""
分析 K-means 聚类结果，生成簇汇总、可视化和簇视图
自动查找最新的 _kmeans_K* 目录
- 簇汇总 CSV
- 静态 PNG 散点图
- 交互式 HTML 散点图
- 簇视图 HTML（可折叠，显示每个簇的所有意图）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import sys
from collections import defaultdict
import json
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ========== 配置 ==========
# 可手动指定聚类结果目录，None 表示自动查找
# CLUSTER_DIR = None
#指定目录  CLUSTER_DIR = "outputs/20260413_105308_base_embeddings_kmeans_K1000"
CLUSTER_DIR = '/home/GUO_Zimeng/coding/Sentence_Transformers/ST_1/outputs/20260413_105308_base_embeddings_kmeans_K750'
# ========================

def get_latest_kmeans_dir(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    dirs = [d for d in output_path.iterdir() if d.is_dir() and '_kmeans_K' in d.name]
    if not dirs:
        return None
    # 按修改时间降序排序（最新修改的在前）
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]

def generate_cluster_view(output_dir, groups, representative):
    """生成可折叠的簇视图 HTML"""
    total_clusters = len(groups) - (1 if -1 in groups else 0)
    total_samples = sum(len(v) for v in groups.values())
    noise_samples = len(groups.get(-1, []))

    html_path = output_dir / "cluster_view.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>聚类结果查看器 (K-means)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .cluster {{
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .cluster-header {{
            background-color: #f0f0f0;
            padding: 10px;
            cursor: pointer;
            font-weight: bold;
            user-select: none;
        }}
        .cluster-header:hover {{ background-color: #e0e0e0; }}
        .cluster-content {{
            display: none;
            padding: 10px;
            background-color: #fff;
            border-top: 1px solid #ddd;
            max-height: 400px;
            overflow-y: auto;
        }}
        .cluster-content ul {{ margin: 0; padding-left: 20px; }}
        .cluster-content li {{ margin: 5px 0; }}
        .stats {{ margin-bottom: 20px; }}
    </style>
    <script>
        function toggleCluster(id) {{
            var content = document.getElementById('content-' + id);
            if (content.style.display === 'none' || content.style.display === '') {{
                content.style.display = 'block';
            }} else {{
                content.style.display = 'none';
            }}
        }}
    </script>
</head>
<body>
    <h1>聚类结果查看器 (K-means)</h1>
    <div class="stats">
        <p>总簇数: {total_clusters} | 总样本数: {total_samples} | 噪声样本数: {noise_samples}</p>
    </div>
""")
        # 按簇大小降序排列
        sorted_clusters = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        for cid, intents in sorted_clusters:
            size = len(intents)
            display_name = f"簇 {cid}"
            if cid in representative:
                rep_text = representative[cid]
                if len(rep_text) > 80:
                    rep_text = rep_text[:77] + "..."
                display_name += f' <span style="font-size:0.9em; font-weight:normal;">[代表: "{rep_text}"]</span>'
            f.write(f"""
    <div class="cluster">
        <div class="cluster-header" onclick="toggleCluster({cid})">
            {display_name} (共 {size} 条意图)
        </div>
        <div id="content-{cid}" class="cluster-content">
            <ul>
""")
            for intent in intents:
                intent_escaped = intent.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                f.write(f"                <li>{intent_escaped}</li>\n")
            f.write("""            </ul>
        </div>
    </div>
""")
        f.write("""
</body>
</html>""")
    print(f"已生成簇视图 HTML: {html_path}")

def main():
    if CLUSTER_DIR is None:
        output_dir = get_latest_kmeans_dir()
        if output_dir is None:
            print("错误：未找到任何 _kmeans_K* 目录，请先运行聚类脚本")
            sys.exit(1)
        print(f"自动使用聚类结果目录: {output_dir}")
    else:
        output_dir = Path(CLUSTER_DIR)
        if not output_dir.exists():
            print(f"错误：目录 {output_dir} 不存在")
            sys.exit(1)

    # 从目录名提取基础信息
    # 目录名格式：{timestamp}_base_embeddings_kmeans_K{cluster_num}
    parts = output_dir.name.split("_kmeans_K")
    if len(parts) != 2:
        print(f"错误：无法解析目录名 {output_dir.name}")
        return
    base_name = parts[0]   # 例如 "20260409_131912_base_embeddings"
    cluster_num = parts[1]  # 例如 "1000"
    # 原始时间戳 = base_name 去掉 "_base_embeddings"
    raw_timestamp = base_name.replace("_base_embeddings", "")
    input_dir = output_dir.parent / raw_timestamp
    base_emb_dir = output_dir.parent / base_name

    texts_path = input_dir / "cleaned_intents.txt"
    labels_path = output_dir / "cluster_labels.npy"
    # 优先使用 PCA 降维后的向量（更小，可视化更快）
    emb_path = base_emb_dir / "embeddings_pca.npy"
    if not emb_path.exists():
        emb_path = base_emb_dir / "embeddings.npy"
        print("未找到 embeddings_pca.npy，使用原始向量（可能较慢）")

    if not texts_path.exists():
        print(f"错误：{texts_path} 不存在")
        return
    if not labels_path.exists():
        print(f"错误：{labels_path} 不存在，请先运行聚类脚本")
        return

    # 读取数据
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    labels = np.load(labels_path)
    embeddings = np.load(emb_path)

    # 1. 确保 clustered_intents.csv 存在（如果不存在则生成）
    csv_path = output_dir / "clustered_intents.csv"
    if not csv_path.exists():
        df = pd.DataFrame({"intent": texts, "cluster_id": labels})
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存 {csv_path}")

    # 2. 生成簇汇总
    cluster_summary = []
    unique_clusters = np.unique(labels)
    for cid in unique_clusters:
        indices = np.where(labels == cid)[0]
        size = len(indices)
        cluster_center = np.mean(embeddings[indices], axis=0)
        dists = np.linalg.norm(embeddings[indices] - cluster_center, axis=1)
        best_idx = indices[np.argmin(dists)]
        representative = texts[best_idx]
        cluster_summary.append({
            "cluster_id": int(cid),
            "size": size,
            "representative_text": representative
        })
    cluster_summary.sort(key=lambda x: x["size"], reverse=True)
    summary_df = pd.DataFrame(cluster_summary)
    summary_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"已保存簇汇总 {summary_path}")

    # 3. 准备分组字典（用于簇视图）
    groups = defaultdict(list)
    for intent, label in zip(texts, labels):
        groups[label].append(intent)
    representative_dict = {row['cluster_id']: row['representative_text'] for _, row in summary_df.iterrows()}
    generate_cluster_view(output_dir, groups, representative_dict)

    # 4. 可视化：UMAP 降维到 2D
    print("生成可视化散点图...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    reduced_2d = reducer_2d.fit_transform(embeddings)

    # 静态 PNG
    plt.figure(figsize=(16, 12))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i % 20)
        plt.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                    c=[color], s=1, alpha=0.6, label=f'Cluster {label}')
    plt.title(f"Intent Clustering with K-means (K={cluster_num}, UMAP projection)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    if len(handles) > 25:
        plt.legend(handles[:25], labels_leg[:25], loc='upper right', fontsize=8)
    else:
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    png_path = output_dir / "cluster_visualization.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"已保存静态图 {png_path}")

    # 交互式 HTML
    try:
        import plotly.express as px
        df_plot = pd.DataFrame({
            'x': reduced_2d[:, 0],
            'y': reduced_2d[:, 1],
            'cluster': labels.astype(str),
            'intent': texts
        })
        fig = px.scatter(df_plot, x='x', y='y', color='cluster',
                         hover_data=['intent'],
                         title=f'Intent Clustering (K-means, K={cluster_num})',
                         labels={'cluster': 'Cluster ID'},
                         color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_traces(marker=dict(size=2, opacity=0.6))
        html_path = output_dir / "cluster_visualization_interactive.html"
        fig.write_html(html_path)
        print(f"已保存交互式 HTML: {html_path}")
    except ImportError:
        print("plotly 未安装，跳过交互式 HTML")

    # 统计信息
    print("\n=== 聚类统计 ===")
    print(f"总样本数: {len(texts)}")
    print(f"簇数量: {len(unique_labels)}")
    print(f"最大簇大小: {summary_df['size'].max()}")
    print(f"最小簇大小: {summary_df['size'].min()}")

    # 计算指标
    '''
    轮廓系数（Silhouette Score）: 范围：[-1, 1]
    解读：
    > 0.5：聚类效果良好，簇内紧致、簇间分离。
    0.25 ~ 0.5：中等，有一定结构但存在重叠。
    < 0.25：效果较差，几乎没有明显的簇结构。
    '''
    sil = silhouette_score(embeddings, labels)

    '''
    Calinski-Harabasz (CH) 指数: 范围：无上界，越高越好。
    解读：CH 指数与簇内离散度成反比、与簇间离散度成正比。
    但对于不同 K 值不能直接比较绝对值，需要看相对变化。
    175 这个数值对于 12 万样本、1000 个簇来说偏低（通常理想情况下可达数千甚至上万），表明簇间差异不大。
    '''
    ch = calinski_harabasz_score(embeddings, labels)

    '''
    Davies-Bouldin (DB) 指数: 
    范围：[0, ∞)，越低越好。0 表示完美分离。
    解读：
    < 1.0：聚类效果优秀。
    1.0 ~ 1.5：可接受。
    > 2.0：簇间重叠严重，分离度差。
    '''
    db = davies_bouldin_score(embeddings, labels)

    metrics = {
        "silhouette_score": float(sil),
        "calinski_harabasz_score": float(ch),
        "davies_bouldin_score": float(db),
        "num_clusters": len(unique_labels),
        "total_samples": len(texts)
    }
    with open(output_dir / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"聚类评估指标已保存到 {output_dir}/cluster_metrics.json")

if __name__ == "__main__":
    main()