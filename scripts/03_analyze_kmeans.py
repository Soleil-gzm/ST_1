#!/usr/bin/env python3
"""
分析 K-means 聚类结果，生成：
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

# ========== 配置 ==========
INPUT_TIMESTAMP = None          # 原始文本目录（不含后缀）
OUTPUT_TIMESTAMP = None         # 聚类结果目录（如 xxx_text2vec_kmeans），None则自动构造
# ========================

def get_latest_input_timestamp(output_dir="outputs"):
    """获取最新的原始数据目录（排除包含 '_text2vec_kmeans' 或 '_cluster' 的目录）"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    # 排除包含 '_text2vec_kmeans' 或 '_cluster' 的目录
    timestamps = [d.name for d in output_path.iterdir() 
                  if d.is_dir() and '_text2vec_kmeans' not in d.name and '_cluster' not in d.name]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def generate_cluster_view(output_dir, groups, representative):
    """生成簇视图 HTML"""
    total_clusters = len(groups) - (1 if -1 in groups else 0)
    total_samples = sum(len(v) for v in groups.values())
    noise_samples = len(groups.get(-1, []))

    html_path = output_dir / "cluster_view.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>聚类结果查看器</title>
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
        .noise {{ background-color: #ffebee; }}
        .noise .cluster-header {{ background-color: #ffcdd2; }}
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
    <h1>聚类结果查看器</h1>
    <div class="stats">
        <p>总簇数（不含噪声）: {total_clusters} | 总样本数: {total_samples} | 噪声样本数: {noise_samples}</p>
    </div>
""")
        sorted_clusters = sorted([(cid, intents) for cid, intents in groups.items() if cid != -1],
                                 key=lambda x: len(x[1]), reverse=True)
        if -1 in groups:
            sorted_clusters.append((-1, groups[-1]))
        for cid, intents in sorted_clusters:
            is_noise = (cid == -1)
            cluster_class = "noise" if is_noise else "cluster"
            display_name = "噪声 (Noise)" if is_noise else f"簇 {cid}"
            size = len(intents)
            if not is_noise and cid in representative:
                rep_text = representative[cid]
                if len(rep_text) > 80:
                    rep_text = rep_text[:77] + "..."
                display_name += f' <span style="font-size:0.9em; font-weight:normal;">[代表: "{rep_text}"]</span>'
            f.write(f"""
    <div class="{cluster_class}">
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

def main(input_ts, output_ts):
    input_dir = Path("outputs") / input_ts
    output_dir = Path("outputs") / output_ts

    texts_path = input_dir / "cleaned_intents.txt"
    labels_path = output_dir / "cluster_labels.npy"
    # 优先使用 PCA 降维后的向量（如果存在），否则使用原始向量
    emb_path = output_dir / "embeddings_pca.npy"
    if not emb_path.exists():
        emb_path = output_dir / "embeddings.npy"
        print("未找到 embeddings_pca.npy，使用原始向量进行可视化（可能较慢）")

    if not texts_path.exists():
        print(f"错误：{texts_path} 不存在")
        return
    if not labels_path.exists():
        print(f"错误：{labels_path} 不存在，请先运行聚类脚本")
        return

    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    labels = np.load(labels_path)
    embeddings = np.load(emb_path)

    # 1. 保存带标签的 CSV（如果未生成）
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

    # 3. 准备用于分组的字典（用于生成簇视图）
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
    plt.title("Intent Clustering with K-means (UMAP projection)")
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
                         title=f'Intent Clustering (K-means, {len(unique_labels)} clusters)',
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

if __name__ == "__main__":
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_input_timestamp()
        if input_ts is None:
            print("错误：未找到输入时间戳目录")
            sys.exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP

    if OUTPUT_TIMESTAMP is None:
        # 根据输入时间戳自动构造输出目录名
        output_ts = input_ts + "_text2vec_kmeans"
        output_dir_check = Path("outputs") / output_ts
        if not output_dir_check.exists():
            print(f"错误：输出目录 {output_dir_check} 不存在，请先运行聚类脚本")
            sys.exit(1)
        print(f"自动构造输出目录: {output_ts}")
    else:
        output_ts = OUTPUT_TIMESTAMP

    main(input_ts, output_ts)