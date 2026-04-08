#!/usr/bin/env python3
"""
按 cluster_id 分组意图，生成一个便于观察的 HTML 文件，并显示每个簇的代表性意图（从 cluster_summary.csv 读取）
输入：outputs/{timestamp}_cluster/clustered_intents.csv
      outputs/{timestamp}_cluster/cluster_summary.csv
输出：outputs/{timestamp}_cluster/cluster_view.html
"""

import csv
from pathlib import Path
from collections import defaultdict
import sys

# ========== 配置 ==========
# 指定时间戳（原始数据的时间戳，例如 "20260407_174104"），脚本会自动加上 "_cluster" 后缀
TARGET_TIMESTAMP = None          # None=自动使用最新（排除 _cluster 后缀）
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

def main(timestamp):
    # 输出目录 = 原始时间戳 + "_cluster"
    output_dir = Path("outputs") / (timestamp + "_cluster")
    csv_path = output_dir / "clustered_intents.csv"
    summary_path = output_dir / "cluster_summary.csv"

    if not csv_path.exists():
        print(f"错误：{csv_path} 不存在，请先运行 04_analyze.py")
        return
    if not summary_path.exists():
        print(f"警告：{summary_path} 不存在，将不显示代表性意图")

    # 读取簇代表性文本
    representative = {}
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row['cluster_id'])
                rep_text = row['representative_text']
                representative[cid] = rep_text

    # 读取 CSV，按 cluster_id 分组
    groups = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = int(row['cluster_id'])
            intent = row['intent']
            groups[cluster_id].append(intent)

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

        # 按簇大小降序排列（噪声除外）
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
                display_name += f' <span style="font-size:0.9em; font-weight:normal;">[ "{rep_text}"]</span>'
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

    print(f"已生成 HTML 查看器: {html_path}")

if __name__ == "__main__":
    if TARGET_TIMESTAMP is None:
        timestamp = get_latest_input_timestamp()
        if timestamp is None:
            print("错误：未找到任何时间戳目录（不含 _cluster 后缀），请先运行 04_analyze.py")
            sys.exit(1)
        print(f"自动使用最新时间戳: {timestamp}")
    else:
        timestamp = TARGET_TIMESTAMP
        print(f"使用指定时间戳: {timestamp}")

    main(timestamp)