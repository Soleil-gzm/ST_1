#!/usr/bin/env python3
"""
按 cluster_id 分组意图，生成一个便于观察的 HTML 文件
输入：outputs/{input_timestamp}/clustered_intents.csv
输出：outputs/{output_timestamp}/cluster_view.html
"""

import csv
from pathlib import Path
from collections import defaultdict
import sys

# ========== 配置 ==========
INPUT_TIMESTAMP = None          # None=自动使用最新（排除 _cluster 后缀）
OUTPUT_TIMESTAMP = None         # None=自动生成（输入时间戳 + "_cluster"）
# ========================

def get_latest_input_timestamp(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir() and '_cluster' not in d.name]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main(input_ts, output_ts):
    input_dir = Path("outputs") / input_ts
    output_dir = Path("outputs") / output_ts
    csv_path = input_dir / "clustered_intents.csv"
    if not csv_path.exists():
        print(f"错误：{csv_path} 不存在，请先运行 04_analyze.py")
        return

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
        # 使用 f-string 并双写所有 CSS 中的花括号
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
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_input_timestamp()
        if input_ts is None:
            print("错误：未找到任何输入时间戳目录（不含 _cluster 后缀），请先运行 04_analyze.py")
            sys.exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP
        print(f"使用指定输入时间戳: {input_ts}")

    if OUTPUT_TIMESTAMP is None:
        output_ts = input_ts + "_cluster"
        print(f"自动生成输出时间戳: {output_ts}")
    else:
        output_ts = OUTPUT_TIMESTAMP
        print(f"使用指定输出时间戳: {output_ts}")

    main(input_ts, output_ts)