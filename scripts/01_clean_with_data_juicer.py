#!/usr/bin/env python3
"""
使用 Data-Juicer 清洗意图数据（基于独立配置文件）
输入：data/user_intents.json
输出：outputs/{timestamp}/cleaned_intents.txt
"""

import json
import subprocess
import os
import sys
from pathlib import Path
import yaml

def main(timestamp=None):
    # 如果没有传入时间戳，自动生成
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建带时间戳的输出目录
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 原始 JSON 文件路径
    input_json = "data/user_intents.json"
    if not os.path.exists(input_json):
        print(f"错误：{input_json} 不存在")
        return None
    
    # 1. 将原始 JSON 转换为 Data-Juicer 所需的 JSONL 格式
    temp_jsonl = output_dir / "temp_intents.jsonl"
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    intents = data['user_intents']
    
    with open(temp_jsonl, 'w', encoding='utf-8') as f:
        for intent in intents:
            line = json.dumps({"text": intent}, ensure_ascii=False)
            f.write(line + '\n')
    print(f"已生成临时输入文件: {temp_jsonl} (共 {len(intents)} 条)")
    
    # 2. 读取用户提供的配置文件模板
    config_template_path = "configs/data_juicer_clear_intents.yaml"
    if not os.path.exists(config_template_path):
        print(f"错误：配置文件 {config_template_path} 不存在")
        return None
    
    with open(config_template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 3. 替换占位符
    output_jsonl = output_dir / "cleaned_intents.jsonl"
    if 'dataset_path' in config:
        config['dataset_path'] = str(temp_jsonl.absolute())
    if 'export_path' in config:
        config['export_path'] = str(output_jsonl.absolute())
    
    # 4. 写入临时配置文件
    temp_config = output_dir / "temp_config.yaml"
    with open(temp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    # 5. 执行 Data-Juicer
    cmd = f"dj-process --config {temp_config}"
    print("运行 Data-Juicer 清洗...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("清洗失败：", result.stderr)
        print(f"临时配置文件保留在: {temp_config}")
        print(f"临时输入文件保留在: {temp_jsonl}")
        return None
    
    # 6. 提取清洗后的意图文本
    if not output_jsonl.exists():
        print(f"错误：清洗后未生成输出文件 {output_jsonl}")
        return None
    
    cleaned_txt = output_dir / "cleaned_intents.txt"
    with open(output_jsonl, 'r', encoding='utf-8') as inf, open(cleaned_txt, 'w', encoding='utf-8') as outf:
        for line in inf:
            obj = json.loads(line)
            text = obj.get('text', '').strip()
            if text:
                outf.write(text + '\n')
    
    # 统计
    with open(cleaned_txt, 'r') as f:
        count = sum(1 for _ in f)
    print(f"清洗完成，原始 {len(intents)} 条，剩余 {count} 条")
    
    # 清理临时文件（可选）
    temp_jsonl.unlink()
    temp_config.unlink()
    
    print(f"清洗结果已保存到 {cleaned_txt}")
    return timestamp

if __name__ == "__main__":
    # 如果直接运行，生成时间戳并执行
    main()