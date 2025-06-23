#!/usr/bin/env python3
"""
日志查看工具
用于快速查看和分析运行日志
"""
import sys
from pathlib import Path
from datetime import datetime
import re

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def list_log_files():
    """列出所有日志文件"""
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        print("❌ logs文件夹不存在")
        return []
    
    log_files = sorted(logs_dir.glob("integrated_scheduler_demo_*.log"), reverse=True)
    return log_files

def parse_log_file(log_file):
    """解析日志文件提取关键信息"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键信息
    info = {
        'file': log_file.name,
        'size': log_file.stat().st_size,
        'start_time': None,
        'end_time': None,
        'errors': [],
        'warnings': [],
        'reallocations': 0,
        'scheduler_updates': 0
    }
    
    # 查找开始时间
    start_match = re.search(r'⏰ 启动时间: ([\d-]+ [\d:]+)', content)
    if start_match:
        info['start_time'] = start_match.group(1)
    
    # 查找结束时间
    end_match = re.search(r'⏰ 结束时间: ([\d-]+ [\d:]+)', content)
    if end_match:
        info['end_time'] = end_match.group(1)
    
    # 统计错误
    info['errors'] = re.findall(r'❌ (.+)', content)
    info['error_count'] = len(info['errors'])
    
    # 统计动态重调度
    info['reallocations'] = len(re.findall(r'触发动态重调度', content))
    
    # 统计调度更新
    info['scheduler_updates'] = len(re.findall(r'定期调度更新', content))
    
    return info

def display_log_summary(log_files):
    """显示日志摘要"""
    print("📊 日志文件摘要")
    print("=" * 80)
    
    for i, log_file in enumerate(log_files[:10]):  # 只显示最近10个
        info = parse_log_file(log_file)
        
        print(f"\n{i+1}. {info['file']}")
        print(f"   大小: {info['size'] / 1024:.1f} KB")
        print(f"   开始: {info['start_time'] or '未知'}")
        print(f"   结束: {info['end_time'] or '运行中'}")
        print(f"   统计: 错误={info['error_count']}, 重调度={info['reallocations']}, "
              f"更新={info['scheduler_updates']}")
        
        if info['errors']:
            print(f"   最后错误: {info['errors'][-1][:60]}...")

def view_log_detail(log_file):
    """查看日志详情"""
    print(f"\n📄 查看日志: {log_file.name}")
    print("=" * 80)
    
    # 选择查看选项
    print("\n选择查看选项:")
    print("1. 查看全部内容")
    print("2. 只看错误信息")
    print("3. 只看调度信息")
    print("4. 查看最后100行")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if choice == '1':
        # 显示全部（分页）
        for i, line in enumerate(lines):
            print(line.rstrip())
            if (i + 1) % 40 == 0:
                input("\n--- 按回车继续 ---")
    
    elif choice == '2':
        # 只看错误
        print("\n🔴 错误信息:")
        error_lines = [line for line in lines if '❌' in line or 'ERROR' in line or 'error' in line.lower()]
        for line in error_lines:
            print(line.rstrip())
    
    elif choice == '3':
        # 只看调度信息
        print("\n📋 调度信息:")
        schedule_lines = [line for line in lines if any(kw in line for kw in 
                         ['调度', 'Layer', '🔄', '📊', 'schedule', 'allocation'])]
        for line in schedule_lines:
            print(line.rstrip())
    
    elif choice == '4':
        # 最后100行
        print("\n📜 最后100行:")
        for line in lines[-100:]:
            print(line.rstrip())

def main():
    """主函数"""
    print("🔍 集成调度器日志查看器")
    print("=" * 60)
    
    # 列出日志文件
    log_files = list_log_files()
    
    if not log_files:
        print("没有找到日志文件")
        return
    
    # 显示摘要
    display_log_summary(log_files)
    
    # 选择详细查看
    print(f"\n找到 {len(log_files)} 个日志文件")
    choice = input("\n输入编号查看详情 (1-10) 或按回车退出: ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(log_files):
            view_log_detail(log_files[idx])
        else:
            print("❌ 无效的选择")

if __name__ == "__main__":
    main()