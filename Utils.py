import os
import sys
import csv

def load_user_coordinates(filename):
    """
    加载用户坐标文件，返回 [(lat, lon), ...] 列表
    """
    coords = []
    if not os.path.exists(filename):
        print(f"Error: 找不到用户文件 {filename}")
        sys.exit(1)
        
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader) # 跳过表头 (name, latitude, longitude)
        except StopIteration:
            pass
            
        for row in reader:
            if len(row) < 3: continue
            try:
                lat = float(row[1])
                lon = float(row[2])
                coords.append((lat, lon))
            except ValueError:
                continue
    
    if not coords:
        print("Error: 用户坐标文件是空的或格式错误！")
        sys.exit(1)
        
    print(f"成功加载 {len(coords)} 个固定用户坐标。")
    return coords

def get_trace_start_time(filename):
    """
    辅助函数：读取 Trace 文件的第一行时间戳，作为仿真的开始时间
    """
    if not os.path.exists(filename):
        print(f"\nError: 找不到 Trace 文件: {filename}")
        sys.exit(1)
        
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)

            try:
                first_row = next(reader) # 读取第一行
            except StopIteration:
                print(f"\n文件 {filename} 是空的！")
                sys.exit(1)

            try:
                start_time = float(first_row[1]) # 第二列为时间戳
                return start_time
            except (IndexError, ValueError):
                print(f"\n文件 {filename} 格式不对！")
                sys.exit(1)

    except Exception as e:
        print(f"\n读取文件时发生未知异常: {e}")
        sys.exit(1)
