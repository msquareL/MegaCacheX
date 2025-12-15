import os
import sys
import time
import csv 
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Classes import MegaConstellation, User

# 文件路径
TLE_SAT_FILE = "configuration/S_constellation.tle"
TLE_SDC_FILE = "configuration/SDC_constellation.tle"
GS_CSV_FILE  = "configuration/GS_Starlink_Nodes.csv"
USER_NODES_FILE = "configuration/User_Nodes.csv"
TRACE_FILE   = "StarFront_CDN_Trace_Dataset.csv"

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

def process(mc, requests, current_time, stats, user_coord_iterator):
    """
    处理当前时间步的所有请求
    """
    potential_actions = []

    step_latencies = []

    aggregation_map = {} # 聚合字典，用于累加延迟数据

    for req in requests:
        content = req['content']
        # 更新内容热度
        content.update_popularity(1)
        
        lat, lon = next(user_coord_iterator)
        # 创建临时 User 对象
        temp_user = User(f"User_{lat:.2f}_{lon:.2f}", lat, lon)
        
        # 寻找最近接入卫星
        min_dist = float('inf')
        access_sat = None
        
        # 遍历所有卫星寻找最近的
        for sat in mc.satellites.values():
            d = temp_user.get_distance(sat)
            # 1500km 视距限制
            if d < 1500 and d < min_dist:
                min_dist = d
                access_sat = sat
        
        if not access_sat:
            continue

        # 用户接入卫星作为路由起点
        routing_start_node_id = access_sat.id

        # 计算第一跳延迟 (User -> Access Sat)
        first_hop_latency = min_dist / 299792.458 #+ content.size / 12.5

        # 寻找最近的内容副本
        best_path = None
        min_path_latency = float('inf')

        # 候选目标：所有 SDC + 所有缓存了该内容的卫星
        candidates = list(mc.sdcs.keys())
        for sat_id, sat_node in mc.satellites.items():
            if content in sat_node.cached_contents:
                candidates.append(sat_id)

        # 遍历候选者，找最快路径
        for target_id in candidates:
            # 接入卫星本身有缓存
            if target_id == routing_start_node_id:
                temp_path = [routing_start_node_id]
                path_latency = 0.0
            else:
                # 计算从接入卫星到目标的路径
                temp_path = mc.ospc_routing(routing_start_node_id, target_id, content.size)
                
                if temp_path:
                    # 计算路径延迟
                    path_latency = 0.0
                    for i in range(len(temp_path)-1):
                        path_latency += mc.get_link_delay(temp_path[i], temp_path[i+1], content.size)
                else:
                    continue # 不可达
        
            # 计算全路径延迟
            total_latency_check = first_hop_latency + path_latency

            if total_latency_check < min_path_latency:
                min_path_latency = total_latency_check
                best_path = temp_path

        # 缓存决策 (MLC3)
        if best_path:
            # min_path_latency 是包含了 User->Sat->...->SDC 的全链路延迟
            latency_sat = min_path_latency
            
            latency_gs = 0.200

            final_request_latency = 0.0

            if len(best_path) == 1:
                # 接入卫星直接命中
                stats['total_hits'] += 1
                final_request_latency = first_hop_latency
                step_latencies.append(final_request_latency)

                continue 
            else:
                # 没命中接入卫星
                final_request_latency = min_path_latency
                step_latencies.append(final_request_latency)

            # best_candidate_node = None
            # max_score = -1.0

            current_accumulated_latency = first_hop_latency
            
            # 遍历路径上的中间卫星
            for i, node_id in enumerate(best_path[:-1]):
                if node_id in mc.satellites:
                    sat_node = mc.satellites[node_id]

                    if content in sat_node.cached_contents:
                        continue

                    # 动态累加延迟
                    if i > 0:
                        prev_node_id = best_path[i-1]
                        link_delay = mc.get_link_delay(prev_node_id, node_id, content.size)
                        current_accumulated_latency += link_delay

                    # 将延迟累加到 aggregation_map 中
                    key = (node_id, content.id)
                    if key not in aggregation_map:
                        aggregation_map[key] = {
                            'node': sat_node,
                            'content': content,
                            'sum_latency_local': 0.0,
                            'sum_latency_remote': 0.0,
                            'req_count': 0
                        }

                    # 累加操作
                    aggregation_map[key]['sum_latency_local'] += current_accumulated_latency
                    aggregation_map[key]['sum_latency_remote'] += latency_sat
                    aggregation_map[key]['req_count'] += 1

    # 遍历聚合数据，计算分数并生成候选动作
    for key, data in aggregation_map.items():
        node = data['node']
        content = data['content']
        sum_lat_remote = data['sum_latency_remote']
        sum_lat_local = data['sum_latency_local']
        
        score = node.satellite_score(content, sum_lat_remote, sum_lat_local)
        
        if score > 0.01:
            potential_actions.append({
                'score': score,
                'node': node,
                'content': content,
                'user_id': f"Aggregated({data['req_count']})"
            })

    # 按分数排序，优先缓存高分内容
    potential_actions.sort(key=lambda x: x['score'], reverse=True)

    sat_new_usage = {}

    count_cached = 0
    for action in potential_actions:
        node = action['node']
        content = action['content']
        score = action['score']
        user_id = action['user_id']

        # 初始化计数器
        if node.id not in sat_new_usage:
            sat_new_usage[node.id] = 0.0
        
        if content in node.cached_contents:
            continue 
        
        # 检查容量，容量不足则跳过
        if sat_new_usage[node.id] + content.size > node.capacity:
            continue

        if node.cache_content(content):
            count_cached += 1
            sat_new_usage[node.id] += content.size
            print(f"      [Cache] {user_id} -> {node.id} 存入 {content.id} (Score:{score:.2f})")

    return step_latencies

def run_simulation():
    # 初始化环境
    print("正在初始化 MegaCacheX 仿真环境")
    
    mc = MegaConstellation()
    
    # 加载节点文件
    if os.path.exists(TLE_SAT_FILE):
        mc.load_tle_file(TLE_SAT_FILE, "satellite")
        print(f"已加载卫星: {len(mc.satellites)}")
    else:
        print(f"错误: 找不到文件 {TLE_SAT_FILE}")
        return

    if os.path.exists(TLE_SDC_FILE):
        mc.load_tle_file(TLE_SDC_FILE, "sdc")
        print(f"已加载 SDC: {len(mc.sdcs)}")
    else:
        print(f"错误: 找不到文件 {TLE_SDC_FILE}")
        return

    if os.path.exists(GS_CSV_FILE):
        mc.load_gs_csv(GS_CSV_FILE)
        print(f"已加载地面站: {len(mc.ground_stations)}")
    else:
        print(f"错误: 找不到文件 {GS_CSV_FILE}")
        return
    
    user_coords_list = load_user_coordinates(USER_NODES_FILE)
    user_coord_iter = itertools.cycle(user_coords_list)

    # 获取 CSV 文件中的开始时间
    start_timestamp = get_trace_start_time(TRACE_FILE)
    print(f"仿真起始时间戳为: {start_timestamp}")

    simulation_duration = 1209600 # 仿真持续时间，单位秒
    time_step = 60             # 步长
    
    current_time = start_timestamp
    end_time = start_timestamp + simulation_duration

    if len(mc.sdcs) > 0:
        print(f"成功加载 {len(mc.sdcs)} 个源站 (SDC)。")
    else:
        print("错误: 没有加载到 SDC，无法运行。")
        return

    global_stats = {'total_hits': 0}

    all_request_latencies = []

    # 记录每个时间步的平均延迟，用于画趋势图
    history_avg_latencies = []
    history_timestamps = []

    while current_time < end_time:
        print(f"\n[Time: {current_time:.1f}] 正在更新拓扑...")
        
        # 更新拓扑
        mc.update_topology(current_time)
        print(f"  - 图规模: {mc.graph.number_of_nodes()} 节点, {mc.graph.number_of_edges()} 边")

        # 加载 CDN Trace 数据
        requests = mc.load_trace_batch(TRACE_FILE, current_time, time_step)
        print(f"  - 本轮 Trace 请求数: {len(requests)}")

        current_step_latencies = process(mc, requests, current_time, global_stats, user_coord_iter)

        if current_step_latencies:
            all_request_latencies.extend(current_step_latencies)

            # 计算本轮平均值并记录
            avg_step_lat = sum(current_step_latencies) / len(current_step_latencies)
            history_avg_latencies.append(avg_step_lat * 1000) # 转换为 ms
            history_timestamps.append(current_time - start_timestamp)
            print(f"  > 本轮平均延迟: {avg_step_lat * 1000:.2f} ms")

        # 时间步进
        current_time += time_step

    print("仿真结束")
    print(f"最终总命中次数: {global_stats['total_hits']}")

    if all_request_latencies:
        # 转换为毫秒
        latencies_ms = [t * 1000 for t in all_request_latencies]
        
        # 平均延迟
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        
        # 最小/最大延迟
        min_latency = min(latencies_ms)
        max_latency = max(latencies_ms)
        
        # 百分位延迟
        latencies_ms.sort()
        p50_idx = int(len(latencies_ms) * 0.5)
        p95_idx = int(len(latencies_ms) * 0.95)
        p99_idx = int(len(latencies_ms) * 0.99)
        
        p50_latency = latencies_ms[p50_idx] # 中位数
        p95_latency = latencies_ms[p95_idx] # 95% 的请求延迟都低于此值
        p99_latency = latencies_ms[p99_idx] # 99% 的请求延迟都低于此值

        print(f"统计请求总数: {len(latencies_ms)}")
        print(f"平均延迟 (Avg): {avg_latency:.2f} ms")
        print(f"中位数延迟 (P50): {p50_latency:.2f} ms")
        print(f"95%延迟 (P95): {p95_latency:.2f} ms")
        print(f"99%延迟 (P99): {p99_latency:.2f} ms")
        print(f"最小延迟 (Min): {min_latency:.2f} ms")
        print(f"最大延迟 (Max): {max_latency:.2f} ms")
    else:
        print("没有成功处理任何请求，无法计算延迟统计。")

    # 引入字体管理器
    import matplotlib.font_manager as fm
    
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')

    # 定义系统常见的中文自体文件路径列表 (按优先级)
    mac_font_paths = [
        '/System/Library/Fonts/STHeiti Light.ttc',         # 华文黑体-轻
        r'C:\Windows\Fonts\simhei.ttf',    # SimHei (黑体) 
        r'C:\Windows\Fonts\msyh.ttc',      # Microsoft YaHei (微软雅黑)
        r'C:\Windows\Fonts\simsun.ttc',    # SimSun (宋体)
        r'C:\Windows\Fonts\kaiu.ttf',      # KaiTi (楷体)
        r'C:\Windows\Fonts\Deng.ttf',      # DengXian (等线)
    ]
    
    # 遍历查找存在的字体文件
    selected_font_path = None
    for f_path in mac_font_paths:
        if os.path.exists(f_path):
            selected_font_path = f_path
            break
            
    # 加载字体
    if selected_font_path:
        # 将字体文件加入 Matplotlib 管理器
        fm.fontManager.addfont(selected_font_path)
        # 获取该字体的内部名称
        prop = fm.FontProperties(fname=selected_font_path)
        custom_font_name = prop.get_name()
        
        # 设置全局默认字体
        plt.rcParams['font.sans-serif'] = [custom_font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print("未找到任何系统中文字体文件")
 
    # 端到端延迟 CDF (累积分布函数) 
    # 展示有多少比例的请求延迟低于某个值 
    if all_request_latencies:
        plt.figure(figsize=(8, 6))
        # 转换为毫秒
        lat_data_ms = np.array(all_request_latencies) * 1000
        # 排序
        sorted_data = np.sort(lat_data_ms)
        # 计算 y 轴 (0 ~ 1)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        plt.plot(sorted_data, yvals, linewidth=2, color='darkblue')
        plt.xlabel('端到端延迟 (ms)', fontsize=12)
        plt.ylabel('累积概率 (CDF)', fontsize=12)
        plt.title('用户请求延迟累积分布图', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 标出 P95 线
        p95_val = sorted_data[int(len(sorted_data)*0.95)]
        plt.axvline(p95_val, color='red', linestyle='--', alpha=0.5, label=f'P95: {p95_val:.1f}ms')
        plt.legend()
        plt.savefig('Result_Latency_CDF.png', dpi=300)
        print("  - 已保存: Result_Latency_CDF.png")
        plt.close()

    # 平均延迟随时间变化趋势
    if history_avg_latencies:
        plt.figure(figsize=(10, 5))
        plt.plot(history_timestamps, history_avg_latencies, marker='o', markersize=4, linestyle='-', color='teal')
        plt.xlabel('仿真时间 (s)', fontsize=12)
        plt.ylabel('平均延迟 (ms)', fontsize=12)
        plt.title('网络平均延迟随时间变化趋势', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('Result_Latency_Trend.png', dpi=300)
        print("  - 已保存: Result_Latency_Trend.png")
        plt.close()

    
    plt.show()

if __name__ == "__main__":
    run_simulation()