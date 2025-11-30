import os
import sys
import time
import csv 
import random
from datetime import datetime
from Classes import MegaConstellation, User, COST_PARAMS

# 文件路径
TLE_SAT_FILE = "configuration/S_constellation.tle"
TLE_SDC_FILE = "configuration/SDC_constellation.tle"
GS_CSV_FILE  = "configuration/GS_Starlink_Nodes.csv"
TRACE_FILE   = "StarFront_CDN_Trace_Dataset.csv"

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

def MLC3(mc, requests, current_time, stats):
    """
    处理当前时间步的所有请求。
    流程：生成随机用户 -> 寻找接入卫星 -> 动态挂载用户到图 -> 路由(多源站) -> 缓存决策 -> 移除用户
    """
    for req in requests:
        content = req['content']
        # 更新内容热度
        content.update_popularity(1)
        
        # 生成随机用户并接入
        rand_lat = random.uniform(-90, 90)
        rand_lon = random.uniform(-180, 180)
        # rand_lat = 39.9
        # rand_lon = 116.4
        user_id = f"User_Temp_{time.time()}_{random.randint(1000,9999)}" # 确保ID唯一
        user = User(user_id, rand_lat, rand_lon)
        
        # 寻找最近接入卫星
        min_dist = float('inf')
        access_sat = None
        
        # 遍历所有卫星寻找最近的
        for sat in mc.satellites.values():
            d = user.get_distance(sat)
            # 1500km 视距限制
            if d < 1500 and d < min_dist:
                min_dist = d
                access_sat = sat
        
        if not access_sat:
            # print(f"    [无信号] 用户({rand_lat:.1f}, {rand_lon:.1f}) 无卫星覆盖")
            continue

        # 动态修改拓扑 (挂载用户)
        mc.graph.add_node(user.id, type='user')
        # 添加 UserLink，权重为物理距离
        mc.graph.add_edge(user.id, access_sat.id, weight=min_dist, type='UserLink') 

        # 多源站路由计算 (Tier-1 SDC Selection)
        best_path = None
        min_path_latency = float('inf')
        target_source_id = None

        # 遍历所有 SDC，寻找延迟最低的源站路径
        all_sources = list(mc.sdcs.keys())
        
        for potential_source in all_sources:
            # 路由起点：User
            temp_path = mc.ospc_routing(user.id, potential_source, content.size)
            
            if temp_path:
                # 计算全路径延迟
                path_latency = 0.0
                for i in range(len(temp_path)-1):
                    # 调用类内部的计算函数
                    path_latency += mc.get_link_delay(temp_path[i], temp_path[i+1], content.size)
                
                if path_latency < min_path_latency:
                    min_path_latency = path_latency
                    best_path = temp_path
                    target_source_id = potential_source

        # 缓存决策 (MLC3 Policy)
        if best_path:
            # min_path_latency 已经是包含了 User->Sat->...->SDC 的全链路延迟
            latency_sat = min_path_latency
            
            # 地面基准延迟 (可优化为计算值，这里暂用固定值)
            latency_gs = 0.200 

            best_candidate_node = None
            max_score = -1.0

            # 遍历路径上的中间卫星 (跳过 User 和 最终源站)
            for node_id in best_path[1:-1]:
                if node_id in mc.satellites:
                    sat_node = mc.satellites[node_id]
                    
                    # 如果已缓存，跳过
                    if content in sat_node.cached_contents:
                        best_candidate_node = None # 命中
                        stats['total_hits'] += 1
                        break
                    
                    # 计算分数 & 决策
                    score = sat_node.satellite_score(content, latency_sat, latency_gs)
                    
                    if score > max_score:
                        max_score = score
                        best_candidate_node = sat_node

            if best_candidate_node:
                if best_candidate_node.cache_content(content):
                    print(f"      [Cache] {user.id} -> 选中最佳节点 {best_candidate_node.id} 缓存 {content.id} (Score:{max_score:.2f})")
        # 移除用户节点，防止内存泄漏和图膨胀
        if user.id in mc.graph:
            mc.graph.remove_node(user.id)

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

    # 获取 CSV 文件中的开始时间
    start_timestamp = get_trace_start_time(TRACE_FILE)
    print(f"仿真起始时间戳为: {start_timestamp}")

    simulation_duration = 6000 # 仿真持续时间，单位秒
    time_step = 60             # 步长
    
    current_time = start_timestamp
    end_time = start_timestamp + simulation_duration

    if len(mc.sdcs) > 0:
        source_node_id = list(mc.sdcs.keys())[0] 
        print(f"源站设定为: {source_node_id}")
    else:
        print("错误: 没有加载到 SDC，无法运行。")
        return

    global_stats = {'total_hits': 0}

    while current_time < end_time:
        print(f"\n[Time: {current_time:.1f}] 正在更新拓扑...")
        
        # 更新拓扑
        mc.update_topology(current_time)
        print(f"  - 图规模: {mc.graph.number_of_nodes()} 节点, {mc.graph.number_of_edges()} 边")

        # 加载 CDN Trace 数据
        requests = mc.load_trace_batch(TRACE_FILE, current_time, time_step)
        print(f"  - 本轮 Trace 请求数: {len(requests)}")

        MLC3(mc, requests, current_time, global_stats)

        # 时间步进
        current_time += time_step

    print("仿真结束")
    print(f"仿真结束。最终总命中次数: {global_stats['total_hits']}")

if __name__ == "__main__":
    run_simulation()