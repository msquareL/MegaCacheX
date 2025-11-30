import os
import sys
import time
import csv  # 引入csv库用于预读取时间
from datetime import datetime
from Classes import MegaConstellation, COST_PARAMS

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

def run_simulation():
    # 1. 初始化环境
    print("-" * 30)
    print("正在初始化 MegaCacheX 仿真环境 (使用自定义数据)...")
    
    mc = MegaConstellation()
    
    # 2. 加载您的自定义节点文件
    # 检查文件是否存在，防止路径错误
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

    # 3. 时间设置 (关键步骤)
    # 自动获取 CSV 里的开始时间，而不是用 time.time()
    start_timestamp = get_trace_start_time(TRACE_FILE)
    print(f"仿真起始时间戳已对齐为: {start_timestamp}")

    simulation_duration = 600 # 例如：跑 600 秒 (10分钟) 的数据
    time_step = 10            # 每 10 秒更新一次拓扑
    
    current_time = start_timestamp
    end_time = start_timestamp + simulation_duration

    # 假设第一个 SDC 作为源站 (Tier-1)
    # 如果您的 SDC TLE 文件里名字不是 "SDC-1"，请修改这里
    if len(mc.sdcs) > 0:
        source_node_id = list(mc.sdcs.keys())[0] 
        print(f"源站设定为: {source_node_id}")
    else:
        print("错误: 没有加载到 SDC，无法运行。")
        return

    # =====================
    # 4. 主循环
    # =====================
    while current_time < end_time:
        print(f"\n[Time: {current_time:.1f}] 正在更新拓扑...")
        
        # 4.1 更新拓扑
        mc.update_topology(current_time)
        print(f"  - 图规模: {mc.graph.number_of_nodes()} 节点, {mc.graph.number_of_edges()} 边")

        # 4.2 加载您的 CDN Trace 数据
        # 【修改点】这里传入的是 TRACE_FILE 变量
        requests = mc.load_trace_batch(TRACE_FILE, current_time, time_step)
        print(f"  - 本轮 Trace 请求数: {len(requests)}")

        # 4.3 处理请求 (MLC3 逻辑)
        for req in requests:
            # req 结构: {'time': ts, 'content': ContentObj}
            content = req['content']
            
            # 更新热度
            content.update_popularity(1)
            
            # 确定接入点 (这里暂时随机选一个 GS，或者写死一个做测试)
            # 在未来您可以根据请求里是否包含经纬度来动态匹配最近的 GS
            if len(mc.ground_stations) > 0:
                user_access_node_id = list(mc.ground_stations.keys())[0] # 取第一个 GS
            else:
                continue

            # 路由计算
            target_id = source_node_id
            path = mc.ospc_routing(user_access_node_id, target_id, content.size)
            
            if not path:
                # print(f"    [不可达] {content.id}")
                continue
                
            # print(f"    [Route] {content.id} size={content.size:.1f}MB: {len(path)} hops")

            # 缓存决策
            # 估算路径延迟 (Latency Sat)
            latency_sat = 0.0
            for i in range(len(path)-1):
                # 调用 Classes.py 里的 get_link_delay
                latency_sat += mc.get_link_delay(path[i], path[i+1], content.size)
            
            latency_gs = 0.200 # 基准地面延迟

            # 遍历路径做缓存
            for node_id in path[1:-1]:
                if node_id in mc.satellites:
                    sat_node = mc.satellites[node_id]
                    
                    if content in sat_node.cached_contents:
                        continue
                    
                    # 计算分数
                    score = sat_node.satellite_score(content, latency_sat, latency_gs)
                    
                    if score > 0.8: # 阈值
                        if sat_node.cache_content(content):
                            print(f"      [Cache Success] {sat_node.id} <- {content.id} (Score: {score:.2f})")

        # 时间步进
        current_time += time_step

    print("-" * 30)
    print("仿真结束。")

if __name__ == "__main__":
    run_simulation()