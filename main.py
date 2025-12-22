from Classes import MegaConstellation
from config import get_config
from Algorithm import mlc3, execute_tier3_caching
from Utils import load_user_coordinates, get_trace_start_time
import itertools
import Visualization


def run_simulation():
    # 初始化环境
    print("正在初始化 MegaCacheX 仿真环境")
    
    cfg = get_config()
    mc = MegaConstellation(
        sat_file=cfg['paths']['tle_sat'], 
        sdc_file=cfg['paths']['tle_sdc'],
        gs_file=cfg['paths']['gs_nodes'],
    )
    
    user_coords_list = load_user_coordinates(cfg['paths']['user_nodes'])
    user_coord_iter = itertools.cycle(user_coords_list)

    # 获取 CSV 文件中的开始时间
    start_timestamp = get_trace_start_time(cfg['paths']['trace_file'])
    print(f"仿真起始时间戳为: {start_timestamp}")

    trace_file = cfg['paths']['trace_file'] # CDN Trace 文件路径
    simulation_duration = cfg['simulation']['duration'] # 仿真持续时间，单位秒
    time_step = cfg['simulation']['step']             # 步长
    
    current_time = start_timestamp
    end_time = start_timestamp + simulation_duration

    gs_history_recorder = {} # 全局的地面站请求历史记录器
    history_avg_latencies = [] # 每个时间步的平均延迟 (ms)
    all_request_latencies = [] # 所有请求的延迟 (秒)
    global_stats = {'total_hits': 0} # 全局统计命中次数

    MAX_HISTORY_PER_GS = 5000

    while current_time < end_time:
        print(f"\n[Time: {current_time:.1f}] 正在更新拓扑...")
        
        # 更新拓扑
        mc.update_topology(current_time)
        print(f"  - 图规模: {mc.graph.number_of_nodes()} 节点, {mc.graph.number_of_edges()} 边")

        # 加载 CDN Trace 数据
        requests = mc.load_trace_batch(trace_file, current_time, time_step)
        print(f"  - 本轮 Trace 请求数: {len(requests)}")

        current_step_latencies = mlc3(mc, requests, global_stats, user_coord_iter, gs_history_recorder)

        execute_tier3_caching(mc, gs_history_recorder)

        # 滑动窗口，限制历史记录大小，防止内存爆炸
        for gs_id in gs_history_recorder:
            if len(gs_history_recorder[gs_id]) > MAX_HISTORY_PER_GS:
                # 截断，只保留最近的 N 条
                gs_history_recorder[gs_id] = gs_history_recorder[gs_id][-MAX_HISTORY_PER_GS:]

        if current_step_latencies:
            all_request_latencies.extend(current_step_latencies)

            # 计算本轮平均值并记录
            avg_step_lat = sum(current_step_latencies) / len(current_step_latencies)
            history_avg_latencies.append(avg_step_lat * 1000) # 转换为 ms
            print(f"  > 本轮平均延迟: {avg_step_lat * 1000:.2f} ms")

        # 时间步进
        current_time += time_step

    print("仿真结束")
    print(f"最终总命中次数: {global_stats['total_hits']}")

    # 可视化结果
    viz = Visualization.ResultVisualizer()
    viz.plot_latency_cdf(all_request_latencies, save_name=cfg['paths']['plot_save_cdf']) # 画延迟 CDF

if __name__ == "__main__":
    run_simulation()