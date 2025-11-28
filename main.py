import os
import random
from Classes import MegaConstellation, Satellite, GroundStation

def run_simulation():
    print(">>> 初始化 MegaCacheX 仿真系统...")
    system = MegaConstellation()
    
    # 1. 加载 TLE 数据 (Tier-1 & Tier-2)
    print("--- 加载卫星轨道数据 ---")
    if os.path.exists('configuration/S_constellation.tle'):
        system.load_tle_file('configuration/S_constellation.tle', 'satellite')
        print(f"已加载 Tier-2 卫星: {len(system.satellites)} 颗")
    
    if os.path.exists('configuration/SDC_constellation.tle'):
        system.load_tle_file('configuration/SDC_constellation.tle', 'sdc')
        print(f"已加载 Tier-1 SDC: {len(system.sdcs)} 个")

    # 2. 加载地面站数据 (Tier-3)
    print("--- 加载地面节点数据 ---")
    if os.path.exists('configuration/GS_Starlink_Nodes.csv'):
        system.load_gs_csv('configuration/GS_Starlink_Nodes.csv')
        print(f"已加载 Tier-3 地面站: {len(system.ground_stations)} 个")
    
    # (可选) 加载 GDC 数据，可视作超级地面站
    # if os.path.exists('GDC_Globalping_Nodes.csv'):
    #     system.load_gs_csv('GDC_Globalping_Nodes.csv')
    #     print("已加载 GDC 节点补充地面覆盖")

    # 3. 仿真参数设置
    START_TIME = 1424534404 # 对应 Trace 中的起始时间戳
    DURATION_PER_SLOT = 60  # 每个时间片 60秒
    TOTAL_SLOTS = 5         # 跑 5 分钟 (演示用)

    print(f"\n>>> 开始仿真循环 (Start: {START_TIME}, Slots: {TOTAL_SLOTS})")
    
    for t in range(TOTAL_SLOTS):
        current_time = START_TIME + t * DURATION_PER_SLOT
        
        print(f"\n[Time Slot {t}] 正在更新全网拓扑 (Timestamp={current_time})...")
        system.update_topology(current_time)
        
        # B. 读取 Trace
        requests = system.load_trace_batch('StarFront_CDN_Trace_Dataset.csv', current_time, DURATION_PER_SLOT)
        print(f"当前时间片请求数: {len(requests)}")
        
        # C. 处理请求 (简易演示)
        hits = 0
        for req in requests:
            content = req['content']
            
            # 随机指派一个地面站
            if not system.ground_stations: break
            user_gs_id = random.choice(list(system.ground_stations.keys()))
            user_gs = system.ground_stations[user_gs_id]
            
            # 检查本地缓存
            if content.id in user_gs.cached_contents:
                hits += 1
                continue
            
            # 路由演示
            if system.sdcs:
                target_sdc_id = list(system.sdcs.keys())[0]
                path = system.ospc_routing(user_gs_id, target_sdc_id)
            
        print(f"Slot {t} 完成: Hits={hits}, Req={len(requests)}")

    print("\n>>> 仿真结束")

if __name__ == "__main__":
    run_simulation()