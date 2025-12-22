from Classes import User
import math
import random

def get_activity_weight(lon, current_utc_timestamp):
    """
    根据经度和当前 UTC 时间，计算当地活跃度权重 (0.1 ~ 1.0)
    模拟"日出而作，日落而息"的互联网潮汐效应
    """
    # 计算当地时间
    utc_seconds = current_utc_timestamp % 86400
    utc_hour = utc_seconds / 3600.0
    
    # 经度偏移: 东经加，西经减
    offset = lon / 15.0
    local_hour = (utc_hour + offset) % 24
    
    # 定义活跃度曲线
    if 0 <= local_hour < 6:
        return 0.1  # 深夜睡眠 (权重极低)
    elif 6 <= local_hour < 9:
        return 0.4  # 早起 (逐渐上升)
    elif 9 <= local_hour < 18:
        return 0.8  # 工作时间 (平稳)
    elif 18 <= local_hour < 23:
        return 1.0  # 晚高峰 (最高)
    else: # 23 - 24
        return 0.5  # 睡前

def mlc3(mc, requests, current_time, stats, user_coords_list, gs_history_recorder):
    """
    处理当前时间步的所有请求
    """
    potential_actions = []

    step_latencies = []

    aggregation_map = {} # 聚合字典，用于累加延迟数据

    # 计算用户权重
    weights = []
    for (lat, lon) in user_coords_list:
        w = get_activity_weight(lon, current_time)
        weights.append(w)

    # 批量抽样
    assigned_locations = random.choices(user_coords_list, weights=weights, k=len(requests))

    for req, (u_lat, u_lon) in zip(requests, assigned_locations):
        content = req['content']
        # 更新内容热度
        content.update_popularity(1)

        # 创建临时 User 对象
        temp_user = User(f"User_{u_lat:.2f}_{u_lon:.2f}", u_lat, u_lon)
        
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

        SPEED_OF_LIGHT = 299792.458
        LATENCY_THRESHOLD = 0.050
        if access_sat.id in mc.graph:
            for neighbor_id in mc.graph.neighbors(access_sat.id):
                # 判断邻居是不是地面站
                if neighbor_id in mc.ground_stations:
                    # 获取 卫星->地面站 的距离权重
                    dist_sat_gs = mc.graph[access_sat.id][neighbor_id]['weight']
                    lat_sat_gs = dist_sat_gs / SPEED_OF_LIGHT
                    
                    # 计算总延迟: 用户->卫星 + 卫星->地面站
                    total_local_latency = first_hop_latency + lat_sat_gs
                    
                    if total_local_latency <= LATENCY_THRESHOLD:
                        if neighbor_id not in gs_history_recorder:
                            gs_history_recorder[neighbor_id] = []
                        
                        gs_history_recorder[neighbor_id].append({
                            'content': req['content'],
                            'local_latency': total_local_latency
                        })

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

def execute_tier3_caching(mc, gs_history_recorder):
    """
    【新增函数】Tier-3 地面站缓存决策
    基于收集到的历史请求进行打分 (Equation 4)
    """
    for gs_id, history_records in gs_history_recorder.items():
        if gs_id not in mc.ground_stations:
            continue
            
        gs = mc.ground_stations[gs_id]
        total_records = len(history_records)
        if total_records == 0:
            continue

        # 聚合统计 (统计每个内容在历史中出现了几次，平均本地延迟是多少)
        content_stats = {}
        for record in history_records:
            content = record['content']
            lat = record['local_latency']
            
            if content not in content_stats:
                content_stats[content] = {'count': 0, 'sum_lat': 0.0}
            
            content_stats[content]['count'] += 1
            content_stats[content]['sum_lat'] += lat
        
        # 计算打分
        scores = []
        for content, stats in content_stats.items():
            term1 = stats['count'] / total_records
            
            avg_local_lat = stats['sum_lat'] / stats['count']
            avg_remote_lat = 0.200 # 假设远程延迟为 200ms

            term2 = avg_remote_lat / avg_local_lat if avg_local_lat > 0 else 1.0 
            
            term3 = content.size / gs.capacity
            
            final_score = term1 * term2 * term3
            scores.append((final_score, content))
        
        # 排序并执行缓存
        scores.sort(key=lambda x: x[0], reverse=True)
        
        for score, content in scores:
            gs.cache_content(content)
