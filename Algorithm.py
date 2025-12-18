from Classes import User
import networkx as nx

def mlc3(mc, requests, stats, user_coord_iterator):
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