import os
import math
import csv
import heapq
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from collections import OrderedDict

# 物理常数
SPEED_OF_LIGHT = 299792.458  # km/s (c)
SPEED_OF_FIBER = SPEED_OF_LIGHT * 0.667  # 光纤光速 (km/s)
EARTH_RADIUS = 6371.0        # km

# 链路带宽配置 (单位: MB/s)
BANDWIDTH_ISL = 1250.0   # 星间链路 (10 Gbps)
BANDWIDTH_GSL = 62.5     # 星地链路 (500 Mbps)
BANDWIDTH_U = 12.5       # 用户链路 (100Mbps)
BANDWIDTH_IGL = 2500.0   # 地面光纤 (20 Gbps)


# 参考论文 Table I 的成本参数 (归一化 0.0 - 1.0)
COST_PARAMS = {
    'lambda_1': 1.0,  # S存储成本系数
    'lambda_2': 0.6,  # SDC存储成本系数
    'beta_1': 0.3,    # GS存储成本系数
    'beta_2': 0.1,    # GDC存储成本系数
    'theta': 0.5,     # ISL 传输成本系数
    'eta': 1.0,       # GSL 传输成本系数
    'xi': 0.1,        # IGL 传输成本系数
}

class Content:
    def __init__(self, content_id, size):
        self.id = content_id
        self.size = float(size) / (1024*1024) # MB
        self.popularity = 0.0
        self.history_popularity = []
    
    def update_popularity(self, req_count, alpha=0.7, theta=0.9): 
        history_total = 0.0
        for i, past_pop in enumerate(reversed(self.history_popularity)):
            
            weight = theta ** (i + 1)
            history_total += past_pop * weight

            if weight < 0.001:
                break

        self.popularity = (alpha * req_count) + ((1 - alpha) * history_total)
        self.history_popularity.append(self.popularity) # 存入历史记录

    def __hash__(self):
        return hash(self.id) # 使用 id 计算哈希值
    
    def __eq__(self, other):
        if isinstance(other, Content):
            return self.id == other.id # 若两个对象的 id 相同，则两个对象是一个东西
        return False

class Node:
    def __init__(self, node_id, node_type, capacity):
        self.id = node_id
        self.type = node_type
        self.capacity = capacity # 存储容量
        self.used_storage = 0.0 # 已使用的存储空间
        self.position = np.array([0., 0., 0.])
        self.cached_contents = OrderedDict()

    def get_distance(self, other):
        return np.linalg.norm(self.position - other.position)
        # 数组减法求差值向量，并求范数

    def has_space(self, size):
        return (self.used_storage + size) <= self.capacity
        # 根据大小判断内容能否被缓存
    
    def cache_content(self, content):
        # 命中
        if content in self.cached_contents:
            # 移动到字典末尾，标记为"最近刚刚使用过"
            self.cached_contents.move_to_end(content)
            return True

        # 未命中，容量不够，删除最老的数据
        while not self.has_space(content.size):
            # 文件大于卫星总容量
            if not self.cached_contents:
                return False
            
            # 弹出字典最开头的元素（最久未使用的）
            removed_content, _ = self.cached_contents.popitem(last=False)
            self.used_storage -= removed_content.size
            
        # 存入新内容，放在字典末尾
        self.cached_contents[content] = None 
        self.used_storage += content.size
        return True

class User(Node):
    def __init__(self, user_id, lat, lon):
        super().__init__(user_id, 'user', capacity=0)
        
        # 将经纬度坐标转换为ECEF（地固系）坐标
        rad_lat, rad_lon = math.radians(lat), math.radians(lon)
        self.position[0] = EARTH_RADIUS * math.cos(rad_lat) * math.cos(rad_lon)
        self.position[1] = EARTH_RADIUS * math.cos(rad_lat) * math.sin(rad_lon)
        self.position[2] = EARTH_RADIUS * math.sin(rad_lat)

class Satellite(Node):
    def __init__(self, sat_id, tle_line1, tle_line2, capacity=1024*10):
        super().__init__(sat_id, 'satellite', capacity)
        self.satrec = Satrec.twoline2rv(tle_line1, tle_line2) # 将TLE变为可计算的数学模型对象
        self.storage_cost = COST_PARAMS['lambda_1'] # 卫星存储数据的单位代价

    def update_position(self, current_timestamp):
        """
        利用 SGP4 计算指定时间戳的位置
        并从 TEME (惯性系) 转换为 ECEF (地固系)
        (TEME 是准惯性系, ECEF 随地球转)
        """
        dt = datetime.fromtimestamp(current_timestamp)
        # jday 函数把年月日时分秒转换成两个浮点数：jd(整日)和 fr(小数日)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
        
        # 输入时间，satrec 对象自动根据 TLE 算出这一刻的位置(r，单位km)和速度(v，单位km/s)，e为错误码
        e, r, v = self.satrec.sgp4(jd, fr)
        
        if e != 0:
            # 错误码不为0则计算错误，放弃
            return 

        x_teme, y_teme, z_teme = r
        
        # 坐标转换: TEME -> ECEF
        d = (jd - 2451545.0) + fr
        gmst_hours = 18.697374558 + 24.06570982441908 * d
        gmst_rad = (gmst_hours % 24) * 15 * (math.pi / 180) # 将时间转换为弧度
        
        # 旋转矩阵所需正余弦
        cos_theta = math.cos(gmst_rad)
        sin_theta = math.sin(gmst_rad)
        
        # 绕 Z 轴旋转
        x_ecef = x_teme * cos_theta + y_teme * sin_theta
        y_ecef = -x_teme * sin_theta + y_teme * cos_theta
        z_ecef = z_teme # 地球绕 Z 轴自转，故不变
        
        self.position = np.array([x_ecef, y_ecef, z_ecef])

    def satellite_score(self, content, latency_sat, latency_gs):
        current_total_pop = 0.0
        for cached_obj in self.cached_contents:
            current_total_pop += cached_obj.popularity

        term1 = content.popularity / current_total_pop if current_total_pop > 0 else 1.0
        # 该内容的流行度 / 该卫星上全部内容的流行度之和

        term2 = latency_sat / latency_gs if latency_gs > 0 else 1.0
        # 通过太空路径访问的延迟 / 通过地面站访问的延迟

        term3 = content.size / self.capacity
        # 内容大小 / 卫星总容量

        return term1 * term2 * term3

class SpaceDataCenter(Satellite):
    def __init__(self, sdc_id, tle_line1, tle_line2):
        # SDC 使用同样的 SGP4 逻辑
        super().__init__(sdc_id, tle_line1, tle_line2, capacity=float('inf'))
        self.type = 'sdc'
        self.storage_cost = COST_PARAMS['lambda_2']

class GroundStation(Node):
    def __init__(self, gs_id, lat, lon, capacity=1024*100):
        super().__init__(gs_id, 'ground_station', capacity)
        self.storage_cost = COST_PARAMS['beta_1']
        # 将经纬度坐标转换为ECEF（地固系）坐标
        rad_lat, rad_lon = math.radians(lat), math.radians(lon)
        self.position[0] = EARTH_RADIUS * math.cos(rad_lat) * math.cos(rad_lon)
        self.position[1] = EARTH_RADIUS * math.cos(rad_lat) * math.sin(rad_lon)
        self.position[2] = EARTH_RADIUS * math.sin(rad_lat)

class GroundDataCenter(Node):
    def __init__(self, gdc_id, lat, lon):
        # GDC 视为无限存储源站
        super().__init__(gdc_id, 'gdc', capacity=float('inf'))
        self.storage_cost = COST_PARAMS['beta_2']
        
        # 将经纬度坐标转换为ECEF（地固系）坐标
        rad_lat, rad_lon = math.radians(lat), math.radians(lon)
        self.position[0] = EARTH_RADIUS * math.cos(rad_lat) * math.cos(rad_lon)
        self.position[1] = EARTH_RADIUS * math.cos(rad_lat) * math.sin(rad_lon)
        self.position[2] = EARTH_RADIUS * math.sin(rad_lat)

class MegaConstellation:
    def __init__(self):
        self.graph = nx.Graph()
        self.satellites = {}
        self.sdcs = {}
        self.ground_stations = {}
        self.contents = {} 
    
    def load_tle_file(self, filename, node_type):
        """加载TLE文件"""
        with open(filename, 'r') as f:
            lines = f.readlines() # 将整个文件的每一行作为一个字符串，存入列表 lines
        
        # 检查行数是否符合 3 的倍数
        if len(lines) % 3 != 0:
            print(f"Warning: File {filename} has {len(lines)} lines, which is not divisible by 3. Some data may be lost.")

        # TLE 标准格式通常是3行一组，每次循环跳3行
        for i in range(0, len(lines), 3): 
            name = lines[i].strip() # 提取名称
            l1 = lines[i+1].strip() # 提取第一行轨道参数
            l2 = lines[i+2].strip() # 提取第二行轨道参数
            
            if node_type == 'satellite':
                node = Satellite(name, l1, l2) # 创建 Satellite 对象
                self.satellites[name] = node # 存入 self.satellites 字典，name 为 key，node 为 value
            elif node_type == 'sdc':
                node = SpaceDataCenter(name, l1, l2)
                self.sdcs[name] = node
            else:
                raise ValueError(f"Unknown node_type: {node_type}") # 类型不对直接报错
            
            # 将节点添加到 NetworkX 图中
            self.graph.add_node(name, type=node_type) 
            
    def load_gs_csv(self, filename):
        """加载GS节点"""
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                name = row[0]
                try:
                    lat = float(row[1])
                    lon = float(row[2])
                    gs = GroundStation(name, lat, lon)
                    self.ground_stations[name] = gs
                    self.graph.add_node(name, type='ground_station')
                except (ValueError, IndexError):
                    continue

    def _start_offset(self, f, target_time: float, file_size: int) -> int:
        """
        二分法查找对应时间戳开始的行，作为加载 Trace 文件的辅助
        """
        low = 0
        high = file_size
        best_offset = file_size # 默认没找到就指向文件末尾

        while low < high:
            mid = (low + high) // 2

            f.seek(mid) # 瞬移到文件中间某个字节
            
            if mid > 0:
                f.readline() # 读完该行，下一行为完整开始
            
            current_offset = f.tell() # 记录完整行的起始位置
            
            if current_offset >= file_size:
                # 如果跳过一行直接跳出了文件，说明这一段太靠后了
                high = mid
                continue

            line = f.readline()
            if not line:
                high = mid
                continue

            try:
                # 解析这一行的时间戳
                parts = line.split(',')
                if len(parts) < 2: 
                    # 坏数据，保守处理，当作时间太小，往后找
                    low = mid + 1
                    continue
                
                current_ts = float(parts[1])

                if current_ts >= target_time:
                    # 找到了一个符合条件的时间，但不确定是不是"第一个"
                    # 记录下这个位置，尝试往左边找找看有没有更早的
                    best_offset = current_offset
                    high = mid
                else:
                    # 时间太早了，往右边找
                    low = mid + 1

            except (ValueError, IndexError):
                low = mid + 1
        
        return best_offset

    def load_trace_batch(self, filename: str, time_window_start: float, duration: float):
        """
        读取trace文件，返回时间戳、内容类
        """
        requests = []
        time_window_end = time_window_start + duration # 不在循环中计算
        
        file_size = os.path.getsize(filename) # 获取文件总字节大小

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                start_offset = self._start_offset(f, time_window_start, file_size)
                
                f.seek(start_offset) # 将文件指针移动到计算出的最佳位置
                
                reader = csv.reader(f) # 顺序读取
                
                for row in reader:
                    if not row: continue # 基本校验
                    
                    try: 
                        ts = float(row[1]) # 时间戳
                    except (ValueError, IndexError):
                        continue

                    if ts >= time_window_end: # 卫语句
                        break 
                    if ts < time_window_start:
                        continue

                    c_id = row[2] # 内容ID
                    c_size = float(row[3]) # 内容大小，单位B(Byte)
                    
                    if c_id not in self.contents:
                        self.contents[c_id] = Content(c_id, c_size) # 存入字典，c_id 为 key ，content 类为 value
                    
                    requests.append({
                        'time': ts, 
                        'content': self.contents[c_id] # 列表中为整个内容类，可通过ID查找
                    })
                    
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return []
            
        return requests

    def update_topology(self, current_timestamp):
        """更新全网拓扑，使用 KD-Tree 实现 O(N log N) 的近邻搜索"""
        self.graph.clear_edges()
        
        sat_objs = list(self.satellites.values()) # 打包所有卫星值，无索引
        sat_coords = [] # 
        sat_ids = [] # 

        for sat in sat_objs:
            sat.update_position(current_timestamp)
            sat_coords.append(sat.position) # 收集卫星坐标 [x, y, z]
            sat_ids.append(sat.id)          # 收集ID

        sat_coords_np = np.array(sat_coords) # 转为 numpy 数组，(N, 3)

        # 构建KD-Tree
        tree = cKDTree(sat_coords_np)

        # 查询最近邻，实现+Grid拓扑
        # k=5 ：找最近的 5 个点，对应自己和最近的 4 个卫星
        # query 返回两个矩阵：
        # dists: 距离矩阵 (N, 5)，每个卫星与其它卫星最近的距离中，提取前5个
        # idxs:  邻居在 sat_objs 列表里的下标 (N, 5)
        dists, idxs = tree.query(sat_coords_np, k=5)

        # 遍历每颗卫星的查询结果
        for i in range(len(sat_objs)):
            src_id = sat_ids[i]
            
            # 遍历它找到的 4 个邻居
            # j 是邻居在 dists/idxs 这一行的列索引 (1到4)
            for j in range(1, 5): 
                neighbor_idx = idxs[i][j]
                dist = dists[i][j]
                
                # 设置 5000km 为最大通信距离
                if dist < 5000:
                    neighbor_id = sat_ids[neighbor_idx]
                    
                    # 添加边，NetworkX 自动去重，边的权重为卫星之间的物理距离
                    self.graph.add_edge(src_id, neighbor_id, weight=dist, type='ISL')

        # 连接 SDC
        if self.sdcs:
            sdc_objs = list(self.sdcs.values())
            for sdc in sdc_objs:
                sdc.update_position(current_timestamp)
                
                # k=1：离 SDC 最近的 1 个卫星
                dist, idx = tree.query(sdc.position, k=1)
                
                # 设置 5000km 为最大通信距离
                if dist < 5000:
                    target_sat_id = sat_ids[idx]
                    self.graph.add_edge(sdc.id, target_sat_id, weight=dist, type='ISL')
        
        # 连接 GS
        if self.ground_stations:
            for gs in self.ground_stations.values():
                # k=1：离 GS 最近的 1 个卫星
                dist, idx = tree.query(gs.position, k=1)
                
                # 设置 2500km 为地面站最大通信距离
                if dist < 2500: 
                    target_sat_id = sat_ids[idx]
                    self.graph.add_edge(gs.id, target_sat_id, weight=dist, type='GSL')

    def get_link_delay(self, u, v, content_size):
        """
        辅助函数：计算两点间特定内容的传输总延迟
        """
        edge = self.graph[u][v]
        dist = edge['weight']
        link_type = edge['type']
        
        if link_type == 'ISL':
            bandwidth = BANDWIDTH_ISL
            prop_speed = SPEED_OF_LIGHT
        elif link_type == 'GSL':
            bandwidth = BANDWIDTH_GSL
            prop_speed = SPEED_OF_LIGHT
        elif link_type == 'UserLink':
            bandwidth = BANDWIDTH_U
            prop_speed = SPEED_OF_LIGHT
        else: # IGL
            bandwidth = BANDWIDTH_IGL
            prop_speed = SPEED_OF_FIBER

        # 传播延迟 (物理距离 / 速度)
        prop_delay = dist / prop_speed

        # 传输延迟 (内容大小 / 带宽)
        if content_size > 0:
            trans_delay = content_size / bandwidth
        else:
            trans_delay = 0.0
            
        return prop_delay + trans_delay

    def ospc_routing(self, v_start, v_end, content_size):
        """OSPC路由"""
        # 边界检查
        if v_start not in self.graph or v_end not in self.graph:
            return None

        # 所有节点距离赋值为无穷
        delay = {node: float('inf') for node in self.graph.nodes()}
        # 
        prev = {node: None for node in self.graph.nodes()}
        delay[v_start] = 0.0

        Q = [(0.0, v_start)] # 创建优先列表

        while Q:
            # 弹出堆中延迟最小的节点
            current_delay, u = heapq.heappop(Q)

            if current_delay > delay[u]:
                continue
            if u == v_end: # 终点
                break

            for v in self.graph.neighbors(u):
                
                edge_delay = self.get_link_delay(u, v, content_size)
                
                alt = delay[u] + edge_delay

                if alt < delay[v]:
                    delay[v] = alt
                    prev[v] = u
                    
                    heapq.heappush(Q, (alt, v))

        path = []
        curr = v_end
        
        if prev[curr] is None and curr != v_start:
            return None # 无法到达

        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        
        path.reverse() # 反转列表
        
        return path
