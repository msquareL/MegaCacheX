import os
import math
import csv
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday

# --- 物理常数与全局配置 ---
SPEED_OF_LIGHT = 299792.458  # km/s (c)
SPEED_OF_FIBER = SPEED_OF_LIGHT * 0.67  # 光纤光速 (km/s)
EARTH_RADIUS = 6371.0        # km

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
        return hash(self.id) # 使用 id 计算哈希值，从而 set 可通过 id 去重
    
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
        self.cached_contents = set() # 缓存内容集合，具有去重功能

    def get_distance(self, other):
        return np.linalg.norm(self.position - other.position)
        # 数组减法求差值向量，并求范数

    def has_space(self, size):
        return (self.used_storage + size) <= self.capacity
        # 根据大小判断内容能否被缓存
    
    def cache_content(self, content):
        if self.has_space(content.size):
            self.cached_contents.add(content) # 缓存整个 content 类
            self.used_storage += content.size
            return True # 进行缓存
        return False

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
        
        # 3. 坐标转换: TEME -> ECEF
        # 为了计算与地面站(地固系)的距离，必须考虑地球自转角度 (GMST)
        # 简易计算格林尼治恒星时 (GMST) 角度 (弧度)
        # 公式参考: AA (Astronomical Algorithms) 简化版
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

class Link:
    def __init__(self, node_a, node_b, link_type):
        self.node_a = node_a
        self.node_b = node_b
        self.type = link_type
        
        if self.type == 'ISL':
            # 星间链路: 10 Gbps ~ 1250 MB/s
            self.bandwidth = 1250.0 
            self.trans_cost_factor = COST_PARAMS['theta']
            
        elif self.type == 'IGL': 
            # 地面光纤: 20 Gbps ~ 2500 MB/s
            self.bandwidth = 2500.0 
            self.trans_cost_factor = COST_PARAMS['xi']
            
        else: # GSL
            # 星地链路: 500 Mbps ~ 62.5 MB/s (瓶颈)
            self.bandwidth = 62.5  
            self.trans_cost_factor = COST_PARAMS['eta']

        if self.type == 'ISL' or self.type == 'GSL':
            self.prop_speed = SPEED_OF_LIGHT
            self.dist_factor = 1.0 
        else: # 地面之间
            self.prop_speed = SPEED_OF_FIBER
            self.dist_factor = 1.5   # 实际铺设长度通常是直线距离的 1.5 倍

    def get_latency(self, content_size):
        # 传输延迟 (内容大小 / 带宽)
        if content_size <= 0:
            trans_delay = 0
        else:
            trans_delay = content_size / self.bandwidth
        
        # 传播延迟 (物理距离 / 光速)
        straight_dist = self.node_a.get_distance(self.node_b)
        actual_dist = straight_dist * self.dist_factor 
        prop_delay = actual_dist / self.prop_speed

        return trans_delay + prop_delay

class MegaConstellation:
    def __init__(self):
        self.graph = nx.Graph()
        self.satellites = {}
        self.sdcs = {}
        self.ground_stations = {}
        self.contents = {} 
    
    def load_tle_file(self, filename, node_type):
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
                self.satellites[name] = node # 存入 self.satellites 字典
            elif node_type == 'sdc':
                node = SpaceDataCenter(name, l1, l2)
                self.sdcs[name] = node
            else:
                raise ValueError(f"Unknown node_type: {node_type}") # 类型不对直接报错
            
            # 将节点添加到 NetworkX 图中
            self.graph.add_node(name, type=node_type) 
            
    def load_gs_csv(self, filename):
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
                except ValueError:
                    continue

    def _start_offset(self, f, target_time: float, file_size: int) -> int:
        """
        二分法查找对应时间戳开始的行
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
        读取trace文件
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
                        self.contents[c_id] = Content(c_id, c_size)
                    
                    requests.append({
                        'time': ts,
                        'content': self.contents[c_id]
                    })
                    
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return []
            
        return requests

    def update_topology(self, current_timestamp):
        """
        更新全网拓扑，传入的是绝对时间戳 (current_timestamp)
        """
        self.graph.clear_edges()
        
        # 1. 更新轨道节点位置 (传入绝对时间)
        all_orbit_nodes = list(self.satellites.values()) + list(self.sdcs.values())
        for node in all_orbit_nodes:
            node.update_position(current_timestamp)
            
        # 2. 构建 ISL (暴力搜索最近邻居)
        nodes_list = list(self.satellites.values())
        for i, sat1 in enumerate(nodes_list):
            dists = []
            for j, sat2 in enumerate(nodes_list):
                if i == j: continue
                d = sat1.get_distance(sat2)
                if d < 5000: 
                    dists.append((d, sat2))
            
            dists.sort(key=lambda x: x[0])
            for d, target in dists[:4]:
                self.graph.add_edge(sat1.id, target.id, weight=d/SPEED_OF_LIGHT, type='ISL')
        
        # 3. 连接 SDC
        for sdc in self.sdcs.values():
            min_d = float('inf')
            nearest = None
            for sat in self.satellites.values():
                d = sdc.get_distance(sat)
                if d < min_d:
                    min_d = d
                    nearest = sat
            if nearest:
                self.graph.add_edge(sdc.id, nearest.id, weight=min_d/SPEED_OF_LIGHT, type='ISL')

        # 4. 连接 GS (GSL)
        for gs in self.ground_stations.values():
            min_d = float('inf')
            nearest = None
            for sat in self.satellites.values():
                d = gs.get_distance(sat)
                if d < 2000 and d < min_d:
                    min_d = d
                    nearest = sat
            
            if nearest:
                self.graph.add_edge(gs.id, nearest.id, weight=min_d/SPEED_OF_LIGHT, type='GSL')

    def ospc_routing(self, source_id, target_id):
        try:
            return nx.shortest_path(self.graph, source_id, target_id, weight='weight')
        except:
            return None
        
    def calculate_path_latency(self, user_position, path_node_ids, propagation_factor=1.0):
        """
        计算一条完整路径的累积传播延迟。
        
        Args:
            user_position (np.array): 用户的物理坐标 [x, y, z]，单位 km。
                                      如果路径是从地面站开始，这里可以传 None。
            path_node_ids (list): 路径上经过的节点 ID 列表。
                                  例如: ['GS_1', 'Sat_10', 'Sat_11', 'SDC_2']
            propagation_factor (float): 介质系数。
                                        1.0 = 真空光速 (无线电/激光)
                                        1.5 = 光纤 (地面站之间如果走光纤，速度约为 2/3 c)
        
        Returns:
            float: 总延迟 (秒)
        """
        total_distance = 0.0
        
        # 1. 处理 [用户] -> [路径第一个节点] 的这一跳
        # 如果提供了用户坐标，说明第一跳是 "User -> Access Node"
        if user_position is not None and len(path_node_ids) > 0:
            first_node_id = path_node_ids[0]
            first_node = self.get_node_obj(first_node_id)
            
            if first_node:
                # 手动计算距离: norm(UserPos - NodePos)
                dist = np.linalg.norm(user_position - first_node.position)
                total_distance += dist
            else:
                print(f"Warning: Start node {first_node_id} not found.")

        # 2. 处理 [节点] -> [节点] 的中间路径
        # 遍历列表，计算 i 和 i+1 之间的距离
        for i in range(len(path_node_ids) - 1):
            curr_id = path_node_ids[i]
            next_id = path_node_ids[i+1]
            
            node_a = self.get_node_obj(curr_id)
            node_b = self.get_node_obj(next_id)
            
            if node_a and node_b:
                # 调用 Node 类自带的 get_distance
                dist = node_a.get_distance(node_b)
                total_distance += dist
            else:
                print(f"Warning: Link {curr_id}->{next_id} contains invalid nodes.")

        # 3. 计算延迟
        # 公式: 时间 = 距离 / (光速 / 介质系数)
        # SPEED_OF_LIGHT 是你代码开头的全局变量 (约 300,000 km/s)
        latency = total_distance / (SPEED_OF_LIGHT / propagation_factor)
        
        return latency