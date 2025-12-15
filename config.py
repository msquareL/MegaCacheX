import os

def get_config():
    """
    获取全局仿真配置参数
    返回一个字典，包含路径、仿真参数、物理参数等
    """
    
    # 基础目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, 'configuration')
    
    conf = {
        # 文件路径配置 (Paths)
        'paths': {
            'tle_sat':      os.path.join(CONFIG_DIR, "S_constellation.tle"),
            'tle_sdc':      os.path.join(CONFIG_DIR, "SDC_constellation.tle"),
            'gs_nodes':     os.path.join(CONFIG_DIR, "GS_Starlink_Nodes.csv"),
            'user_nodes':   os.path.join(CONFIG_DIR, "User_Nodes.csv"),
            'trace_file':   os.path.join(BASE_DIR,   "StarFront_CDN_Trace_Dataset.csv"),
            
            # 结果输出图片命名
            'plot_save_cdf':   'Result_Latency_CDF.png',
            'plot_save_trend': 'Result_Latency_Trend.png',
            'plot_save_topo':  'Result_3D_Topology.png',
        },

        # 仿真控制参数 (Simulation Control)
        'simulation': {
            'duration': 6000,   # 仿真总时长 (秒)
            'step': 60,         # 时间步长 (秒)
            'start_time_str': "2024-01-01 00:00:00", # 仿真开始时间字符串
        },

        # 物理与网络参数 (Network Physics)
        'physics': {
            'bandwidth_isl': 1250.0,  # 星间链路带宽 (MB/s)
            'bandwidth_gsl': 62.5,    # 星地链路带宽 (MB/s)
            'bandwidth_user': 12.5,   # 用户链路带宽 (MB/s)
            'sat_capacity': 500.0,    # 单颗卫星存储容量 (MB)
        },

        # 算法参数 (Algorithm)
        'algo': {
            'mlc3_threshold': 0.01,   # 缓存打分阈值 (低于此分不存)
        },
        
        # 可视化字体配置 (Visualization)
        'font': {
            # 在这里填入您确定好的字体路径，Visualization.py 可以读取这个配置
            'path_win': r'C:\Windows\Fonts\simhei.ttf',
            'path_mac': '/System/Library/Fonts/STHeiti Light.ttc'
        }
    }
    
    return conf