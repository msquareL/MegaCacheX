import os

def get_config():
    """
    获取全局仿真配置参数
    返回一个字典，包含路径、仿真参数、物理参数等
    """
    
    # 基础目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRACE_DIR = os.path.join(BASE_DIR, 'trace')
    CONFIG_DIR = os.path.join(BASE_DIR, 'configuration')
    RESULT_DIR = os.path.join(BASE_DIR, 'results')
    
    conf = {
        # 文件路径配置
        'paths': {
            'tle_sat':      os.path.join(CONFIG_DIR, "S_constellation.tle"),
            'tle_sdc':      os.path.join(CONFIG_DIR, "SDC_constellation.tle"),
            'gs_nodes':     os.path.join(CONFIG_DIR, "GS_Starlink_Nodes.csv"),
            'user_nodes':   os.path.join(CONFIG_DIR, "User_Nodes.csv"),
            'trace_file':   os.path.join(TRACE_DIR,   "StarFront_CDN_Trace_Dataset.csv"),
            
            # 结果输出图片
            'plot_save_cdf':   os.path.join(RESULT_DIR, 'Result_Latency_CDF.png'),
            'plot_save_trend': os.path.join(RESULT_DIR, 'Result_Latency_Trend.png'),
            'plot_save_topo':  os.path.join(RESULT_DIR, 'Result_3D_Topology.png'),
        },

        # 仿真控制参数
        'simulation': {
            'duration': 6000,   # 仿真总时长 (秒)
            'step': 60,         # 时间步长 (秒)
        },

        # 物理与网络参数
        'physics': {
            'bandwidth_isl': 1250.0,  # 星间链路带宽 (MB/s)
            'bandwidth_gsl': 62.5,    # 星地链路带宽 (MB/s)
            'bandwidth_user': 12.5,   # 用户链路带宽 (MB/s)
            'sat_capacity': 500.0,    # 单颗卫星存储容量 (MB)
        },

        # 可视化字体配置
        'font': {
            'path_win': r'C:\Windows\Fonts\simhei.ttf',
            'path_mac': '/System/Library/Fonts/STHeiti Light.ttc' # 黑体
        }
    }
    
    return conf