import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from config import get_config

class ResultVisualizer:
    def __init__(self):
        """初始化可视化器"""
        cfg = get_config()

        # 设置基础绘图风格
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')

        # 根据操作系统选择字体路径
        if sys.platform.startswith('win'):
            font_path = cfg['font']['path_win']
        else:
            font_path = cfg['font']['path_mac']

        # 加载字体
        if os.path.exists(font_path):
            self.my_font = fm.FontProperties(fname=font_path)
            print(f"绘图字体： {font_path}")
        else:
            print(f"error: 找不到路径 {font_path}")
            self.my_font = fm.FontProperties() # 回退

    def plot_latency_cdf(self, all_latencies, save_name):
        """
        端到端延迟的 CDF (累积分布)
        """
        if not all_latencies:
            print(f"数据为空，跳过 {save_name}")
            return

        plt.figure(figsize=(8, 6))
        
        lat_data_ms = np.array(all_latencies) * 1000 # 转换为毫秒
        
        sorted_data = np.sort(lat_data_ms) # 排序
        
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1) # 计算 y 轴 (0 ~ 1)
        
        plt.plot(sorted_data, yvals, linewidth=2, color='darkblue')

        plt.xlim(0, 410) # 设置 x 轴范围
        plt.ylim(0, 1.05) # 设置 y 轴范围

        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.6, label='50% 概率线 (P50)')

        for x_val in [100, 200, 300, 400]:
            plt.axvline(x=x_val, color='gray', linestyle=':', alpha=0.5)

        plt.xlabel('端到端延迟 (毫秒)', fontproperties=self.my_font, fontsize=12)
        plt.ylabel('累积概率 (CDF)', fontproperties=self.my_font, fontsize=12)
        plt.title('用户请求延迟分布图', fontproperties=self.my_font, fontsize=14)
        
        plt.legend(prop=self.my_font, fontsize=10, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.savefig(save_name, dpi=300)
        print(f"  - 已保存: {save_name}")
        plt.close()

    def plot_3d_topology(self, mc, save_name='Result_3D_Topology.png'):
        """
        图3: 画 3D 星座拓扑快照
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 画卫星
        xs, ys, zs = [], [], []
        for sat in mc.satellites.values():
            xs.append(sat.position[0])
            ys.append(sat.position[1])
            zs.append(sat.position[2])
        ax.scatter(xs, ys, zs, s=1, c='#0077BE', alpha=0.6, label='低轨卫星')
        
        # 2. 画源站 (SDC)
        if mc.sdcs:
            s_xs, s_ys, s_zs = [], [], []
            for sdc in mc.sdcs.values():
                s_xs.append(sdc.position[0])
                s_ys.append(sdc.position[1])
                s_zs.append(sdc.position[2])
            ax.scatter(s_xs, s_ys, s_zs, s=50, c='red', marker='*', label='空间源站')

        # 3. 画地球线框
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        earth_x = 6371 * np.cos(u) * np.sin(v)
        earth_y = 6371 * np.sin(u) * np.sin(v)
        earth_z = 6371 * np.cos(v)
        ax.plot_wireframe(earth_x, earth_y, earth_z, color='gray', alpha=0.2, linewidth=0.5)

        # 4. 设置标签
        ax.set_title('大规模星座三维拓扑', fontproperties=self.my_font, fontsize=15)
        ax.set_xlabel('X (千米)', fontproperties=self.my_font)
        ax.set_ylabel('Y (千米)', fontproperties=self.my_font)
        ax.set_zlabel('Z (千米)', fontproperties=self.my_font)
        
        ax.view_init(elev=20, azim=45)
        plt.legend(loc='upper right', prop=self.my_font, fontsize=10)
        plt.tight_layout()
        
        plt.savefig(save_name, dpi=300)
        print(f"  - 已保存: {save_name}")
        plt.close()

    def plot_all(self, all_latencies, timestamps, avg_latencies, mc):
        """
        一键生成所有图表
        """
        print("\n[Visualization] 正在生成汇报图表...")
        self.plot_latency_cdf(all_latencies)
        self.plot_3d_topology(mc)
        print("[Visualization] 所有图表生成完毕。")