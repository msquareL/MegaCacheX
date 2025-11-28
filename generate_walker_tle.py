import math
import os

def calculate_checksum(line):
    """计算 TLE 行的校验和"""
    checksum = 0
    for char in line[:-1]: # 不包含最后的校验位位置
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10

def generate_walker_tle(filename, sat_name_prefix, 
                        altitude_km, inclination_deg, 
                        num_planes, sats_per_plane, 
                        phase_offset=0):
    """
    生成 Walker 星座 TLE 文件
    
    参数:
    - altitude_km: 轨道高度 (km)
    - inclination_deg: 轨道倾角 (度)
    - num_planes: 轨道面数量 (P)
    - sats_per_plane: 每个轨道面的卫星数量 (S)
    - phase_offset: 相位因子 (通常为 0 或 1)
    """
    
    print(f"正在生成 {filename} ...")
    print(f"  - 配置: 高度={altitude_km}km, 倾角={inclination_deg}°, P={num_planes}, S={sats_per_plane}")
    
    # 1. 物理常数与轨道计算
    mu = 398600.4418  # 地球引力常数 (km^3/s^2)
    earth_radius = 6378.137 # 地球赤道半径 (km)
    
    # 半长轴 (Semi-major axis)
    a = earth_radius + altitude_km
    
    # 轨道周期 (秒) T = 2*pi * sqrt(a^3 / mu)
    period_seconds = 2 * math.pi * math.sqrt(math.pow(a, 3) / mu)
    
    # 平均运动 (Mean Motion) revs/day
    mean_motion = 86400.0 / period_seconds
    
    tles = []
    sat_id = 1
    
    # 2. 循环生成卫星
    for p in range(num_planes):
        # 升交点赤经 (RAAN): 将轨道面均匀分布在 0-360 度
        raan = (360.0 / num_planes) * p
        
        for s in range(sats_per_plane):
            # 平近点角 (Mean Anomaly): 卫星在轨道面内的位置
            # Walker Delta 相位偏移: (p * phase_offset * 360 / (P * S))
            # 这里简化处理，通常相位偏移是为了避免碰撞和优化覆盖
            ma_offset = (p * phase_offset * 360.0) / (num_planes * sats_per_plane)
            mean_anomaly = (s * (360.0 / sats_per_plane) + ma_offset) % 360.0
            
            # 3. 构建 TLE 字符串
            # 名字行
            name_line = f"{sat_name_prefix}_P{p+1}_S{s+1}"
            
            # Line 1 (简化版，只要 ID 和列对齐正确，Skyfield 就能读)
            # 格式: 1 NNNNNU YYMMMAAA YYDDD.DDDDDDDD +.00000000 +00000-0 +00000-0 0  999C
            # 这里的 Epoch Day 用 20001.00000000 (2020年第1天) 假定
            line1_template = "1 {0:05d}U 20001A   25001.00000000  .00000000  00000-0  00000-0 0  999"
            line1_raw = line1_template.format(sat_id)
            line1 = f"{line1_raw}{calculate_checksum(line1_raw)}"
            
            # Line 2
            # 格式: 2 NNNNN III.IIII RRR.RRRR EEEEEEE AAA.AAAA MMM.MMMM NN.NNNNNNNNRRRRC
            # Inclination, RAAN, Eccentricity(0), ArgPerigee(0), MeanAnomaly, MeanMotion
            
            # 格式化各个字段
            inc_str = f"{inclination_deg:8.4f}"
            raan_str = f"{raan:8.4f}"
            ecc_str = "0001000" # 接近圆轨道
            argp_str = "000.0000" # 近地点幅角
            ma_str = f"{mean_anomaly:8.4f}"
            mm_str = f"{mean_motion:11.8f}"
            
            line2_raw = f"2 {sat_id:05d} {inc_str} {raan_str} {ecc_str} {argp_str} {ma_str} {mm_str}00001"
            line2 = f"{line2_raw}{calculate_checksum(line2_raw)}"
            
            tles.append(f"{name_line}\n{line1}\n{line2}\n")
            sat_id += 1

    # 4. 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(tles)
    
    print(f"完成！生成了 {len(tles)} 颗卫星 -> {filename}\n")

# =========================================================================
# 配置区域：请根据 MegaCacheX 论文修改这里的数值
# =========================================================================

if __name__ == "__main__":
    
    # --- 1. 生成 Tier-2 (Relay Satellites, e.g., Starlink Shell 1) ---
    # 论文中通常使用 Starlink 第一阶段参数
    # 参数参考: 550km, 53度倾角, 72个轨道面, 每个面22颗星 (总共1584颗)
    generate_walker_tle(
        filename="S_constellation.tle",
        sat_name_prefix="RELAY",
        altitude_km=540,       # 高度
        inclination_deg=53.2,  # 倾角
        num_planes=72,         # 轨道面数
        sats_per_plane=22,     # 每面卫星数
        phase_offset=1         # 相位因子 (Starlink通常错开)
    )

    # --- 2. 生成 Tier-1 (SDC Satellites) ---
    # 论文中 SDC 通常少很多，或者是 Starlink 的一个子集
    # 假设参数: 比如一个较稀疏的覆盖层 (例如 OneWeb 风格或自定义)
    # 如果论文没具体说，可以假设它是一个较小的 Walker 星座
    # 下面是一个示例配置 (10个面，每面5个 = 50颗 SDC)
    generate_walker_tle(
        filename="SDC_constellation.tle",
        sat_name_prefix="SDC",
        altitude_km=700,       # SDC 可能稍微高一点或一样
        inclination_deg=97.5,  # 极轨或高倾角覆盖全球
        num_planes=1,          # 轨道面
        sats_per_plane=8,     # 每面卫星数
        phase_offset=0
    )