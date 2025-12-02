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
    """
    
    print(f"正在生成 {filename} ...")
    print(f"  - 配置: 高度={altitude_km}km, 倾角={inclination_deg}°, P={num_planes}, S={sats_per_plane}")
    
    # 物理常数与轨道计算
    mu = 398600.4418  # 地球引力常数 (km^3/s^2)
    earth_radius = 6378.137 # 地球赤道半径 (km)
    
    # 半长轴
    a = earth_radius + altitude_km
    
    # 轨道周期 (秒) T = 2*pi * sqrt(a^3 / mu)
    period_seconds = 2 * math.pi * math.sqrt(math.pow(a, 3) / mu)
    
    # 平均运动 (Mean Motion) revs/day
    mean_motion = 86400.0 / period_seconds
    
    tles = []
    sat_id = 1
    
    # 循环生成卫星
    for p in range(num_planes):
        # 升交点赤经 (RAAN): 将轨道面均匀分布在 0-360 度
        raan = (360.0 / num_planes) * p
        
        for s in range(sats_per_plane):
            # 平近点角 (Mean Anomaly): 卫星在轨道面内的位置
            # Walker Delta 相位偏移: (p * phase_offset * 360 / (P * S))
            ma_offset = (p * phase_offset * 360.0) / (num_planes * sats_per_plane)
            mean_anomaly = (s * (360.0 / sats_per_plane) + ma_offset) % 360.0
            
            # 构建 TLE 字符串
            # 名字行
            name_line = f"{sat_name_prefix}_P{p+1}_S{s+1}"
            
            # Line 1
            # 格式: 1 NNNNNU YYMMMAAA YYDDD.DDDDDDDD +.00000000 +00000-0 +00000-0 0  999C
            line1_template = "1 {0:05d}U 20001A   15052.66671296  .00000000  00000-0  00000-0 0  999"
            line1_raw = line1_template.format(sat_id)
            line1 = f"{line1_raw}{calculate_checksum(line1_raw)}"
            
            # Line 2
            # 格式: 2 NNNNN III.IIII RRR.RRRR EEEEEEE AAA.AAAA MMM.MMMM NN.NNNNNNNNRRRRC
       
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

if __name__ == "__main__":
    
    # 生成 S
    generate_walker_tle(
        filename="S_constellation.tle",
        sat_name_prefix="RELAY",
        altitude_km=540,       # 高度
        inclination_deg=53.2,  # 倾角
        num_planes=72,         # 轨道面数
        sats_per_plane=22,     # 每面卫星数
        phase_offset=1         # 相位因子 
    )

    # 生成 SDC
    generate_walker_tle(
        filename="SDC_constellation.tle",
        sat_name_prefix="SDC",
        altitude_km=700,       # 高度
        inclination_deg=97.5,  # 倾角
        num_planes=1,          # 轨道面数
        sats_per_plane=8,      # 每面卫星数
        phase_offset=0         # 相位因子
    )