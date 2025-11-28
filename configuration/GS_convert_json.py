import json
import pandas as pd
import os

def convert_custom_json_to_csv(json_file_path, output_csv_path):
    print(f"正在读取文件: {json_file_path} ...")
    
    if not os.path.exists(json_file_path):
        print(f"错误: 找不到文件 '{json_file_path}'。")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 准备列表存储提取的数据
        gs_list = []
        
        # 遍历你的 JSON 列表
        for item in data:
            # 1. 提取基础信息
            # 使用 .get() 防止某些字段缺失报错
            lat = item.get("lat")
            lng = item.get("lng")
            town = item.get("town", "Unknown Location")
            country = item.get("country", "Unknown")
            min_el = item.get("minElevation", 25) # 如果没有，默认为25度
            
            # 2. 数据清洗与重命名
            # 确保经纬度存在且不为 None
            if lat is not None and lng is not None:
                gw_node = {
                    "name": f"{town}, {country}",  # 组合名字，例如 "Rolette, ND, US"
                    "latitude": float(lat),
                    "longitude": float(lng),
                    #"min_elevation": float(min_el), # 保留仰角数据，仿真可能用到
                }
                gs_list.append(gw_node)

        # 转换为 DataFrame
        df = pd.DataFrame(gs_list)
        
        # ---------------------------------------------------------------
        # 3. (可选) 为了完全复现论文的 ~357 个节点，我们在这里补上 AWS/Azure 节点
        # 如果你只想要 Starlink，可以注释掉下面这一段
        # ---------------------------------------------------------------
        # cloud_gs_data = [
        #     {"name": "AWS Oregon", "latitude": 45.8, "longitude": -119.7, "min_elevation": 25, "provider": "AWS", "type": "Ground Station"},
        #     {"name": "AWS Ohio", "latitude": 40.1, "longitude": -83.0, "min_elevation": 25, "provider": "AWS", "type": "Ground Station"},
        #     {"name": "AWS Ireland", "latitude": 51.9, "longitude": -8.5, "min_elevation": 25, "provider": "AWS", "type": "Ground Station"},
        #     {"name": "AWS Cape Town", "latitude": -33.9, "longitude": 18.4, "min_elevation": 25, "provider": "AWS", "type": "Ground Station"},
        #     {"name": "Azure Quincy", "latitude": 47.2, "longitude": -119.8, "min_elevation": 25, "provider": "Azure", "type": "Ground Station"},
        #     {"name": "Azure Sweden", "latitude": 60.6, "longitude": 17.1, "min_elevation": 25, "provider": "Azure", "type": "Ground Station"},
        # ]
        # df_cloud = pd.DataFrame(cloud_gs_data)
        
        # 合并 Starlink 和 Cloud 节点
        df_final = pd.concat([df], ignore_index=True)
        # ---------------------------------------------------------------

        # 保存为 CSV
        df_final.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        print("-" * 40)
        print(f"转换成功！")
        print(f"输入节点数: {len(data)}")
        print(f"Starlink 有效节点: {len(df)}")
        print(f"总输出节点 (含云厂商): {len(df_final)}")
        print(f"文件已保存为: {output_csv_path}")
        print("-" * 40)
        print("CSV 数据预览:")
        print(df_final.head())

    except json.JSONDecodeError:
        print("错误: JSON 格式解析失败，请检查文件内容是否完整。")
    except Exception as e:
        print(f"发生错误: {e}")

# 执行转换
if __name__ == "__main__":
    input_json = "gateways.json"          # 你的 json 文件名
    output_csv = "MegaCacheX_GS_Nodes.csv" # 输出的 csv 文件名
    
    convert_custom_json_to_csv(input_json, output_csv)