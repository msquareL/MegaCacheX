import requests
import pandas as pd

def fetch_and_convert_globalping():
    # Globalping 官方 API 地址
    url = "https://api.globalping.io/v1/probes"
    
    print(f"正在从 {url} 获取实时探针数据...")
    
    try:
        # 发送请求
        response = requests.get(url, timeout=10)
        response.raise_for_status() # 检查请求是否成功
        data = response.json()
        
        print(f"API 请求成功，获取到 {len(data)} 个原始探针数据。")
        
        gdc_list = []
        
        # 遍历数据并提取字段
        for item in data:
            # 提取 location 字典
            loc = item.get("location", {})
            
            # 提取关键坐标
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            
            # 仅处理包含有效坐标的在线探针
            if lat is not None and lon is not None:
                
                # 构建名字: City, Country
                city = loc.get("city", "Unknown")
                country = loc.get("country", "XX")
                name_str = f"{city}, {country}"
                
                node = {
                    "name": name_str,
                    "latitude": float(lat),
                    "longitude": float(lon)
                }
                gdc_list.append(node)

        # 转换为 DataFrame
        df = pd.DataFrame(gdc_list)
        
        # 保存文件名
        output_file = "GDC_Globalping_Nodes.csv"
        
        # 保存 CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print("-" * 40)
        print("处理完成！")
        print(f"有效 GDC 节点数: {len(df)}")
        print(f"文件已保存为: {output_file}")
        print("-" * 40)
        print("数据预览:")
        print(df.head())
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    fetch_and_convert_globalping()