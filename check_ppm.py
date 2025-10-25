import sys
import re

def read_ppm(ppm_file_path):
    """
    读取 PPM 文件 (P3 或 P6 格式) 并打印其头部信息和像素数据。
    """
    try:
        with open(ppm_file_path, 'rb') as f:
            # --- 1. 读取并解析头部 ---

            # 读取魔数 (P3 或 P6)
            magic_number = f.readline().strip().decode('ascii')
            if magic_number not in ['P3', 'P6']:
                print(f"错误: 不是一个有效的 PPM 文件。文件魔数为: {magic_number}")
                return

            print(f"PPM 文件: {ppm_file_path}")
            print(f"格式: {magic_number} ({'ASCII' if magic_number == 'P3' else 'Binary'})")

            # 读取尺寸和最大颜色值，同时跳过注释行
            header_data = []
            while len(header_data) < 3:
                line = f.readline().strip()
                if line and not line.startswith(b'#'):
                    header_data.extend([int(val) for val in line.split()])
            
            width, height, max_val = header_data[0], header_data[1], header_data[2]
            
            print(f"尺寸: {width} x {height}")
            print(f"最大颜色值: {max_val}")

            # --- 2. 读取像素数据 ---
            pixels = []
            print("\n像素数据 (前 5 个像素):")

            if magic_number == 'P3': # ASCII 格式
                # 读取剩余的所有文本数据并分割
                data_str = f.read().decode('ascii')
                # 使用正则表达式查找所有数字
                pixel_values = [int(v) for v in re.findall(r'\d+', data_str)]
                
                # 将数字分组为 RGB 元组
                for i in range(0, len(pixel_values), 3):
                    if i + 2 < len(pixel_values):
                        pixels.append((pixel_values[i], pixel_values[i+1], pixel_values[i+2]))

            elif magic_number == 'P6': # 二进制格式
                # 每个像素3个字节 (R, G, B)
                expected_bytes = width * height * 3
                raw_data = f.read(expected_bytes)
                
                # 将字节流分组为 RGB 元组
                for i in range(0, len(raw_data), 3):
                    if i + 2 < len(raw_data):
                        pixels.append((raw_data[i], raw_data[i+1], raw_data[i+2]))
            
            # 打印前几个像素
            for i, p in enumerate(pixels[:5]):
                print(f"  Pixel {i}: R={p[0]}, G={p[1]}, B={p[2]}")
            
            if len(pixels) > 5:
                print(f"  ... (共 {len(pixels)} 个像素)")

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{ppm_file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else r"img_ComputerVision_1_0_1665494035500661900.ppm"
    read_ppm(file_path)

