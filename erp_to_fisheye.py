import sys, os
import math
import cv2
import numpy as np

def equirectangular_to_fisheye(erp_img, fov_deg=180, out_size=1024):
    """
    将等距柱状 (ERP) 图像重投影为等距鱼眼图像（前向）。
    erp_img: HxW[xC] numpy array (BGR or gray).
    fov_deg: 鱼眼视场，单位度（通常 180）。
    out_size: 输出鱼眼宽/高（正方形）。
    返回： out_img (out_size x out_size x C)
    """
    h, w = erp_img.shape[:2]
    C = 1 if erp_img.ndim == 2 else erp_img.shape[2]

    # 目标像素坐标（归一化到 [-1,1]）
    xs = np.linspace(-1.0, 1.0, out_size)
    ys = np.linspace(-1.0, 1.0, out_size)
    XX, YY = np.meshgrid(xs, ys)           # XX: (row,col) x, YY: (row,col) y (y + downward)

    # 把图像坐标系的 y（下为正）转换为上为正
    YY_up = -YY

    R = np.sqrt(XX**2 + YY_up**2)
    mask = R <= 1.0                         # 鱼眼圆内有效

    fov_rad = math.radians(fov_deg)
    theta = R * (fov_rad / 2.0)             # 等距鱼眼：theta = r * (fov/2)
    phi = np.arctan2(YY_up, XX)             # 方位角

    # 在相机坐标系下的方向向量 (假设 +Z 为前)
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)

    # 把方向向量转换成经纬角用于ERP采样
    # longitude (λ) ∈ [-π, π], latitude (φ_lat) ∈ [-π/2, π/2]
    lon = np.arctan2(vx, vz)                # 注意顺序：atan2(x,z) 以 +Z 为参考前向
    lat = np.arcsin(vy)

    # 归一化到 ERP 像素坐标
    map_x = (lon + math.pi) / (2.0 * math.pi) * (w - 1)   # 横向 u
    map_y = (math.pi/2.0 - lat) / math.pi * (h - 1)       # 纵向 v (top = +pi/2)

    # 对 ERP 之外或鱼眼圆外的点设置任意值（remap 要求）
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 使用 OpenCV remap 采样
    if C == 1:
        out = cv2.remap(erp_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = out.copy()
        out[~mask] = 0
    else:
        out = cv2.remap(erp_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = out.copy()
        out[~mask] = 0

    return out

def main():
    if len(sys.argv) < 3:
        print("用法: python erp_to_fisheye.py <input_erp_image> <output_fisheye_image> [fov_deg] [out_size]")
        return

    input_path = sys.argv[1]
    out_path = sys.argv[2]
    fov = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
    out_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1024

    if not os.path.exists(input_path):
        print("输入文件不存在:", input_path)
        return

    # 读取 ERP 图像（保持通道）
    erp = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if erp is None:
        print("无法读取图片:", input_path)
        return

    fisheye = equirectangular_to_fisheye(erp, fov_deg=fov, out_size=out_size)
    # 保存
    cv2.imwrite(out_path, fisheye)
    print("已保存鱼眼图像:", out_path)

if __name__ == "__main__":
    main()