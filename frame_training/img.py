import gymnasium as gym
import numpy as np
import cv2
import os
import math

# ================= 配置 =================
SAVE_DIR = "datasets/mountaincar_tuned"
IMAGES_DIR = os.path.join(SAVE_DIR, "images/train")
LABELS_DIR = os.path.join(SAVE_DIR, "labels/train")
DEBUG_DIR = os.path.join(SAVE_DIR, "debug_visuals") 

NUM_SAMPLES = 2000
DEBUG_COUNT = 50

# 物理与屏幕参数
MIN_POSITION = -1.2
MAX_POSITION = 0.6
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
WORLD_WIDTH = MAX_POSITION - MIN_POSITION
SCALE = SCREEN_WIDTH / WORLD_WIDTH 

# ==========================================
# [核心微调区域] TUNING KNOBS
# 根据您的反馈，我调整了以下参数：
# ==========================================

# 1. 垂直基准修正 (Vertical Pivot Offset)
# 这是"轨道线"到"车身几何中心"的像素距离。
# 之前是 15 (偏高)，现在改为 10 (让红点下沉 5px)
PIVOT_OFFSET_Y = 10 

# 2. 几何宽高 (从中心点到端点的距离)
# 车宽的一半: 18 (保持不变，看起来宽度是对的)
CAR_HALF_WIDTH = 18 

# 3. 车高的一半: 之前是 12。
# 如果红点下沉了，端点也会跟着下沉。为了确保点在车顶角上，保持 11-12 左右即可。
CAR_HALF_HEIGHT = 20

# 4. 全局手动 Bias (以防万一还需要硬推)
# 如果发现整体还偏左/偏右，改这里
GLOBAL_BIAS_X = 0 
GLOBAL_BIAS_Y = 0

# ==========================================

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def get_car_geometry_tuned(position):
    # --- 1. 计算中心点 (cx, cy) ---
    cx = (position - MIN_POSITION) * SCALE
    
    # 物理高度
    car_y_phys = 0.45 * math.sin(3 * position) + 0.55
    
    # 屏幕高度计算
    # PIVOT_OFFSET_Y 控制红点的高度
    cy = SCREEN_HEIGHT - (car_y_phys * SCALE) - PIVOT_OFFSET_Y
    
    # 应用全局 Bias
    cx += GLOBAL_BIAS_X
    cy += GLOBAL_BIAS_Y
    
    # --- 2. 向量法计算端点 (保持几何刚性) ---
    k = 1.35 * math.cos(3 * position)
    
    # 切线向量
    tx, ty = 1.0, -k
    length = math.sqrt(tx**2 + ty**2)
    tx /= length; ty /= length
    
    # 法线向量
    nx, ny = ty, -tx
    
    # 左上角
    lx = cx - tx * CAR_HALF_WIDTH + nx * CAR_HALF_HEIGHT
    ly = cy - ty * CAR_HALF_WIDTH + ny * CAR_HALF_HEIGHT
    
    # 右上角
    rx = cx + tx * CAR_HALF_WIDTH + nx * CAR_HALF_HEIGHT
    ry = cy + ty * CAR_HALF_WIDTH + ny * CAR_HALF_HEIGHT
    
    return (cx, cy), (lx, ly), (rx, ry)

def main():
    # 使用 unwrapped 强制设置状态
    env = gym.make("MountainCar-v0", render_mode="rgb_array").unwrapped
    env.reset()
    
    print(f"生成微调版数据... (PIVOT_Y={PIVOT_OFFSET_Y}, H={CAR_HALF_HEIGHT})")
    print(f"请检查: {DEBUG_DIR}")
    
    # 均匀采样
    positions = np.linspace(MIN_POSITION, MAX_POSITION, NUM_SAMPLES)
    np.random.shuffle(positions)
    
    for i, pos in enumerate(positions):
        vel = np.random.uniform(-0.07, 0.07)
        env.state = np.array([pos, vel])
        
        img_rgb = env.render()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H, W = img_bgr.shape[:2]
        
        # 计算坐标
        (cx, cy), (lx, ly), (rx, ry) = get_car_geometry_tuned(pos)
        
        # 归一化
        ncx, ncy = np.clip(cx/W, 0, 1), np.clip(cy/H, 0, 1)
        nlx, nly = np.clip(lx/W, 0, 1), np.clip(ly/H, 0, 1)
        nrx, nry = np.clip(rx/W, 0, 1), np.clip(ry/H, 0, 1)
        
        # 框的大小
        nw, nh = 40/W, 50/H 
        
        # 标签格式: 0 cx cy w h k1x k1y 2 k2x k2y 2
        label_str = (f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f} "
                     f"{nlx:.6f} {nly:.6f} 2.000000 "
                     f"{nrx:.6f} {nry:.6f} 2.000000")
        
        cv2.imwrite(os.path.join(IMAGES_DIR, f"mc_{i:05d}.jpg"), img_bgr)
        with open(os.path.join(LABELS_DIR, f"mc_{i:05d}.txt"), "w") as f:
            f.write(label_str)
            
        # 验证图
        if i < DEBUG_COUNT:
            debug_img = img_bgr.copy()
            # 黄线: 车顶
            cv2.line(debug_img, (int(lx), int(ly)), (int(rx), int(ry)), (0, 255, 255), 1)
            # 端点
            cv2.circle(debug_img, (int(lx), int(ly)), 2, (0, 255, 0), -1)
            cv2.circle(debug_img, (int(rx), int(ry)), 2, (255, 0, 0), -1)
            # 中心
            cv2.circle(debug_img, (int(cx), int(cy)), 2, (0, 0, 255), -1)
            
            # 辅助十字准星 (检查红点是否在车身几何中心)
            # cv2.line(debug_img, (int(cx)-10, int(cy)), (int(cx)+10, int(cy)), (0,0,255), 1)
            # cv2.line(debug_img, (int(cx), int(cy)-10), (int(cx), int(cy)+10), (0,0,255), 1)
            
            cv2.putText(debug_img, f"Pos: {pos:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
            cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_{i:03d}.jpg"), debug_img)

    env.close()
    print("生成完毕。")

if __name__ == "__main__":
    main()