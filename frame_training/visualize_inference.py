import gymnasium as gym
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    # 默认路径指向刚训练好的模型，请根据实际情况修改
    parser.add_argument("--model", type=str, default="./best.pt", help="Path to trained best.pt")
    args = parser.parse_args()

    # 1. 加载模型
    print(f"正在加载模型: {args.model} ...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"错误: 无法加载模型。请检查路径是否正确。\n{e}")
        return

    # 2. 启动环境
    # render_mode="rgb_array" 让我们能获取图像数据进行推理
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    observation, info = env.reset()

    print("开始可视化... 按 'q' 键退出。")

    while True:
        # 随机动作让车乱跑 (或者你可以写个简单的策略让它动起来)
        # 这里给一个稍微有点动量的策略，让车能跑高点
        position, velocity = observation
        action = 2 if velocity > 0 else 0  # 简单的动量策略
        
        observation, reward, terminated, truncated, info = env.step(action)

        # 3. 获取图像
        frame_rgb = env.render()
        # Gym 返回 RGB，OpenCV 需要 BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 4. YOLO 推理
        # verbose=False 不打印每帧的日志
        results = model.predict(frame_bgr, verbose=False, conf=0.5)
        
        annotated_frame = frame_bgr.copy()
        
        # 5. 解析与绘制
        if results and results[0].boxes and len(results[0].boxes) > 0:
            # --- 目标已找到 (FOUND) ---
            
            # A. 获取检测框 (xyxy)
            box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            conf = float(results[0].boxes.conf[0])
            x1, y1, x2, y2 = box
            
            # 画框 (绿色)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 标示置信度
            cv2.putText(annotated_frame, f"Conf: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # B. 获取关键点
            if results[0].keypoints is not None:
                # shape: [2, 2] -> [[x1, y1], [x2, y2]]
                kpts = results[0].keypoints.xy[0].cpu().numpy().astype(int)
                
                # 确保检测到了 2 个点
                if len(kpts) >= 2:
                    pt1 = tuple(kpts[0]) # 左上
                    pt2 = tuple(kpts[1]) # 右上
                    
                    # 画连线 (黄色 - 代表车顶)
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 2)
                    
                    # 画端点 (绿色/蓝色)
                    cv2.circle(annotated_frame, pt1, 4, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, pt2, 4, (255, 0, 0), -1)
                    
                    # [核心验证] 计算推导出的几何中心 (红点)
                    # 这是 Server 端用来反推物理位置的核心依据
                    center_x = int((pt1[0] + pt2[0]) / 2)
                    center_y = int((pt1[1] + pt2[1]) / 2)
                    
                    # 画推导中心 (红色)
                    cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)
                    
                    # 辅助线：连接中心到底盘 (示意)
                    # cv2.line(annotated_frame, (center_x, center_y), (center_x, center_y+15), (0,0,255), 1)

        else:
            # --- 目标丢失 (LOST) ---
            h, w = annotated_frame.shape[:2]
            cv2.putText(annotated_frame, "LOST TARGET", (w//2 - 100, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 6. 显示
        cv2.imshow("YOLOv11 Pose Inference Check", annotated_frame)

        # 这里的 20ms 决定了播放速度，太快可以改大
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()