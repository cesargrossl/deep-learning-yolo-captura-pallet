import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# 1) Carrega modelo YOLO treinado para 'pallet_hole'
model = YOLO("pallet_hole_yolo.pt")

# 2) Configura RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Alinhamento depth->color
align_to = rs.stream.color
align = rs.align(align_to)

# Intrínsecos
color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

try:
    while True:
        # 3) Captura frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 4) YOLO detecta pallet_hole
        results = model(img_rgb, verbose=False)[0]

        target_point_3d = None

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            if cls_name != "pallet_hole":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            # 5) Lê profundidade (Z) no centro da janela
            Z = depth_frame.get_distance(u, v)  # em metros
            if Z == 0:
                continue

            # converte para coordenadas 3D da câmera
            X = (u - cx) / fx * Z
            Y = (v - cy) / fy * Z

            target_point_3d = (X, Y, Z)

            # desenha no frame
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(color_image, (u, v), 5, (0,0,255), -1)
            cv2.putText(color_image, f"Z={Z:.2f}m", (u, v-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            break  # usa só a melhor detecção

        # 6) A partir de target_point_3d você gera comandos pro robô
        if target_point_3d is not None:
            X, Y, Z = target_point_3d
            # EXEMPLO: controle proporcional bem simples
            Z_target = 0.25  # 25cm antes do pallet
            k_lin = 0.5
            k_ang = 1.0

            erro_z = Z - Z_target
            erro_x = X  # lateral

            cmd_linear = k_lin * erro_z
            cmd_angular = k_ang * erro_x

            # aqui você manda cmd_linear e cmd_angular pro seu robô (ROS2 ou direto)

        cv2.imshow("YOLO + D415", color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
