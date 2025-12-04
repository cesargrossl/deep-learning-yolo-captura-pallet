sudo apt update
sudo apt install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0


python3 -m venv ~/venv-pallet
source ~/venv-pallet/bin/activate



pip install --upgrade pip
pip install opencv-python-headless ultralytics numpy



pip install pyrealsense2




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import math

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================

MODEL_PATH = "pallet_yolo.pt"   # seu modelo YOLO treinado
CLASS_NAME = "pallet_hole"      # nome da classe no modelo

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CENTER_TOLERANCE_PX = 25        # tolerância lateral em pixels para considerar alinhado
FAR_DISTANCE = 0.8              # acima disso anda rápido (m)
ENGAGE_DISTANCE = 0.28          # ponto de início do engate (m)
FORWARD_ENGAGE_EXTRA = 0.10     # avanço adicional para "entrar" no pallet (m estimado)

LINEAR_FAST = 0.25              # m/s aproximado
LINEAR_SLOW = 0.12              # m/s aproximado
ANGULAR_GAIN = 0.8              # ganho para correção lateral (rad/s por metro de X)

REVERSE_TIME = 3.0              # segundos de ré após pegar o pallet
LOOP_HZ = 10.0                  # frequência do loop (10 Hz)

# ==========================
# ABSTRAÇÃO DE CONTROLE DO ROBÔ
# ==========================
# Aqui você adapta para:
#  - GPIO direto
#  - biblioteca própria
#  - ou ROS2 (cmd_vel)
# Por enquanto vou deixar só prints.

def set_velocity(linear_x: float, angular_z: float):
    """
    Substitua este corpo para enviar comandos reais aos motores.
    Ex:
      - via ROS2: publicar geometry_msgs/Twist
      - via GPIO: calcular PWM para cada roda
    """
    # TODO: integrar com sua camada de controle
    print(f"[CMD] linear_x={linear_x:.3f} m/s | angular_z={angular_z:.3f} rad/s")

def stop_robot():
    set_velocity(0.0, 0.0)

def fork_up():
    """
    Substituir pela função real que levanta o garfo.
    Pode ser controle de servo, motor DC com fim de curso, etc.
    """
    print("[GARFO] Subindo garfo...")
    # TODO: acionar garfo
    time.sleep(2.0)

def fork_down():
    """
    Se quiser usar em outro fluxo (não usado aqui, mas deixado para referência).
    """
    print("[GARFO] Descendo garfo...")
    # TODO: acionar garfo
    time.sleep(2.0)

# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()

    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    return pipeline, align, (fx, fy, cx, cy)

def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())
    return depth_frame, color_frame, color_image

def get_pallet_detection(model, color_image):
    """
    Roda YOLO e retorna a melhor detecção da classe CLASS_NAME.
    Retorna (x1, y1, x2, y2, conf) ou None se não achar.
    """
    results = model(color_image[..., ::-1], verbose=False)[0]  # BGR->RGB para YOLO

    best = None
    best_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        if cls_name != CLASS_NAME:
            continue

        conf = float(box.conf[0].item())
        if conf > best_conf:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            best = (x1, y1, x2, y2, conf)
            best_conf = conf

    return best

def bbox_center(bbox):
    x1, y1, x2, y2, conf = bbox
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    return u, v

def pixel_to_3d(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return X, Y, Z

# ==========================
# MÁQUINA DE ESTADOS
# ==========================

class PalletPicker:
    def __init__(self, model, pipeline, align, intrinsics):
        self.model = model
        self.pipeline = pipeline
        self.align = align
        self.fx, self.fy, self.cx, self.cy = intrinsics

        self.state = "SEARCH"
        self.engage_start_time = None
        self.reverse_start_time = None

    def step(self):
        depth_frame, color_frame, color_image = get_aligned_frames(self.pipeline, self.align)
        if depth_frame is None:
            print("[WARN] Sem frames válidos da camera")
            stop_robot()
            return False  # continua

        detection = get_pallet_detection(self.model, color_image)

        if self.state == "SEARCH":
            self.handle_search(detection)

        elif self.state == "ALIGN":
            self.handle_align(detection)

        elif self.state == "APPROACH":
            self.handle_approach(detection, depth_frame)

        elif self.state == "ENGAGE":
            self.handle_engage()

        elif self.state == "LIFT":
            self.handle_lift()

        elif self.state == "REVERSE":
            self.handle_reverse()

        elif self.state == "DONE":
            stop_robot()
            print("[INFO] Ciclo concluído.")
            return True  # terminou

        return False  # segue rodando

    def handle_search(self, detection):
        print(f"[STATE] SEARCH")
        if detection is None:
            # Fica parado ou gira devagar para procurar
            set_velocity(0.0, 0.0)
            print("[SEARCH] Pallet não detectado.")
        else:
            print("[SEARCH] Pallet detectado, indo para ALIGN.")
            stop_robot()
            self.state = "ALIGN"

    def handle_align(self, detection):
        print(f"[STATE] ALIGN")
        if detection is None:
            print("[ALIGN] Perdi o pallet, voltando para SEARCH.")
            stop_robot()
            self.state = "SEARCH"
            return

        u, v = bbox_center(detection)
        center_x = IMAGE_WIDTH / 2
        error_px = u - center_x

        print(f"[ALIGN] u={u}, center_x={center_x:.1f}, error_px={error_px:.1f}")

        if abs(error_px) <= CENTER_TOLERANCE_PX:
            print("[ALIGN] Alinhado lateralmente, indo para APPROACH.")
            stop_robot()
            self.state = "APPROACH"
        else:
            # converte erro em pixels em giro angular. Simplesmente proporcional:
            # suposição: 1 pixel ~ k * rad (ajuste empírico).
            # Aqui: normaliza pelo meio da imagem para ter valor em [-1, 1]
            norm = error_px / center_x  # ~ [-1, 1]
            angular_z = -norm * 0.4  # ganho fixo para girar
            set_velocity(0.0, angular_z)

    def handle_approach(self, detection, depth_frame):
        print(f"[STATE] APPROACH")
        if detection is None:
            print("[APPROACH] Perdi o pallet, voltando para SEARCH.")
            stop_robot()
            self.state = "SEARCH"
            return

        u, v = bbox_center(detection)
        center_x = IMAGE_WIDTH / 2
        error_px = u - center_x
        norm = error_px / center_x

        Z = depth_frame.get_distance(u, v)

        if Z <= 0.0:
            print("[APPROACH] Medida de profundidade inválida, mantendo estado.")
            set_velocity(0.0, 0.0)
            return

        print(f"[APPROACH] u={u}, error_px={error_px:.1f}, Z={Z:.3f} m")

        # Pequena correção lateral mesmo aproximando
        angular_z = -norm * 0.3

        if Z > FAR_DISTANCE:
            # longe -> vai mais rápido
            linear_x = LINEAR_FAST
        elif Z > ENGAGE_DISTANCE:
            # perto -> vai mais devagar
            linear_x = LINEAR_SLOW
        else:
            print("[APPROACH] Distância de engate alcançada, indo para ENGAGE.")
            stop_robot()
            self.engage_start_time = time.time()
            self.state = "ENGAGE"
            return

        set_velocity(linear_x, angular_z)

    def handle_engage(self):
        print(f"[STATE] ENGAGE")
        if self.engage_start_time is None:
            self.engage_start_time = time.time()

        # Aqui vamos simplesmente andar para frente por um tempo fixo
        # aproximando FORWARD_ENGAGE_EXTRA à velocidade LINEAR_SLOW
        engage_duration = FORWARD_ENGAGE_EXTRA / LINEAR_SLOW  # t = d / v
        elapsed = time.time() - self.engage_start_time

        print(f"[ENGAGE] elapsed={elapsed:.2f}s / target={engage_duration:.2f}s")

        if elapsed < engage_duration:
            set_velocity(LINEAR_SLOW, 0.0)
        else:
            stop_robot()
            print("[ENGAGE] Engate concluído, indo para LIFT.")
            self.state = "LIFT"

    def handle_lift(self):
        print(f"[STATE] LIFT")
        stop_robot()
        fork_up()
        print("[LIFT] Garfo levantado, indo para REVERSE.")
        self.reverse_start_time = time.time()
        self.state = "REVERSE"

    def handle_reverse(self):
        print(f"[STATE] REVERSE")
        if self.reverse_start_time is None:
            self.reverse_start_time = time.time()

        elapsed = time.time() - self.reverse_start_time
        print(f"[REVERSE] elapsed={elapsed:.2f}s / target={REVERSE_TIME:.2f}s")

        if elapsed < REVERSE_TIME:
            set_velocity(-LINEAR_SLOW, 0.0)
        else:
            stop_robot()
            print("[REVERSE] Ré concluída, indo para DONE.")
            self.state = "DONE"


# ==========================
# MAIN
# ==========================

def main():
    print("[INIT] Carregando modelo YOLO...")
    model = YOLO(MODEL_PATH)

    print("[INIT] Inicializando RealSense...")
    pipeline, align, intrinsics = init_realsense()

    picker = PalletPicker(model, pipeline, align, intrinsics)

    dt = 1.0 / LOOP_HZ

    try:
        while True:
            done = picker.step()
            if done:
                break
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[EXIT] Interrompido pelo usuário.")
    finally:
        stop_robot()
        pipeline.stop()
        print("[EXIT] Finalizado com segurança.")


if __name__ == "__main__":
    main()


rodar

source ~/venv-pallet/bin/activate
python3 pallet_pick.py
