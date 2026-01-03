# Código integración del sistema mecatrónico

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import queue
import math
import sys
import os
import collections
import collections.abc
from pathlib import Path
import platform
from datetime import datetime
from collections import deque
import keyboard  # Librería original para detección global de teclas

# Parche para compatibilidad Python 3.10+ con DroneKit
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# Importación condicional de YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("[WARN] Ultralytics no instalado.")

# ==================== CONFIGURACIÓN DE RUTAS (USER) ====================
MODEL_1_PATH = r"D:\OneDrive - Instituto Politecnico Nacional\UPIITA\TT\Codigos\Interfaz\Dron\weights\Dron.pt"
MODEL_2_PATH = r"D:\OneDrive - Instituto Politecnico Nacional\UPIITA\TT\Codigos\Interfaz\Lampara\weights\Lampara.pt"
MODEL_3_PATH = r"D:\OneDrive - Instituto Politecnico Nacional\UPIITA\TT\Codigos\Interfaz\Persona\weights\Persona.pt"

MAP_FILENAME = r"D:\OneDrive - Instituto Politecnico Nacional\UPIITA\TT\Mapa\Mapa Salon A122\Mapa A122.png"
OUTPUT_FOLDER = r"D:\OneDrive - Instituto Politecnico Nacional\UPIITA\TT\Mapas Generados"

# ==================== CONFIGURACIÓN DE HARDWARE ====================
MAVLINK_PORT = 'COM5'
MAVLINK_BAUD = 57600
MAP_SCALE = 200.0  # 1 pixel = 1 cm
MAP_HOME_PIXEL_X = 160
MAP_HOME_PIXEL_Y = 570

# ==================== PARÁMETROS DE MISIÓN ====================
TARGET_ALTITUDE = 1.5 # metros
TARGET_DISTANCE = 2.0 # metros
TARGET_VELOCITY = 0.07 # m/s
ORIENT_FORWARD = 0
ORIENT_RIGHT   = 2
ORIENT_DOWN    = 25

# ==================== DESCRIPCIÓN DE PASOS (28 PASOS) ====================
STEP_DESCRIPTIONS = {
    0: "Sistema en Espera",
    1: "Armado de Motores",
    2: "Espera de Seguridad",
    3: "Despegue (Ascenso a 1.5m)",
    4: "Estabilización Post-Despegue",
    5: "Recalibración de ángulo de despegue",
    6: "Fin Estabilización Despegue",

    7: "Inicio Tramo 1",
    8: "Avance Tramo 1",
    9: "Fin Tramo 1 (Buscando Dron)",
    10: "Giro 1 (90° CW)",
    11: "Estabilización Giro 1",

    12: "Inicio Tramo 2",
    13: "Avance Tramo 2",
    14: "Fin Tramo 2 (Buscando Lámpara)",
    15: "Giro 2 (90° CW)",
    16: "Estabilización Giro 2",

    17: "Inicio Tramo 3",
    18: "Avance Tramo 3",
    19: "Fin Tramo 3",
    20: "Giro 3 (90° CW)",
    21: "Estabilización Giro 3",

    22: "Inicio Tramo 4",
    23: "Avance Tramo 4",
    24: "Fin Tramo 4",
    25: "Giro 4 (90° CW - Alineación)",
    26: "Estabilización Giro 4",

    27: "Aterrizando", 
    28: "Misión Completada"
}

# ==================== EXCEPCIONES ====================
class EmergencyLandingError(Exception):
    pass

class SmartRTLException(Exception):
    """Excepción para fallos automáticos (Sensores/Intrusos)"""
    pass

class ManualRTLException(Exception):
    """Excepción para retorno manual (Botón o Tecla R)"""
    pass

# ==================== CLASE DE FILTRADO (NUEVA) ====================
class LidarFilter:
    def __init__(self, history_size=5):
        self.history = deque(maxlen=history_size)
        self.last_valid = 0.0
    
    def update(self, raw_dist):
        # 1. Filtrar valores inválidos (0.0 o muy grandes)
        # Se asume error si es menor a 5cm o mayor a 12m
        if raw_dist <= 0.05 or raw_dist > 12.0:
            return self.last_valid # Ignorar y devolver el último bueno
        
        self.history.append(raw_dist)
        
        # 2. Filtro de Mediana (Elimina picos de ruido)
        sorted_vals = sorted(self.history)
        median_val = sorted_vals[len(sorted_vals)//2]
        
        self.last_valid = median_val
        return median_val

# ==================== CLASE PRINCIPAL ====================
class SuperInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # --- CONFIGURACIÓN VENTANA ---
        self.title("Sistema de Vigilancia y Control Autónomo")
        self.state('zoomed')
        self.configure(bg="#75B9BE")
        self.root_dir = Path(__file__).resolve().parent

        # Variables de configuración visual
        offset = 40
        self.battery_warn_threshold = 12.5 # ALERTA SI BAJA DE 12.5V
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Área de Video
        video_width = 640
        video_height = 480
        self.display_w = video_width
        self.display_h = video_height
        
        x = int(screen_width * 0.1)
        y = int((screen_height - video_height) / 2) + offset
        
        self.video_rect = tk.Label(self, bg="#222222", width=video_width, height=video_height)
        self.video_rect.place(x=x, y=y, width=video_width, height=video_height)

        # Estados del Sistema
        self.running = True
        self.inferring = False
        self.last_detections = []
        self.detection_enabled = True
        self.inference_interval = 0.25
        self._last_inference_time = 0.0
        self.connection_time = 0.0 # NUEVO: Para el delay de batería
        
        # Modelos YOLO
        self.models = {}
        self.active_model_id = 0
        self.model = None
        self.model_loaded = False
        
        # Carga de modelos
        print("\n" + "="*60)
        print("INICIALIZACIÓN DEL SISTEMA DE DETECCIÓN")
        print("="*60)
        self._load_all_models()
        print("="*60 + "\n")
        
        self.inference_queue = queue.Queue(maxsize=1)
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

        # ==================== CONFIGURACIÓN DE LA CÁMARA ====================
        print("[INFO] Iniciando sistema de cámaras...")
        self.preferred_camera_keywords = ["USB2.0", "USB", "PC CAMERA"]
        cam_index = self.auto_select_external_camera(max_index=2) 
        print(f"[INFO] Cámara seleccionada: índice {cam_index}\n")
        
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
        if not self.cap.isOpened():
            print(f"[WARN] No se pudo abrir cámara index {cam_index}, reintentando con 0")
            self.cap = cv2.VideoCapture(0)
            
        self.update_video()

        # --- DRONEKIT ---
        self.vehicle = None # NO SE CONECTA AUTOMÁTICAMENTE
        self.mavlink_connected = False
        self.telemetry_data = {
            "lidar_front": None, "lidar_right": None, "lidar_down": 0.0,
            "pos_x": 0.0, "pos_y": 0.0, "yaw": None,
            "opt_qua": 0, "flow_x": 0.0, "flow_y": 0.0,
            "h_spd": 0.0, "v_spd": 0.0, "battery": 0.0
        }
        
        # Variables de Misión
        self.trajectory_points = []
        self.arming_yaw_rad = 0.0
        self.initial_yaw_deg = 0.0 
        self.offset_x = 0.0; self.offset_y = 0.0
        self.pos_buffer_x = deque(maxlen=10)
        self.pos_buffer_y = deque(maxlen=10)
        self.target_heading = 0.0
        self.action_lock = False
        self.current_step = 0
        self.mission_metrics = {"dist_advanced": 0.0, "wall_dist": 0.0, "goal": 0.0}
        self.searching_for = "Nada"
        
        # --- NUEVO: Instancia del filtro ---
        self.lidar_filter = LidarFilter(history_size=5)

        # Cargar Imagen Mapa
        try:
            self.map_image_base = cv2.imread(MAP_FILENAME)
            if self.map_image_base is None:
                print("[ERROR CRITICO] No se encontro la imagen del mapa.")
                self.map_image_base = np.zeros((1000, 1000, 3), dtype=np.uint8)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar mapa: {e}")
            self.map_image_base = np.zeros((1000, 1000, 3), dtype=np.uint8)

        # --- ELEMENTOS UI ---
        tk.Label(self, text="Video en tiempo real", bg="#75B9BE", fg="white", font=("Arial", 24, "bold")).place(x=x + video_width // 2, y=y - 30, anchor="center")

        # Variables Tkinter
        self.bateria = tk.DoubleVar(value=0.0)
        self.altura = tk.DoubleVar(value=0.0)

        # --- ETIQUETA CENTRAL DE ANOMALÍAS ---
        self.center_label = tk.Label(self, text="Sin Anomalías", bg="#75B9BE", fg="green", font=("Arial", 32, "bold"))
        self.center_label.place(x=x + video_width // 2, y=y + video_height + 50, anchor="center")

        # Títulos Superiores
        tk.Label(self, bg="#A8CCC9", width=50, height=3, font=("Arial", 48, "bold")).place(relx=0.5, rely=0.05, anchor="center")
        tk.Label(self, text="Sistema de vigilancia", bg="#A8CCC9", fg="white", font=("Arial", 48, "bold")).place(relx=0.5, rely=0.1, anchor="center")

        try:
            u_img = Image.open("upiita.png").resize((140, 100))
            self.upiita_imgtk = ImageTk.PhotoImage(u_img)
            tk.Label(self, image=self.upiita_imgtk, bg="#A8CCC9").place(relx=0.1, rely=0.1, anchor="center")
            i_img = Image.open("ipn.png").resize((240, 120))
            self.ipn_imgtk = ImageTk.PhotoImage(i_img)
            tk.Label(self, image=self.ipn_imgtk, bg="#A8CCC9").place(relx=0.9, rely=0.1, anchor="center")
        except: pass

        # --- GESTIÓN DE COORDENADAS (SEPARADAS PARA MAPA Y TELEMETRÍA) ---
        img_width = 320
        img_height = 240
        img_x = int(screen_width * 0.9) - img_width
        
        # Coord Y del MAPA (Subido 280px del borde inferior para dejar espacio al texto)
        map_y = screen_height - img_height - 280 + offset 
        
        # Coord Y de TELEMETRÍA (Mantenido en 200px para que no suba y se quede en posición original)
        telem_y_base = screen_height - img_height - 200 + offset

        tk.Label(self, text="Mapa de Recorrido", bg="#75B9BE", fg="white", font=("Arial", 20, "bold")).place(x=img_x + img_width // 2, y=map_y - 20, anchor="center")

        self.image_rect = tk.Label(self, bg="#444444", width=img_width, height=img_height)
        self.image_rect.place(x=img_x, y=map_y, width=img_width, height=img_height)

        # Etiquetas Info (Relativas al MAPA para que estén debajo)
        self.step_label = tk.Label(self, text="Paso 0: Sistema en Espera", bg="#75B9BE", fg="black", font=("Arial", 14, "bold"), wraplength=400)
        self.step_label.place(x=img_x + img_width // 2, y=map_y + img_height + 30, anchor="center")

        self.search_status_label = tk.Label(self, text="Buscando: --", bg="#75B9BE", fg="#333333", font=("Arial", 12, "italic"))
        self.search_status_label.place(x=img_x + img_width // 2, y=map_y + img_height + 55, anchor="center")

        self.metrics_label = tk.Label(self, text="Avance: 0.00m | Pared: -- | Meta: --", bg="#75B9BE", fg="blue", font=("Arial", 13, "bold"))
        self.metrics_label.place(x=img_x + img_width // 2, y=map_y + img_height + 80, anchor="center")

        # --- BOTONES ---
        button_width = 180; button_height = 50
        video_right = x + video_width; map_left = img_x
        button_x = video_right + (map_left - video_right) // 2 - button_width // 2
        button_y = map_y + (img_height // 2) - button_height - 60 

        # 1. BOTÓN CONEXIÓN
        self.second_button = tk.Button(self, text="Desconectado", font=("Arial", 18), bg="red", fg="white", command=self.toggle_connection)
        self.second_button.place(x=button_x, y=button_y, width=button_width, height=button_height)

        # 2. BOTÓN INICIAR MISIÓN
        self.sequence_button = tk.Button(self, text="Iniciar Misión", font=("Arial", 16), bg="blue", fg="white", command=self.start_routine_thread)
        self.sequence_button.place(x=button_x, y=button_y + button_height + 10, width=button_width, height=button_height)

        # 3. BOTÓN REGRESO A LA BASE
        self.rtl_button = tk.Button(self, text="Regreso a la base", font=("Arial", 16), command=self.btn_action_rtl_safe)
        self.rtl_button.place(x=button_x, y=button_y + (button_height + 10)*2, width=button_width, height=button_height)

        # 4. BOTÓN ATERRIZAR
        self.land_button = tk.Button(self, text="Aterrizar", font=("Arial", 16), command=self.btn_action_land_safe)
        self.land_button.place(x=button_x, y=button_y + (button_height + 10)*3, width=button_width, height=button_height)

        # Label de estado de conexión
        self.return_status_label = tk.Label(self, text="", bg="#75B9BE", fg="orange", font=("Arial", 14, "bold"))
        self.return_status_label.place(x=button_x + button_width // 2, y=button_y + (button_height + 10)*3 + button_height + 25, anchor="center")

        # --- TELEMETRÍA (USANDO telem_y_base PARA MANTENER POSICIÓN ORIGINAL) ---
        self.telemetry_label = tk.Label(self, text="Telemetría", bg="#75B9BE", fg="black", font=("Arial", 28, "bold"))
        self.telemetry_label.place(x=img_x, y=telem_y_base - 220, width=img_width, height=40)
        
        self.battery_label = tk.Label(self, text="Batería: N/A", bg="#75B9BE", fg="black", font=("Arial", 18))
        self.battery_label.place(x=img_x, y=telem_y_base - 180, width=img_width, height=30)
        
        self.altitude_label = tk.Label(self, text="Altura: N/A", bg="#75B9BE", fg="black", font=("Arial", 18))
        self.altitude_label.place(x=img_x, y=telem_y_base - 150, width=img_width, height=30)

        # Indicador Modelo
        self.model_indicator = tk.Label(self, text="Modelo: Ninguno", bg="#FF6B6B", fg="white", font=("Arial", 14, "bold"), padx=10, pady=5)
        self.model_indicator.place(x=x + video_width - 10, y=y + 10, anchor="ne")

        # Binds de Modelos (Teclado Visual)
        self.bind('1', lambda e: self.switch_model(1))
        self.bind('2', lambda e: self.switch_model(2))
        self.bind('3', lambda e: self.switch_model(3))
        self.bind('0', lambda e: self.switch_model(0))

        # --- HILO DE CONTROL MANUAL (RESTAURADO DEL CÓDIGO ORIGINAL) ---
        self.print_menu_console()
        self.manual_thread = threading.Thread(target=self.manual_control_loop, daemon=True)
        self.manual_thread.start()

        # Loop UI
        self.update_map_ui()
        self.update_telemetry_ui()

    # ==================== MÉTODOS DE BOTONES Y CALLBACKS (SEGUROS) ====================
    
    def toggle_connection(self):
        if self.mavlink_connected:
            self.disconnect_drone()
        else:
            threading.Thread(target=self.connect_drone, daemon=True).start()

    def start_routine_thread(self):
        if not self.mavlink_connected:
            messagebox.showwarning("Alerta", "Dron no conectado. No se puede iniciar misión.")
            return
        threading.Thread(target=self.run_mission_logic, daemon=True).start()

    def btn_action_rtl_safe(self):
        if not self.mavlink_connected:
            messagebox.showwarning("Alerta", "Dron no conectado. No se puede ejecutar RTL.")
            return
        # AQUI ESPECIFICAMOS QUE ES MANUAL
        threading.Thread(target=self.handle_smart_rtl, kwargs={'is_manual': True}, daemon=True).start()

    def btn_action_land_safe(self):
        if not self.mavlink_connected:
            messagebox.showwarning("Alerta", "Dron no conectado. No se puede aterrizar.")
            return
        self.action_land()

    # ==================== GESTIÓN DE DISPOSITIVOS ====================
    def _load_all_models(self):
        if YOLO is None: return
        paths = {1: MODEL_1_PATH, 2: MODEL_2_PATH, 3: MODEL_3_PATH}
        for mid, path in paths.items():
            try:
                print(f"[INFO] Cargando Modelo {mid}: {path}")
                self.models[mid] = {'model': YOLO(path), 'path': path, 'name': Path(path).name}
                # Añadir classes
                self.models[mid]['classes'] = len(self.models[mid]['model'].names)
                print(f"[SUCCESS] Modelo {mid} cargado: {self.models[mid]['classes']} clases")
            except Exception as e:
                print(f"[ERROR] Modelo {mid} falló: {e}")
                self.models[mid] = None

    def switch_model(self, model_id):
        if model_id == 0:
            self.active_model_id = 0
            self.model = None
            self.model_loaded = False
            self.model_indicator.config(text="Modelo: Ninguno", bg="#FF6B6B")
            self.last_detections = []
            print(f"\n[INFO] Detección desactivada")
            return
        
        if model_id not in self.models or self.models[model_id] is None:
            print(f"\n[ERROR] Modelo {model_id} no está disponible")
            return
        
        self.active_model_id = model_id
        self.model = self.models[model_id]['model']
        self.model_loaded = True
        colors = {1: "#4CAF50", 2: "#2196F3", 3: "#FF9800"}
        name = self.models[model_id]['name']
        self.model_indicator.config(text=f"Modelo {model_id}: {name}", bg=colors.get(model_id, "#999"))
        print(f"\n[SUCCESS] Modelo {model_id} activado: {name}")

    def auto_select_external_camera(self, max_index=8):
        working = [] 
        for i in range(0, max_index):
            print(f"[INFO] Probando cámara index {i}...", end='\r')
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            cap.release()
            if ret: working.append(i)

        print(" " * 40, end='\r') 
        if not working: return 0
        externals = [i for i in working if i > 0]
        if externals:
            if 2 in externals: return 2
            return externals[0] 
        return working[0]

    def connect_drone(self):
        self.second_button.config(state='disabled')
        self.return_status_label.config(text="Conectando...", fg="orange")
        try:
            print(f"[INFO] Conectando a {MAVLINK_PORT}...")
            # LA CONEXIÓN SE REALIZA SOLO AQUI
            self.vehicle = connect(MAVLINK_PORT, baud=MAVLINK_BAUD, wait_ready=False)
            
            self.vehicle.add_message_listener('DISTANCE_SENSOR', self.listener_lidar)
            self.vehicle.add_message_listener('LOCAL_POSITION_NED', self.listener_position)
            self.vehicle.add_message_listener('ATTITUDE', self.listener_attitude)
            self.vehicle.add_message_listener('OPTICAL_FLOW', self.listener_flow)
            self.vehicle.add_message_listener('VFR_HUD', self.listener_hud)
            self.vehicle.add_message_listener('SYS_STATUS', self.listener_bat)
            
            self.mavlink_connected = True
            self.connection_time = time.time() # Guardar tiempo de conexión
            self.second_button.config(text="Conectado", bg="green", fg="white", state='normal')
            self.return_status_label.config(text="Dron Conectado", fg="green")
            print("[SUCCESS] DroneKit Conectado")
        except Exception as e:
            print(f"[ERROR] Conexión fallida: {e}")
            self.mavlink_connected = False
            self.second_button.config(text="Desconectado", bg="red", fg="white", state='normal')
            self.return_status_label.config(text="Error", fg="red")

    def disconnect_drone(self):
        print("[INFO] Desconectando del dron...")
        if self.vehicle: self.vehicle.close()
        self.vehicle = None 
        self.mavlink_connected = False
        self.second_button.config(text="Desconectado", bg="red", fg="white")
        self.return_status_label.config(text="Desconectado", fg="gray")
        print("[SUCCESS] Desconectado")

    def _safe_manual_action(self, action_func, *args, **kwargs):
        """Ejecuta una acción manual protegiendo contra EmergencyLandingError"""
        try:
            action_func(*args, **kwargs)
        except EmergencyLandingError:
            print(f"[MANUAL] Acción {action_func.__name__} interrumpida por Aterrizaje.")
            # Opcional: Si quieres asegurar que aterrice al romperse el hilo manual
            self.action_land()
        except Exception as e:
            print(f"[ERROR HILO MANUAL] {e}")

    # ==================== RESTAURACIÓN DE MENÚ Y CONTROL MANUAL ====================
    def print_menu_console(self):
        print("\n" + "="*40)
        print(" [a] Armar  | [t] Takeoff | [p] Disarm")
        print(" [w] Pared  | [g] CCW -90 | [h] CW 90")
        print(" [y] AUTO RUTINA | [r] SMART RTL")
        print(" [l] Land   | [k] KILL")
        print("="*40 + "\n")

    def manual_control_loop(self):
        while self.running:
            try:
                # Solo procesar si estamos conectados
                if self.vehicle:
                    if keyboard.is_pressed('a'):
                        self.action_arm()
                        time.sleep(0.5)
                    
                    if keyboard.is_pressed('p'):
                        self.action_disarm()
                        time.sleep(0.5)
                    
                    # if keyboard.is_pressed('t'):
                    #     threading.Thread(target=self.action_takeoff_normal, daemon=True).start()
                    #     time.sleep(0.5)
                    
                    # if keyboard.is_pressed('w'):
                    #     threading.Thread(target=self.action_move_forward_smart, daemon=True).start()
                    #     time.sleep(0.2)
                    
                    # if keyboard.is_pressed('g'):
                    #     threading.Thread(target=self.action_rotate_ccw_90, daemon=True).start()
                    #     time.sleep(0.5)
                        
                    # if keyboard.is_pressed('h'):
                    #     threading.Thread(target=self.action_rotate_cw_90, daemon=True).start()
                    #     time.sleep(0.5)

                    if keyboard.is_pressed('t'):
                        # PROTEGIDO: Despegue
                        threading.Thread(target=self._safe_manual_action, 
                                         args=(self.action_takeoff_normal,), 
                                         daemon=True).start()
                        time.sleep(0.5)
                    
                    if keyboard.is_pressed('w'):
                        # PROTEGIDO: Avanzar
                        threading.Thread(target=self._safe_manual_action, 
                                         args=(self.action_move_forward_smart,), 
                                         daemon=True).start()
                        time.sleep(0.2)
                    
                    if keyboard.is_pressed('g'):
                        # PROTEGIDO: Giro CCW
                        threading.Thread(target=self._safe_manual_action, 
                                         args=(self.action_rotate_ccw_90,), 
                                         daemon=True).start()
                        time.sleep(0.5)
                        
                    if keyboard.is_pressed('h'):
                        # PROTEGIDO: Giro CW
                        threading.Thread(target=self._safe_manual_action, 
                                         args=(self.action_rotate_cw_90,), 
                                         daemon=True).start()
                        time.sleep(0.5)
                    
                    if keyboard.is_pressed('y'):
                        self.start_routine_thread()
                        time.sleep(0.5)
                    
                    if keyboard.is_pressed('r'):
                        # Se asume manual si se presiona la tecla
                        threading.Thread(target=self.handle_smart_rtl, kwargs={'is_manual': True}, daemon=True).start()
                        time.sleep(0.5)

                    if keyboard.is_pressed('l'):
                        self.action_land()
                        time.sleep(0.5)
                        
                    if keyboard.is_pressed('k'):
                        self.action_kill()
                        time.sleep(0.5)

                time.sleep(0.05)

            except Exception as e:
                print(f"[ERROR MANUAL LOOP] {e}")
                time.sleep(1)

    # ==================== LISTENER LIDAR MEJORADO ====================
    def listener_lidar(self, v, n, m):
        raw_d = m.current_distance / 100.0 # Convertir cm a metros
        
        if m.orientation == ORIENT_FORWARD:
            # APLICAR FILTRO AQUÍ
            clean_d = self.lidar_filter.update(raw_d)
            self.telemetry_data["lidar_front"] = clean_d
            
            # Debug opcional para ver si está filtrando
            # print(f"Raw: {raw_d:.2f} | Filtered: {clean_d:.2f}")
            
        elif m.orientation == ORIENT_RIGHT: 
            self.telemetry_data["lidar_right"] = raw_d
        elif m.orientation == ORIENT_DOWN:
            self.telemetry_data["lidar_down"] = raw_d

    def listener_position(self, v, n, m):
        self.pos_buffer_x.append(m.x); self.pos_buffer_y.append(m.y)
        if len(self.pos_buffer_x) > 0:
            self.telemetry_data["pos_x"] = sum(self.pos_buffer_x)/len(self.pos_buffer_x)
            self.telemetry_data["pos_y"] = sum(self.pos_buffer_y)/len(self.pos_buffer_y)
        else:
            self.telemetry_data["pos_x"] = m.x; self.telemetry_data["pos_y"] = m.y

    def listener_attitude(self, v, n, m): self.telemetry_data["yaw"] = math.degrees(m.yaw)
    def listener_flow(self, v, n, m): 
        self.telemetry_data["opt_qua"] = m.quality
        self.telemetry_data["flow_x"] = m.flow_x; self.telemetry_data["flow_y"] = m.flow_y
    def listener_hud(self, v, n, m):
        self.telemetry_data["h_spd"] = m.groundspeed; self.telemetry_data["v_spd"] = m.climb
    def listener_bat(self, v, n, m):
        v = m.voltage_battery/1000.0
        self.telemetry_data["battery"] = v

    # ==================== VIDEO E IA (CON PROTECCIÓN FPV) ====================
    # --- NUEVA FUNCIÓN: VALIDAR SEÑAL DE VIDEO ---
    def is_video_signal_valid(self, frame):
        """
        Detecta pérdida de señal (pantalla azul/negra/gris sólida).
        Retorna False si la imagen tiene muy poca variación de color.
        """
        if frame is None: return False
        
        # Calcular desviación estándar de los colores
        (mean, std) = cv2.meanStdDev(frame)
        
        # Si la variación es menor a 15 en todos los canales (imagen plana)
        if std[0] < 15 and std[1] < 15 and std[2] < 15:
            return False
        return True

    # --- MODIFICADO: WORKER CON PERSISTENCIA ---
    def _inference_worker(self):
        # Tiempo de vida de una detección (en segundos) para soportar glitches
        DETECTION_PERSISTENCE = 1.0 
        last_valid_detection_time = 0
        
        while self.running:
            try: frame = self.inference_queue.get(timeout=0.2)
            except: continue
            
            if not self.detection_enabled or not self.model_loaded: continue
            
            # 1. VERIFICAR CALIDAD DE SEÑAL
            if not self.is_video_signal_valid(frame):
                # Si la señal es mala, verificar si estamos en tiempo de persistencia
                if time.time() - last_valid_detection_time < DETECTION_PERSISTENCE:
                    # Mantenemos las detecciones anteriores (asumimos glitch)
                    continue 
                else:
                    # Pasó mucho tiempo sin señal, limpiar
                    self.last_detections = []
                    continue

            try:
                res = self.model.predict(frame, verbose=False)
                dets = []
                found_something = False
                
                if res:
                    for b in res[0].boxes:
                        c = float(b.conf[0])
                        if c > 0.50:
                            dets.append({'xyxy': b.xyxy[0].tolist(), 'conf': c, 'cls': int(b.cls[0]), 'label': self.model.names[int(b.cls[0])]})
                            found_something = True
                
                # 2. LÓGICA DE PERSISTENCIA
                if found_something:
                    self.last_detections = dets
                    last_valid_detection_time = time.time()
                else:
                    # Si no hay detección pero hace poco vimos algo, mantener lo anterior
                    if time.time() - last_valid_detection_time > DETECTION_PERSISTENCE:
                        self.last_detections = []
                    # Si estamos dentro del tiempo de persistencia, self.last_detections no se borra
                
                self.inferring = False
            except: self.inferring = False

    def update_video(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if ret:
            if self.model_loaded and not self.inferring and (time.time() - self._last_inference_time) > self.inference_interval:
                try:
                    self.inference_queue.put_nowait(frame.copy())
                    self.inferring = True; self._last_inference_time = time.time()
                except: pass
            
            frame = cv2.resize(frame, (self.display_w, self.display_h))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.last_detections:
                draw = ImageDraw.Draw(img)
                c = {1:'lime', 2:'cyan', 3:'orange'}.get(self.active_model_id, 'lime')
                for d in self.last_detections:
                    x1,y1,x2,y2 = d['xyxy']
                    draw.rectangle([x1,y1,x2,y2], outline=c, width=2)
                    draw.text((x1, y1-10), f"{d['label']} {d['conf']:.2f}", fill='white')
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_rect.configure(image=imgtk); self.video_rect.imgtk = imgtk
        self.after(30, self.update_video)

    def update_map_ui(self):
        if not self.running: return
        
        dx = self.telemetry_data["pos_x"] - self.offset_x
        dy = self.telemetry_data["pos_y"] - self.offset_y
        th = -self.arming_yaw_rad
        rx = dx*math.cos(th) - dy*math.sin(th)
        ry = dx*math.sin(th) + dy*math.cos(th)
        px = int(MAP_HOME_PIXEL_X + (ry * MAP_SCALE))
        py = int(MAP_HOME_PIXEL_Y - (rx * MAP_SCALE))

        if self.vehicle and self.vehicle.armed:
            if not self.trajectory_points: self.trajectory_points.append((px, py))
            else:
                lp = self.trajectory_points[-1]
                if math.sqrt((px-lp[0])**2 + (py-lp[1])**2) > 5.0: self.trajectory_points.append((px, py))

        dimg = self.map_image_base.copy()
        if len(self.trajectory_points) > 1:
            pts = np.array(self.trajectory_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(dimg, [pts], False, (255, 0, 0), 2)
        cv2.circle(dimg, (px, py), 5, (0, 0, 255), -1)

        final = cv2.resize(dimg, (320, 240))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)))
        self.image_rect.configure(image=imgtk); self.image_rect.imgtk = imgtk
        
        desc = STEP_DESCRIPTIONS.get(self.current_step, f"Paso {self.current_step}")
        self.step_label.config(text=f"{desc}")
        self.search_status_label.config(text=f"Buscando: {self.searching_for}")
        
        wd = self.mission_metrics['wall_dist']; gd = self.mission_metrics['goal']
        wds = f"{wd:.2f}m" if wd > 0 else "--"
        gds = f"{gd:.2f}m" if gd > 0 else "--"
        self.metrics_label.config(text=f"Avance: {self.mission_metrics['dist_advanced']:.2f}m | Pared: {wds} | Meta: {gds}")

        self.after(100, self.update_map_ui)

    # --- MODIFICADO: ALERTA DE BATERÍA EN UI ---
    def update_telemetry_ui(self):
        if self.mavlink_connected:
            b = self.telemetry_data["battery"]
            a = self.telemetry_data["lidar_down"]
            
            # --- ALERTA VISUAL DE BATERÍA BAJA (CON DELAY) ---
            # Solo muestra alerta si han pasado 5s desde la conexión
            if (time.time() - self.connection_time > 5.0) and (b < self.battery_warn_threshold):
                self.battery_label.config(text=f"¡BATERÍA BAJA! {b:.2f}V", fg="red", font=("Arial", 22, "bold"))
                # Parpadeo en etiqueta central
                if int(time.time() * 2) % 2 == 0:
                    self.center_label.config(text="¡BATERÍA CRÍTICA!", fg="red", bg="yellow")
                else:
                    self.center_label.config(text="¡ATERRIZAR AHORA!", fg="white", bg="red")
            else:
                self.battery_label.config(text=f"Batería: {b:.2f}V", fg="green", font=("Arial", 18))
                # Restaurar etiqueta central si no hay otras alertas
                if "CRÍTICA" in self.center_label.cget("text"):
                    self.center_label.config(text="Sin Anomalías", fg="green", bg="#75B9BE")
            # -------------------------------------
            
            diff = abs(a - TARGET_ALTITUDE)
            alt_col = "green" if diff < 0.20 else "red"
            self.altitude_label.config(text=f"Altura: {a:.2f}m", fg=alt_col)
        else:
            self.altitude_label.config(text="Altura: N/A", fg="black")
            
        self.after(1000, self.update_telemetry_ui)

    # ==================== ACCIONES DEL DRON ====================
    
    # --- NUEVA FUNCIÓN: VERIFICAR ARMADO O ABORTAR ---
    def require_armed(self):
        """
        Si el dron no está armado, lanza una excepción para detener la misión inmediatamente.
        """
        if not self.vehicle or not self.vehicle.armed:
            print("[CRITICAL] Dron desarmado en vuelo. Abortando misión.")
            raise EmergencyLandingError()

    def force_full_stop(self, duration):
        """
        Envía comando de velocidad 0,0,0 repetidamente para frenar la inercia.
        Esencial antes de girar para evitar derrapes.
        """
        self.require_armed() # VERIFICACIÓN DE SEGURIDAD
        print(f"[STOP] Frenando inercia por {duration} segundos...")
        
        t_start = time.time()
        while time.time() - t_start < duration:
            self.require_armed() # VERIFICACIÓN CONTINUA
            self.set_velocity(0, 0, 0)
            self.check_emergency()
            time.sleep(0.1)
            
        print("[STOP] Freno completado. Listo para siguiente comando.")

    def action_arm(self):
        print("[CMD] Armando...")
        if self.telemetry_data["yaw"] is None:
            print("[WARN] No hay Yaw disponible")
            return False
        
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        
        # --- VERIFICACIÓN DE SEGURIDAD PARA ARMADO ---
        print("[INFO] Esperando confirmación de armado...")
        t_start_arm = time.time()
        while not self.vehicle.armed:
            if time.time() - t_start_arm > 5.0: # 5 segundos timeout
                print("[ERROR] NO SE PUDO ARMAR. Revise Safety Switch o Pre-Arm Checks.")
                return False
            time.sleep(0.1)
        # ---------------------------------------------------
        
        self.trajectory_points = []
        self.arming_yaw_rad = math.radians(self.telemetry_data["yaw"])
        self.offset_x = self.telemetry_data["pos_x"]
        self.offset_y = self.telemetry_data["pos_y"]
        self.pos_buffer_x.clear(); self.pos_buffer_y.clear()
        time.sleep(1)
        self.target_heading = self.telemetry_data["yaw"]
        print(f"[SYSTEM] Rumbo bloqueado en: {self.target_heading:.1f}°")
        return True

    def action_disarm(self):
        print("[CMD] Desarmando...")
        self.vehicle.armed = False

    def action_takeoff_normal(self):
        # --- NUEVO: SI NO ESTÁ ARMADO, ERROR FATAL ---
        if not self.vehicle.armed:
            print("[CRITICAL] Intento de despegue sin armado. Abortando.")
            raise EmergencyLandingError()
        # ---------------------------------------------
        
        self.action_lock = True
        print(f"[CMD] Despegando a {TARGET_ALTITUDE}m")
        self.vehicle.simple_takeoff(TARGET_ALTITUDE)
        start = time.time()
        while True:
            self.require_armed() # Check continuo
            self.check_emergency()
            alt = self.telemetry_data["lidar_down"]
            if alt is None: alt = 0.0
            
            print(f">> Ascendiendo: {alt:.2f}/{TARGET_ALTITUDE}m", end='\r')
            
            if alt >= TARGET_ALTITUDE * 0.95:
                print("\n[INFO] Altura alcanzada. Recalibrando...")
                time.sleep(0.5)
                self.trajectory_points = []
                self.offset_x = self.telemetry_data["pos_x"]
                self.offset_y = self.telemetry_data["pos_y"]
                self.arming_yaw_rad = math.radians(self.telemetry_data["yaw"])
                # --- NUEVO: GUARDAR YAW INICIAL EN GRADOS PARA ALINEACION FINAL ---
                self.initial_yaw_deg = self.telemetry_data["yaw"]
                break
            
            # Timeout con ERROR FATAL
            if time.time() - start > 20.0: 
                print("\n[CRITICAL] Timeout despegue. Abortando.")
                raise EmergencyLandingError() # AHORA SÍ DETIENE LA MISIÓN
            
            time.sleep(0.1)
        self.action_lock = False

    def action_land(self):
        print("[CMD] Aterrizando (LAND)...")
        if self.vehicle:
            self.vehicle.mode = VehicleMode("LAND")
        
        # --- CAMBIO SOLICITADO AQUÍ: Actualizar texto visual ---
        self.current_step = 27

    def action_kill(self):
        print("[KILL] EMERGENCIA MOTORES")
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0, mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION, 0, 1, 0, 0, 0, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.armed = False

    def action_rotate_cw_90(self):
        self._rotate_generic(90)

    def action_rotate_ccw_90(self):
        self._rotate_generic(-90)

    # --- CORRECCIÓN EN _rotate_generic (Bloqueo Perfecto) ---
    def _rotate_generic(self, angle):
        self.require_armed() 
        self.action_lock = True
        
        # 1. Calcular Ángulo Objetivo (Absoluto)
        # IMPORTANTE: Usamos self.target_heading (la referencia ideal) en vez del sensor real
        # Esto evita que los errores se acumulen.
        # Si es el primer giro, self.target_heading ya se inicializó en el despegue.
        
        # Aseguramos que target_heading esté inicializado
        if self.target_heading is None: 
             self.target_heading = self.telemetry_data["yaw"]

        # Nuevo objetivo ideal
        target_yaw = (self.target_heading + angle + 180) % 360 - 180
        
        self.force_full_stop(2.0) 
        print(f"[CMD] Girando {angle} grados a objetivo {target_yaw:.1f}")
        
        # Enviar comando MAVLink (Se mantiene igual)
        direction = 1 if angle > 0 else -1
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, abs(angle), 15, direction, 1, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        
        # Bucle de espera
        t_start = time.time()
        while time.time() - t_start < 8.0:
            self.require_armed()
            self.check_emergency()
            
            curr = self.telemetry_data["yaw"]
            if curr is not None:
                err = abs(target_yaw - curr)
                if err > 180: err = 360 - err
                
                # RECOMENDACIÓN: Bajar la tolerancia de 5.0 a 3.0
                if err < 3.0: 
                    print(f"[OK] Giro completado. Error: {err:.1f}°")
                    break
            time.sleep(0.1)
            
        # AHORA: Forzamos el objetivo matemático perfecto.
        # Esto obliga al Kp a corregir la diferencia mientras avanza en el siguiente tramo.
        self.target_heading = target_yaw 
        
        self.action_lock = False

    # --- NUEVA FUNCIÓN: FORZAR ALINEACIÓN ANTES DE MOVER ---
    def enforce_heading_alignment(self):
        """
        Alinea el dron con self.target_heading ANTES de iniciar el avance.
        Evita que avance 'chueco' si el giro anterior tuvo error.
        """
        self.require_armed()
        target = self.target_heading
        
        # Solo alinear si el error es significativo (> 2 grados)
        curr = self.telemetry_data["yaw"]
        if curr:
            err = abs(target - curr)
            if err > 180: err = 360 - err
            if err < 2.0: return # Ya está alineado
            
        print(f"[ALIGN] Corrigiendo Yaw a {target:.1f}° antes de avanzar...")
        
        # Comando absoluto para corregir el error remanente
        # CORRECCIÓN: Se añadió el '0' de confirmación que faltaba antes
        target_mav = target if target >= 0 else target + 360
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, # Confirmation
            target_mav, 0, 0, 0, 0, 0, 0) # Param 3 = 0 (Shortest Path)
        self.vehicle.send_mavlink(msg)
        
        # Espera breve
        st = time.time()
        while time.time() - st < 3.0: # Timeout corto 3s
            curr = self.telemetry_data["yaw"]
            if curr:
                err = abs(target - curr)
                if err > 180: err = 360 - err
                if err < 2.0: break
            time.sleep(0.1)

    # --- NUEVA FUNCIÓN: ALINEACIÓN ABSOLUTA AL INICIO (EVITA LATIGAZO) ---
    def action_align_to_start(self):
        self.require_armed()
        self.action_lock = True
        
        target = self.initial_yaw_deg
        # Asegurar rango 0-360 positivo para MAVLink si usamos absoluto
        target_mav = target if target >= 0 else target + 360
        
        # Frenar antes de alinear
        self.force_full_stop(2.0)
        
        print(f"[CMD] Alineando al Norte Inicial (Absoluto): {target:.1f}°")
        
        # Enviar Comando Yaw ABSOLUTO (Param4 = 0)
        # Esto obliga al dron a buscar el camino más corto al 0 original
        # CORRECCIÓN: Se añadió el '0' de confirmación antes de target_mav
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,             # Confirmation
            target_mav,    # Param 1: Angulo objetivo (0-360)
            0,             # Param 2: Velocidad angular (0 = auto)
            0,             # Param 3: 0 = SHORTEST PATH (CAMINO MAS CORTO)
            0,             # Param 4: 0 = ABSOLUTO
            0, 0, 0)       # Param 5, 6, 7 (Vacíos)
            
        self.vehicle.send_mavlink(msg)
        
        # Bucle de espera
        t_start = time.time()
        while time.time() - t_start < 8.0:
            self.require_armed()
            self.check_emergency()
            
            curr = self.telemetry_data["yaw"]
            if curr is not None:
                # Calculo de error con wrap-around (-180 a 180)
                err = abs(target - curr)
                if err > 180: err = 360 - err
                
                if err < 5.0:
                    print(f"[OK] Alineación completada.")
                    break
            time.sleep(0.1)
            
        self.target_heading = self.telemetry_data["yaw"]
        self.action_lock = False

    # --- MODIFICADO: AÑADIDA ALINEACIÓN PREVIA ---
    def action_move_forward_smart(self, detect_intruders=True):
        self.require_armed() 
        if self.telemetry_data["lidar_front"] is None:
            print("[ERROR] No Lidar Frontal"); return
        
        # 1. ALINEACIÓN PREVIA (CORRIGE EL ERROR DEL GIRO ANTERIOR)
        self.enforce_heading_alignment()
        
        print(f"[CMD] Avanzando hasta pared {TARGET_DISTANCE}m")
        # ELIMINADO: self.target_heading = self.telemetry_data["yaw"] (CAUSABA EL ERROR)
        
        self.action_lock = True
        sx = self.telemetry_data["pos_x"]; sy = self.telemetry_data["pos_y"]
        
        if detect_intruders:
            # Ahora la selección de modelo está arriba en run_mission_logic
            self.searching_for = "Personas (Intrusos)"
        
        # --- NUEVO: CONTADOR DE PERSISTENCIA ---
        stop_confirmation_count = 0 
        REQUIRED_STOPS = 3 # Necesita ver la pared 3 veces seguidas para parar

        st = time.time()
        while True:
            self.require_armed() 
            self.check_emergency()
            
            # SOLUCIÓN DEL BUG: Solo detectamos si detect_intruders es True
            if detect_intruders:
                self.check_detections_continuous()
            
            lidar = self.telemetry_data["lidar_front"]
            vel = TARGET_VELOCITY
            if lidar and lidar <= (TARGET_DISTANCE * 1.2): vel = 0.02
            
            self.set_velocity(vel, 0.0, 0.0)
            
            # --- LÓGICA DE PARADA ROBUSTA ---
            if lidar and lidar <= TARGET_DISTANCE:
                stop_confirmation_count += 1
                print(f"[DEBUG] Pared detectada ({stop_confirmation_count}/{REQUIRED_STOPS})")
            else:
                stop_confirmation_count = 0 # Reiniciar si deja de ver la pared
                
            if stop_confirmation_count >= REQUIRED_STOPS:
                print("\n[OK] Pared confirmada. Deteniendo.")
                break
            # --------------------------------
            
            dx = self.telemetry_data["pos_x"] - sx; dy = self.telemetry_data["pos_y"] - sy
            self.mission_metrics['dist_advanced'] = math.sqrt(dx**2 + dy**2)
            self.mission_metrics['wall_dist'] = lidar if lidar else 0.0
            self.mission_metrics['goal'] = TARGET_DISTANCE
            
            l_str = f"{lidar:.2f}" if lidar else "N/A"
            print(f">> Avanzando: {l_str}m", end='\r')
            
            if time.time() - st > 90.0: break
            time.sleep(0.1)
            
        self.set_velocity(0.0, 0.0, 0.0)
        self.action_lock = False

    # --- MODIFICADO: PARÁMETRO detect_intruders ---
    def wait_and_hold(self, seconds, check_human=True):
        t = time.time()
        while time.time() - t < seconds:
            self.require_armed() 
            self.check_emergency()
            if check_human: self.check_detections_continuous()
            self.set_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)

    # --- CORRECCIÓN FINAL DE YAW (Kp=0.12 y TOLERANCIA 2.0°) ---
    def get_yaw_correction(self):
        curr = self.telemetry_data["yaw"]
        if curr is None: return 0.0
        
        # Calcula el error más corto (-180 a 180)
        err = (self.target_heading - curr + 180)%360 - 180
        
        # 1. TOLERANCIA EXACTA DE 2.0 GRADOS
        if abs(err) < 2.0: return 0.0 
        
        # 2. GANANCIA PROPORCIONAL (Kp)
        # AUMENTADA A 0.12 PARA CORREGIR DERIVA
        return max(min(math.radians(err * 0.12), 0.5), -0.5)

    def set_velocity(self, vx, vy, vz):
        yaw_rate = self.get_yaw_correction()
        try:
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 0b0000011111000111,
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)
            self.vehicle.send_mavlink(msg)
        except: pass 

    def check_emergency(self):
        # Esta función revisa teclado DENTRO de los loops de misión
        # Si se presiona 'r', lanza ManualRTLException
        if keyboard.is_pressed('l'): raise EmergencyLandingError()
        if keyboard.is_pressed('k'): self.action_kill(); raise EmergencyLandingError()
        if keyboard.is_pressed('r') and self.current_step < 17: # Paso 17 es la frontera de Tramo 2
            time.sleep(0.2)
            raise ManualRTLException()

    def check_detections_continuous(self):
        if self.active_model_id == 3 and self.last_detections:
            for d in self.last_detections:
                print(f"[ALERTA] Intruso detectado: {d['label']} {d['conf']}")
                # USAR SELF.AFTER para actualizar GUI desde el hilo
                self.after(0, lambda: self.center_label.config(text="ANOMALÍA DETECTADA (Intruso)", fg="red"))
                # Esto es un fallo automático, lanza SmartRTLException
                raise SmartRTLException()
        else:
            try:
                # Comprobación segura
                current_text = self.center_label.cget("text")
                if "ANOMALÍA" not in current_text and "RETORNO" not in current_text:
                    self.after(0, lambda: self.center_label.config(text="Sin Anomalías", fg="green"))
            except RuntimeError: pass

    # --- MODIFICADO: LOGICA RTL DETENER BUSQUEDA ---
    def handle_smart_rtl(self, is_manual=True):
        # PRIMER PASO: APAGAR LA VISIÓN PARA SIEMPRE EN ESTE VUELO
        self.switch_model(0)
        
        step = self.current_step
        print(f"[RTL] Smart RTL activado en paso {step}. Manual: {is_manual}")
        
        # SI ES MANUAL: PONE EL CARTEL NARANJA
        if is_manual:
            self.searching_for = "RETORNO ACTIVO"
            self.after(0, lambda: self.center_label.config(text="RETORNO ACTIVO", fg="orange"))
        # SI ES AUTOMATICO (SMART): NO CAMBIA EL CARTEL (RESPETA "ANOMALÍA DETECTADA")
        
        try:
            # --- FASE 1: ATERRIZAJE INMEDIATO (INICIO) ---
            if 1 <= step <= 6:
                self.action_land()

            # --- FASE 2: RETORNO (TRAMO 1) ---
            elif 7 <= step <= 9:
                # Backtracking
                self.action_rotate_cw_90(); self.action_rotate_cw_90()
                self.wait_and_hold(2, False)
                self.action_move_forward_smart(detect_intruders=False)
                self.wait_and_hold(1, False)
                self.action_rotate_cw_90(); self.action_rotate_cw_90()
                self.wait_and_hold(2, False)
                self.action_land()

            # --- FASE 3: RETORNO DESDE GIRO 1 ---
            elif step == 10:
                print("[RTL] Interrupción en Giro 1. Regresando...")
                self.force_full_stop(2.0)
                # Completar orientación hacia atrás
                self.action_rotate_cw_90() 
                self.wait_and_hold(2, False)
                self.action_move_forward_smart(detect_intruders=False)
                self.wait_and_hold(1, False)
                self.action_rotate_cw_90(); self.action_rotate_cw_90()
                self.action_land()

            # --- FASE 4: RETORNO DESDE ESTABILIZACIÓN 1 ---
            elif step == 11:
                print("[RTL] Interrupción en Estabilización 1. Regresando...")
                self.force_full_stop(2.0)
                self.action_rotate_cw_90() # Mirar al sur
                self.wait_and_hold(2, False)
                self.action_move_forward_smart(detect_intruders=False)
                self.wait_and_hold(1, False)
                self.action_rotate_cw_90(); self.action_rotate_cw_90()
                self.action_land()

            # --- FASE 5: COMPLETAR MISIÓN (PASOS 12+) ---
            # Si ocurre de aquí en adelante, simplemente terminamos la rutina
            elif step >= 12:
                print("[RTL] Continuando misión hasta finalizar...")
                
                # Ejecutamos secuencialmente los pasos restantes
                if step <= 13: # Tramo 2
                    if step == 12:
                         self.action_move_forward_smart(detect_intruders=False) # Ignoramos intrusos
                         self.wait_and_hold(1, False)

                if step <= 14: # Check Lámpara (Saltar check)
                     pass

                if step <= 15: # Giro 2
                    self.force_full_stop(2.0)
                    self.action_rotate_cw_90()
                    self.wait_and_hold(3, False)

                if step <= 18: # Tramo 3 (Saltar inicio 17, ir directo a avance 18)
                    self.action_move_forward_smart(detect_intruders=False)
                    self.wait_and_hold(1, False)

                if step <= 20: # Giro 3
                    self.force_full_stop(2.0)
                    self.action_rotate_cw_90()
                    self.wait_and_hold(3, False)

                if step <= 23: # Tramo 4
                    self.action_move_forward_smart(detect_intruders=False)
                    self.wait_and_hold(1, False)

                if step <= 25: # Giro 4
                    self.force_full_stop(2.0)
                    # CORRECCIÓN: Usar alineación absoluta para evitar latigazo
                    self.action_align_to_start() # NUEVA FUNCIÓN ABSOLUTA
                    self.wait_and_hold(3, False)

                self.action_land()

        except EmergencyLandingError: self.action_land()

    # --- NUEVA FUNCIÓN: VERIFICACIÓN ROBUSTA DE OBJETIVO ---
    def verify_visual_target(self, model_id, target_name, timeout=4.0):
        """
        Verificación robusta para cámaras FPV analógicas.
        """
        self.require_armed() # SEGURIDAD
        print(f"[CHECK] Verificando presencia de {target_name}...")
        self.searching_for = f"{target_name} (Verificando)"
        self.switch_model(model_id)
        
        t_start = time.time()
        positive_frames = 0
        REQUIRED_FRAMES = 5
        
        while time.time() - t_start < timeout:
            self.require_armed() # SEGURIDAD
            self.check_emergency()
            self.set_velocity(0,0,0)
            
            if self.last_detections:
                positive_frames += 1
                if positive_frames >= REQUIRED_FRAMES:
                    print(f"[OK] {target_name} CONFIRMADO ({positive_frames} frames).")
                    return True
            time.sleep(0.1)
            
        print(f"[FALLO] {target_name} NO confirmado tras {timeout}s.")
        return False

    # --- MÉTODO AUXILIAR PARA RESTAURAR UI DE FORMA SEGURA ---
    def reset_ui_after_mission(self):
        self.current_step = 0
        self.sequence_button.config(state='normal', bg='blue', text='Iniciar Misión')
        self.searching_for = "Nada"
        self.switch_model(0)
        self.save_map_image()

    def run_mission_logic(self):
        print("[AUTO] Iniciando Misión...")
        self.after(0, lambda: self.sequence_button.config(state='disabled', bg='orange', text='Misión en curso...'))
        
        try:
            self.current_step = 1
            
            # --- CAMBIO SOLICITADO: DETECCIÓN DE PERSONAS DESDE EL INICIO ---
            self.switch_model(3) # Modelo Personas
            self.searching_for = "Personas (Intrusos)"
            # ---------------------------------------------------------------

            # --- PROTECCIÓN DE INICIO DE MISIÓN ---
            if not self.action_arm():
                messagebox.showerror("Error Crítico", "Fallo al armar motores.\n\nPosibles causas:\n- Safety Switch activado\n- Voltaje bajo")
                self.after(0, self.reset_ui_after_mission)
                return
            # --------------------------------------

            self.current_step = 2; self.wait_and_hold(3.0)
            self.current_step = 3; self.action_takeoff_normal()
            self.current_step = 4; self.wait_and_hold(3.0)
            self.current_step = 5 # Recalibración (Implícita en lógica interna)
            
            self.current_step = 6
            # Fin estabilización
            
            # --- TRAMO 1 ---
            self.current_step = 7 # Inicio Tramo 1 (Bloqueo)
            
            self.current_step = 8 # Avance Tramo 1
            self.action_move_forward_smart()
            
            # --- PASO 9: CHECK DRON ---
            self.current_step = 9
            if not self.verify_visual_target(1, "DRON"):
                self.after(0, lambda: self.center_label.config(text="ANOMALÍA DETECTADA (Objeto perdido)", fg="red"))
                # SI NO HAY DRON -> ABORTAR (RTL)
                raise SmartRTLException() 
            
            self.after(0, lambda: self.center_label.config(text="Sin Anomalías", fg="green"))
            self.switch_model(3)
            self.searching_for = "Personas"

            # --- GIRO 1 ---
            self.current_step = 10
            self.action_rotate_cw_90()
            
            self.current_step = 11; self.wait_and_hold(3.0)
            
            # --- TRAMO 2 ---
            self.current_step = 12
            
            self.current_step = 13 # Avance Tramo 2
            self.action_move_forward_smart()
            
            # --- PASO 14: CHECK LAMPARA ---
            self.current_step = 14
            if not self.verify_visual_target(2, "LÁMPARA"):
                # SI NO HAY LÁMPARA -> RTL SIMULADO (ACTIVAR ALERTA Y SEGUIR)
                print("[AUTO] Lámpara no encontrada. ACTIVANDO PROTOCOLO ANOMALÍA.")
                self.after(0, lambda: self.center_label.config(text="ANOMALÍA DETECTADA (Objeto perdido)", fg="red"))
                # Lanza excepción automática para entrar en modo retorno sin intervención manual
                raise SmartRTLException() 
            
            self.after(0, lambda: self.center_label.config(text="Sin Anomalías", fg="green"))
            self.switch_model(3)
            self.searching_for = "Personas"

            # --- GIRO 2 ---
            self.current_step = 15
            self.action_rotate_cw_90()
            
            self.current_step = 16; self.wait_and_hold(3.0)
            
            # --- TRAMO 3 ---
            self.current_step = 17
            
            self.current_step = 18 # Avance Tramo 3
            self.action_move_forward_smart()
            
            self.current_step = 19
            self.wait_and_hold(3.0)
            
            # --- GIRO 3 ---
            self.current_step = 20
            self.action_rotate_cw_90()
            
            self.current_step = 21; self.wait_and_hold(3.0)
            
            # --- TRAMO 4 ---
            self.current_step = 22
            
            self.current_step = 23 # Avance Tramo 4
            self.action_move_forward_smart()
            
            self.current_step = 24
            self.wait_and_hold(3.0)
            
            # --- GIRO 4 ---
            self.current_step = 25
            # self.action_rotate_cw_90() <-- ESTO CAUSABA EL LATIGAZO
            self.action_align_to_start() # NUEVA FUNCIÓN ABSOLUTA
            
            self.current_step = 26; self.wait_and_hold(3.0)
            
            # --- LAND ---
            self.current_step = 27
            self.action_land()
            
            self.current_step = 28
            print("[FIN] Rutina Exitosa")

        except ManualRTLException:
            self.handle_smart_rtl(is_manual=True)
        except SmartRTLException:
            self.handle_smart_rtl(is_manual=False)
        except EmergencyLandingError:
            self.action_land()
        finally:
            self.after(0, self.reset_ui_after_mission)

    def save_map_image(self):
        if self.trajectory_points:
            f = self.map_image_base.copy()
            pts = np.array(self.trajectory_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(f, [pts], False, (0,0,255), 3)
            cv2.circle(f, self.trajectory_points[0], 8, (0,255,0), -1)
            cv2.circle(f, self.trajectory_points[-1], 8, (0,0,255), -1)
            if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            p = os.path.join(OUTPUT_FOLDER, f"Vuelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(p, f)
            print(f"[SAVE] Mapa guardado: {p}")

    def on_closing(self):
        self.running = False
        if self.vehicle: self.vehicle.close()
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SuperInterface()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()