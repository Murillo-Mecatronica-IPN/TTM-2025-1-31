# Código de la interfaz de usuario usando Python

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageDraw

class SimpleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interfaz Simple")
        self.attributes('-fullscreen', True)
        self.configure(bg="#86b9b0")

        # Offset para bajar todo menos el título principal
        offset = 40

        # Área del rectángulo de video (izquierda, centrado verticalmente)
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        video_width = 640
        video_height = 480
        x = int(screen_width * 0.1)
        y = int((screen_height - video_height) / 2) + offset
        self.video_rect = tk.Label(self, bg="#222222", width=video_width, height=video_height)
        self.video_rect.place(x=x, y=y, width=video_width, height=video_height)

        # Captura de video con OpenCV
        self.cap = cv2.VideoCapture(0)
        self.update_video()

        # Etiqueta centrada sobre el rectángulo de video
        video_label = tk.Label(
            self,
            text="Video en tiempo real",
            bg="#86b9b0",
            fg="white",
            font=("Arial", 24, "bold")
        )
        video_label.place(
            x=x + video_width // 2,
            y=y - 20,
            anchor="center"
        )

        # Variables de ejemplo para telemetría
        self.bateria = tk.DoubleVar(value=21.0)
        self.velocidad = tk.DoubleVar(value=12.3)
        self.altura = tk.DoubleVar(value=5.7)

        # Estado de ejemplo para el centro
        x_state = 1  # Cambia a 0 para probar el otro estado

        # Etiqueta central que cambia según el estado
        if x_state == 1:
            center_text = "Sin Anomalias"
            center_color = "green"
        else:
            center_text = "Anomalia Detectada"
            center_color = "red"

        center_label = tk.Label(
            self,
            text=center_text,
            bg="#86b9b0",
            fg=center_color,
            font=("Arial", 32, "bold")
        )
        center_label.place(
            x=x + video_width // 2,
            y=y + video_height + 50,
            anchor="center"
        )

        # Título principal arriba (no se mueve con offset)
        notif_label = tk.Label(
            self,
            text="Sistema de vigilancia",
            bg="#86b9b0",
            fg="black",
            font=("Arial", 48, "bold")
        )
        notif_label.place(
            relx=0.5, rely=0.1, anchor="center"
        )

        # Dimensiones y posición del rectángulo del mapa (abajo a la derecha)
        img_width = 320
        img_height = 240
        img_x = int(screen_width * 0.9) - img_width
        img_y = screen_height - img_height - 200 + offset

        # Etiqueta arriba y centrada con el rectángulo del mapa
        map_label = tk.Label(
            self,
            text="Ubicación del dron",
            bg="#86b9b0",
            fg="white",
            font=("Arial", 20, "bold")
        )
        map_label.place(
            x=img_x + img_width // 2,
            y=img_y - 20,
            anchor="center"
        )

        # Botones entre el rectángulo de video y el de mapa
        button_width = 180
        button_height = 50

        # Coordenadas de los lados de los rectángulos
        video_right = x + video_width
        map_left = img_x

        # X centrado entre ambos lados
        button_x = video_right + (map_left - video_right) // 2 - button_width // 2
        # Y centrado verticalmente respecto al mapa
        button_y = img_y + (img_height // 2) - button_height - 10
        button2_y = img_y + (img_height // 2) + 10

        # Estado del botón "Modo de Vuelo"
        self.Button1_State = 1  # 1 = Automático, 0 = Manual

        # Función para cambiar el estado y actualizar la etiqueta
        def toggle_modo():
            self.Button1_State = 0 if self.Button1_State == 1 else 1
            if self.Button1_State == 1:
                modo_text = "Modo: Automático"
            else:
                modo_text = "Modo: Manual"
            self.modo_label.config(text=modo_text)

        # Etiqueta arriba del botón "Modo de Vuelo" que depende del estado
        modo_text = "Modo: Automático" if self.Button1_State == 1 else "Modo: Manual"
        self.modo_label = tk.Label(self, text=modo_text, bg="#86b9b0", fg="white", font=("Arial", 16, "bold"))
        self.modo_label.place(
            x=button_x + button_width // 2,
            y=button_y - 20,
            anchor="center"
        )

        # Botón "Modo de Vuelo" con función de cambio de estado
        self.middle_button = tk.Button(self, text="Modo de Vuelo", font=("Arial", 18), command=toggle_modo)
        self.middle_button.place(x=button_x, y=button_y, width=button_width, height=button_height)

        # Segundo botón debajo del primero
        button2_y = button_y + button_height + 60
        self.second_button = tk.Button(self, text="Regreso a Base", font=("Arial", 18))
        self.second_button.place(x=button_x, y=button2_y, width=button_width, height=button_height)

        # Etiqueta de telemetría arriba del rectángulo del mapa
        self.telemetry_label = tk.Label(self, text="Telemetría", bg="#86b9b0", fg="black", font=("Arial", 28, "bold"))
        self.telemetry_label.place(x=img_x, y=img_y - 220, width=img_width, height=40)

        # Etiqueta de nivel de batería (rojo si <= 20 y cambia el texto)
        if self.bateria.get() <= 20:
            battery_color = "red"
            battery_text = "Conectar batería:"
        else:
            battery_color = "black"
            battery_text = "Nivel de batería:"
        self.battery_label = tk.Label(
            self,
            text=f"{battery_text} {self.bateria.get():.2f}",
            bg="#86b9b0",
            fg=battery_color,
            font=("Arial", 18)
        )
        self.battery_label.place(x=img_x, y=img_y - 180, width=img_width, height=30)

        # Etiqueta de velocidad
        self.speed_label = tk.Label(self, text=f"Velocidad: {self.velocidad.get():.2f}", bg="#86b9b0", fg="black", font=("Arial", 18))
        self.speed_label.place(x=img_x, y=img_y - 150, width=img_width, height=30)

        # Etiqueta de altura
        self.altitude_label = tk.Label(self, text=f"Altura: {self.altura.get():.2f}", bg="#86b9b0", fg="black", font=("Arial", 18))
        self.altitude_label.place(x=img_x, y=img_y - 120, width=img_width, height=30)

        # Imagen del mapa con un punto azul
        dot_x = 100  # píxeles desde la izquierda
        dot_y = 60   # píxeles desde arriba
        try:
            mapa_img = Image.open("Mapa.png")
            mapa_img = mapa_img.resize((img_width, img_height))
            draw = ImageDraw.Draw(mapa_img)
            # Dibuja un punto azul cerca de la esquina superior izquierda
            dot_radius = 10
            draw.ellipse(
                (dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius),
                fill="blue"
            )
            self.mapa_imgtk = ImageTk.PhotoImage(mapa_img)
            self.image_rect = tk.Label(self, image=self.mapa_imgtk, bg="#444444")
            self.image_rect.image = self.mapa_imgtk  # Evita que la imagen sea recolectada por el GC
        except Exception as e:
            self.image_rect = tk.Label(self, text="Mapa", bg="#444444", fg="white", font=("Arial", 32), anchor="center")
        self.image_rect.place(x=img_x, y=img_y, width=img_width, height=img_height)

        # Etiqueta debajo del mapa con las coordenadas del punto azul
        coords_text = f"Coordenadas: ({dot_x}, {dot_y})"
        coords_label = tk.Label(
            self,
            text=coords_text,
            bg="#86b9b0",
            fg="black",
            font=("Arial", 16, "bold")
        )
        coords_label.place(
            x=img_x + img_width // 2,
            y=img_y + img_height + 25,
            anchor="center"
        )

        # Actualización periódica de telemetría
        self.after(2000, self.update_telemetry)

    def update_video(self):
        # Actualiza el frame de video en tiempo real
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_rect.imgtk = imgtk
            self.video_rect.configure(image=imgtk)
        self.after(30, self.update_video)

    def update_telemetry(self):
        # Ejemplo: actualiza los valores de telemetría
        self.bateria.set(self.bateria.get() - 0.1)
        self.velocidad.set(self.velocidad.get() + 0.05)
        self.altura.set(self.altura.get() + 0.02)

        # Actualiza el color y texto de la batería según el valor
        if self.bateria.get() <= 75:
            battery_color = "red"
            battery_text = "Conectar batería:"
        else:
            battery_color = "black"
            battery_text = "Nivel de batería:"
        self.battery_label.config(
            text=f"{battery_text} {self.bateria.get():.2f}",
            fg=battery_color
        )
        self.speed_label.config(text=f"Velocidad: {self.velocidad.get():.2f}")
        self.altitude_label.config(text=f"Altura: {self.altura.get():.2f}")

        self.after(2000, self.update_telemetry)

    def on_closing(self):
        # Libera la cámara y cierra la ventana
        if hasattr(self, 'cap'):
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SimpleGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()