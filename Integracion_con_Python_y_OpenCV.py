# Código de integración con Python y OpenCV
import cv2

def find_camera_index(max_index=5):
    """Prueba índices de 0 a max_index-1 y devuelve el primero que abra."""
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return idx
    raise RuntimeError("No se encontró ninguna cámara UVC")
    
def main():
    # 1) Localiza automáticamente el índice de la cámara UVC
    cam_idx = find_camera_index()
    
    # 2) Abre la captura de video
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la cámara índice {cam_idx}")
        return
    
    print(f"Cámara abierta en índice {cam_idx}. Pulsar 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("¡Se perdió la señal de video!")
            break
    
        # 3) Muestra el fotograma
        cv2.imshow("FPV UVC Video", frame)
    
        # 4) Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()