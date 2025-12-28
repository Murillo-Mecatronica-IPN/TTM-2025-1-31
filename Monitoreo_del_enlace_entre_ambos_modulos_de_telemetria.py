# Código de monitoreo del enlace entre ambos módulos de telemetría

import time
from pymavlink import mavutil

# Configuración del puerto serial y velocidad de baudios
PORT = 'COM3'  # Se reemplaza con el puerto donde se conecta modulo Sik correspondiente en base local
BAUD = 57600

# Intervalo para verificar el latido (heartbeat) en segundos
HEARTBEAT_TIMEOUT = 5

# Inicializar la conexión MAVLink
print("Estableciendo conexión con el dron...")
master = mavutil.mavlink_connection(PORT, baud=BAUD)

# Esperar el primer latido del dron para confirmar la conexión
print("Esperando el latido inicial del dron...")
master.wait_heartbeat()
print(f"Conectado al sistema {master.target_system}, componente {master.target_component}")

# Enviar el comando RTL (Return to Launch)
print("Enviando comando RTL al dron...")
master.mav.command_long_send(
    master.target_system,       # ID del sistema objetivo
    master.target_component,    # ID del componente objetivo
    mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,  # Comando RTL
    0,  # Confirmación
    0, 0, 0, 0, 0, 0, 0  # Parámetros no utilizados para este comando
)

# Esperar y validar la respuesta COMMAND_ACK para confirmar la recepción del comando
print("Esperando confirmación del comando...")
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)

# Validar que se recibió el ACK y que corresponde al comando enviado
if ack:
    if ack.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
        result = ack.result
        if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print(" Comando RTL aceptado por el dron.")
        else:
            print(f" Comando RTL rechazado. Código de resultado: {result}")
    else:
        print(f" Se recibió ACK para un comando diferente: {ack.command}")
else:
    print(" No se recibió ninguna confirmación del comando.")

# Variables para monitoreo de enlace
last_heartbeat = time.time()

print("\nIniciando monitoreo del enlace de telemetría...")

try:
    while True:
        # Leer el siguiente mensaje disponible
        msg = master.recv_match(blocking=False)

        if msg:
            msg_type = msg.get_type()

            # Actualizar el tiempo del último latido recibido
            if msg_type == 'HEARTBEAT':
                last_heartbeat = time.time()

            # Procesar mensajes RADIO_STATUS para monitorear la calidad del enlace
            elif msg_type == 'RADIO_STATUS':
                rssi = msg.rssi  # Intensidad de señal local
                remrssi = msg.remrssi  # Intensidad de señal remota
                txbuf = msg.txbuf  # Porcentaje de búfer de transmisión libre
                noise = msg.noise  # Nivel de ruido local
                remnoise = msg.remnoise  # Nivel de ruido remoto
                rxerrors = msg.rxerrors  # Errores de recepción
                fixed = msg.fixed  # Paquetes corregidos

                print(f"[RADIO_STATUS] RSSI: {rssi}, RemRSSI: {remrssi}, TXBuf: {txbuf}%, "
                      f"Ruido: {noise}/{remnoise}, Errores RX: {rxerrors}, Corregidos: {fixed}")

        # Verificar si se ha perdido el latido del dron
        if time.time() - last_heartbeat > HEARTBEAT_TIMEOUT:
            print(" Latido del dron perdido. Intentando reconectar...")
            master.close()
            time.sleep(2)
            master = mavutil.mavlink_connection(PORT, baud=BAUD)
            master.wait_heartbeat()
            print(" Reconexión exitosa.")
            last_heartbeat = time.time()

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nMonitoreo interrumpido por el usuario.")