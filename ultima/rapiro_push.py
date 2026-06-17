"""
RAPIRO — Envia frames por WebSocket al EC2 + Colores y Movimientos
--------------------------------------------------------------------
Instalar en el Pi:
  sudo apt install -y mpg123 python3-serial python3-websockets python3-opencv

Uso:
  python3 rapiro_push.py --ec2 IP_PUBLICA_DEL_EC2
  python3 rapiro_push.py --ec2 54.207.39.101 --serial /dev/ttyS0
"""

import asyncio
import argparse
import json
import os
import tempfile
import time
import cv2
import websockets

# Serial es opcional: si no esta instalado o no hay Rapiro fisico, sigue funcionando
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def abrir_serial(puerto):
    """Intenta abrir el puerto serie del Rapiro. Retorna None si falla."""
    if not SERIAL_AVAILABLE:
        print("[INFO] python3-serial no instalado, sin control de hardware")
        return None
    try:
        robot = serial.Serial(puerto, 57600, timeout=1)
        time.sleep(0.5)  # esperar a que el puerto se estabilice
        print(f"[INFO] Puerto serie conectado: {puerto}")
        return robot
    except Exception as e:
        print(f"[WARNING] No se pudo conectar al Rapiro ({puerto}): {e}")
        return None


def cmd(robot, comando):
    """Envia un comando serial al Rapiro con manejo de errores."""
    if robot is None:
        return
    try:
        robot.write(comando.encode() + b"\r")
        time.sleep(0.05)  # 50ms entre comandos para que el firmware los procese
    except Exception as e:
        print(f"[WARNING] Error serial: {e}")


async def main(ec2_ip, serial_port):
    url = f"ws://{ec2_ip}:8000/ws/frame"
    print(f"{'='*50}")
    print(f"  RAPIRO LSA - Streaming al EC2")
    print(f"  Conectando a: {url}")
    print(f"{'='*50}")

    # Inicializar Rapiro
    robot = abrir_serial(serial_port)
    cmd(robot, "#M0")                    # posicion inicial
    cmd(robot, "#PR000G000B255T005")     # ojos azules = esperando

    # Inicializar camara
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] No se encontro camara")
        return

    print("[INFO] Camara lista")

    while True:
        try:
            async with websockets.connect(url, max_size=5_000_000) as ws:
                print("[INFO] Conectado al EC2!\n")
                esperando_audio = False

                # Ojos verdes = conectado a la nube
                cmd(robot, "#PR000G255B000T003")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.5)
                        continue

                    # Comprimir y enviar
                    _, buffer = cv2.imencode(".jpg", frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, 55])
                    await ws.send(buffer.tobytes())

                    # Recibir respuesta
                    try:
                        respuesta = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue

                    # JSON = datos de letra
                    if isinstance(respuesta, str):
                        data = json.loads(respuesta)
                        letra      = data.get("letra", "---")
                        confianza  = data.get("confianza", 0)
                        texto      = data.get("texto", "")
                        confirmada = data.get("confirmada", False)
                        progreso   = data.get("progreso", 0)
                        manos      = data.get("manos", 0)

                        barra = "#" * int(progreso * 20) + "-" * (20 - int(progreso * 20))

                        if confirmada:
                            print(f"  OK {letra}  ->  \"{texto}\"")
                            if letra == "FINALIZAR":
                                esperando_audio = True
                                # Ojos amarillos + movimiento mientras habla
                                cmd(robot, "#PR255G255B000T001")
                                cmd(robot, "#M5")
                        else:
                            print(f"  [{barra}] {letra} {confianza*100:.0f}%  "
                                  f"Manos:{manos}  Texto: {texto}    ", end="\r")

                    # Binario = audio MP3
                    elif isinstance(respuesta, bytes) and esperando_audio:
                        print(f"\n  Reproduciendo audio...")
                        esperando_audio = False

                        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                        try:
                            f.write(respuesta)
                            f.close()

                            proc = await asyncio.create_subprocess_exec(
                                "mpg123", "-q", f.name,
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL
                            )
                            await proc.wait()
                        finally:
                            if os.path.exists(f.name):
                                os.unlink(f.name)

                            # Volver a posicion base + ojos verdes
                            cmd(robot, "#M0")
                            cmd(robot, "#PR000G255B000T003")

                    await asyncio.sleep(0.05)

        except (websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError, OSError) as e:
            print(f"\n[INFO] Conexion perdida: {e}")
            # Ojos rojos = sin conexion
            cmd(robot, "#PR255G000B000T005")
            print("[INFO] Reintentando en 3 segundos...")
            await asyncio.sleep(3)

    cap.release()
    if robot:
        cmd(robot, "#M0")
        cmd(robot, "#PR000G000B000T003")
        robot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec2", required=True, help="IP publica del EC2")
    parser.add_argument("--serial", default="/dev/ttyAMA0",
                        help="Puerto serie del Rapiro (default: /dev/ttyAMA0)")
    args = parser.parse_args()
    asyncio.run(main(args.ec2, args.serial))