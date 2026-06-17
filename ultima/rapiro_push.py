"""
RAPIRO — Envia frames por WebSocket al EC2 + Colores y Movimientos (Async Fix)
--------------------------------------------------------------------
Instalar en el Pi:
  sudo apt install -y mpg123 python3-serial python3-websockets python3-opencv

Uso:
  python3 rapiro_push.py --ec2 IP_PUBLICA_DEL_EC2
"""

import asyncio
import argparse
import json
import os
import signal
import tempfile
import cv2
import websockets

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def abrir_serial(puerto):
    if not SERIAL_AVAILABLE:
        print("[INFO] python3-serial no instalado, sin control de hardware")
        return None
    try:
        robot = serial.Serial(puerto, 57600, timeout=1)
        import time
        time.sleep(0.5)
        print(f"[INFO] Puerto serie conectado: {puerto}")
        return robot
    except Exception as e:
        print(f"[WARNING] No se pudo conectar al Rapiro ({puerto}): {e}")
        return None


async def cmd(robot, comando):
    """Envia un comando serial sin bloquear el event loop."""
    if robot is None:
        return
    try:
        robot.write(comando.encode() + b"\r")
        await asyncio.sleep(0.06)
    except Exception as e:
        print(f"[WARNING] Error serial: {e}")


async def resetear_rapiro(robot):
    """Lleva al Rapiro a posicion inicial con brazos abajo y ojos verdes."""
    if robot is None:
        return
    print("[INFO] Retornando a posicion base...")
    await cmd(robot, "#M0")
    await asyncio.sleep(0.3)
    await cmd(robot, "#M0")
    await asyncio.sleep(0.1)
    await cmd(robot, "#PR000G255B000T003")
    print("[INFO] Rapiro en estado base con ojos verdes.")


async def main(ec2_ip, serial_port):
    url = f"ws://{ec2_ip}:8000/ws/frame"
    print(f"{'='*50}")
    print(f"  RAPIRO LSA - Streaming al EC2")
    print(f"  Conectando a: {url}")
    print(f"{'='*50}")

    robot = abrir_serial(serial_port)
    await resetear_rapiro(robot)

    def al_cerrar(sig, frame):
        print("\n[INFO] Cerrando...")
        if robot:
            robot.write(b"#M0\r")
            robot.write(b"#PR000G000B255T001\r")
            robot.close()
        exit(0)
    signal.signal(signal.SIGINT, al_cerrar)
    signal.signal(signal.SIGTERM, al_cerrar)

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

                await cmd(robot, "#PR000G255B000T003")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.5)
                        continue

                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    await ws.send(buffer.tobytes())

                    try:
                        respuesta = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue

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
                                await cmd(robot, "#PR000G255B000T001")
                                await cmd(robot, "#M5")
                        else:
                            print(f"  [{barra}] {letra} {confianza*100:.0f}%  "
                                  f"Manos:{manos}  Texto: {texto}    ", end="\r")

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

                        print("[INFO] Audio terminado. Esperando transicion...")
                        await asyncio.sleep(2.0)
                        await resetear_rapiro(robot)

                    await asyncio.sleep(0.05)

        except (websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError, OSError) as e:
            print(f"\n[INFO] Conexion perdida: {e}")
            await cmd(robot, "#PR255G000B000T005")
            print("[INFO] Reintentando en 3 segundos...")
            await asyncio.sleep(3)

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec2", required=True, help="IP publica del EC2")
    parser.add_argument("--serial", default="/dev/ttyAMA0",
                        help="Puerto serie del Rapiro (default: /dev/ttyAMA0)")
    args = parser.parse_args()
    asyncio.run(main(args.ec2, args.serial))