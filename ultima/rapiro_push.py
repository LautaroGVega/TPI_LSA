"""
RAPIRO — Envía frames por WebSocket al EC2 (versión optimizada)
-----------------------------------------------------------------
Fixes aplicados:
  - asyncio.create_subprocess_exec en vez de subprocess.run
  - os.unlink para limpiar archivos temporales
  - CAP_PROP_BUFFERSIZE = 1 para eliminar lag del buffer

Instalar en el Pi:
  sudo apt install -y mpg123
  pip install websockets opencv-python-headless --break-system-packages

Uso:
  python3 rapiro_push.py --ec2 IP_PUBLICA_DEL_EC2
"""

import asyncio
import argparse
import json
import os
import tempfile
import cv2
import websockets


async def main(ec2_ip):
    url = f"ws://{ec2_ip}:8000/ws/frame"
    print(f"{'='*50}")
    print(f"  RAPIRO LSA - Streaming al EC2")
    print(f"  Conectando a: {url}")
    print(f"{'='*50}")

    cap = cv2.VideoCapture(0) #cv2.VideoCapture(0) en ws pero en linuix usar cv2.CAP_V4L2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Fix: elimina lag del buffer

    if not cap.isOpened():
        print("[ERROR] No se encontro camara")
        return

    print("[INFO] Camara lista")

    while True:
        try:
            async with websockets.connect(url, max_size=5_000_000) as ws:
                print("[INFO] Conectado al EC2!\n")
                esperando_audio = False

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

                        barra = "█" * int(progreso * 20) + "░" * (20 - int(progreso * 20))

                        if confirmada:
                            print(f"  ✓ {letra}  ->  \"{texto}\"")
                            if letra == "FINALIZAR":
                                esperando_audio = True
                        else:
                            print(f"  [{barra}] {letra} {confianza*100:.0f}%  "
                                  f"Manos:{manos}  Texto: {texto}    ", end="\r")

                    # Binario = audio MP3
                    elif isinstance(respuesta, bytes) and esperando_audio:
                        print(f"\n  ♫ Reproduciendo audio...")
                        esperando_audio = False

                        # Fix: archivo temporal con limpieza
                        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                        try:
                            f.write(respuesta)
                            f.close()

                            # Fix: subprocess async, no bloquea el event loop
                            proc = await asyncio.create_subprocess_exec(
                                "mpg123", "-q", f.name,
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL
                            )
                            await proc.wait()
                        finally:
                            # Fix: limpieza del archivo temporal
                            if os.path.exists(f.name):
                                os.unlink(f.name)

                    await asyncio.sleep(0.05)

        except (websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError, OSError) as e:
            print(f"\n[INFO] Conexion perdida: {e}")
            print("[INFO] Reintentando en 3 segundos...")
            await asyncio.sleep(3)

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec2", required=True, help="IP publica del EC2")
    args = parser.parse_args()
    asyncio.run(main(args.ec2))