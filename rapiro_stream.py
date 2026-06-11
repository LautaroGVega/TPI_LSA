"""
RAPIRO — Streaming de video por WiFi (super liviano)
------------------------------------------------------
Solo captura y transmite. No corre MediaPipe ni modelo.
Usa ~80MB de RAM.

Instalar en el Pi:
  pip install flask opencv-python-headless

Uso:
  python rapiro_stream.py

El stream queda disponible en:
  http://<IP_DEL_PI>:8080/video
"""

from flask import Flask, Response
import cv2

app = Flask(__name__)


def generar_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Comprimir a JPEG (calidad 70 = buen balance tamaño/calidad)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video")
def video():
    return Response(generar_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return """
    <html><body style="background:#111;color:#eee;text-align:center;font-family:sans-serif">
    <h2>RAPIRO LSA — Camera Stream</h2>
    <img src="/video" style="max-width:100%;border-radius:12px">
    <p>Stream activo en /video</p>
    </body></html>
    """


if __name__ == "__main__":
    print("=" * 50)
    print("  RAPIRO — Stream de video")
    print("  Ver en: http://<IP_DEL_PI>:8080")
    print("  Stream: http://<IP_DEL_PI>:8080/video")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8080, threaded=True)