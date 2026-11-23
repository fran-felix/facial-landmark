from pathlib import Path
from urllib.request import urlretrieve


YUNET = Path("modelos/face_detection_yunet_2023mar.onnx")
if not YUNET.exists():
    YUNET.parent.mkdir(exist_ok=True)
    try:
        urlretrieve("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx", YUNET)
    except OSError as erro:
        print("Não foi possível baixar o YuNet; a seção 10 será pulada.", erro)