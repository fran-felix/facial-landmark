import numpy as np
import torch
import cv2

from model import FaceNet
from train import MEDIA, DESVIO
from yunet import YUNET
from dataset import TAMANHO

if YUNET.exists():  # baixado na célula de preparação
  detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold=0.3, nms_threshold=0.3)
else:
  Exception("YuNet não encontrado.")

class LandmarkModel:
  
  def __init__(self, weights: str, device: str = "cpu"):
    """
    Carrega o detector de faces e o modelo de landmarks.
    weights: caminho dos pesos (pesos/modelo.pt);
    device: "cpu" ou "cuda".
    """
    self.device = torch.device(device)
    self.model = FaceNet()
    self.model.cabeca.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
    self.model.to(self.device)
    self.model.eval()

  def para_tensor(self, images):
    """(N, 256, 256, 3) uint8 RGB -> tensor (N, 3, 256, 256) normalizado, no dispositivo."""
    if isinstance(images, np.ndarray):
      images = torch.from_numpy(images)

    if images.ndim == 3:
        images = images.unsqueeze(0)  # HWC -> NHWC
    if images.ndim != 4:
        raise ValueError(f"Formato inesperado: {tuple(images.shape)}")

    x = images.to(self.device).permute(0, 3, 1, 2).float() / 255
    return (x - MEDIA.to(self.device)) / DESVIO.to(self.device)

  def predict(self, image: np.ndarray) -> np.ndarray:
    """
    Detecta as faces da imagem e prevê os 98 landmarks de cada uma.
    image: imagem inteira, RGB, uint8, formato (altura, largura, 3)
    retorna: array float (N, 98, 2), uma linha por face detectada,
    com (x, y) em pixels da imagem inteira, na ordem dos
    landmarks do WFLW; N = 0 se nenhuma face for encontrada
    """
    caixas = detectar_faces(image)
    previsoes = []

    for caixa in caixas:
      M = matriz_recorte(caixa)
      recorte = cv2.warpAffine(image, M, (TAMANHO, TAMANHO))
      entrada = self.para_tensor(recorte)
      with torch.no_grad():
        predicao = self.model(entrada).cpu().numpy() * TAMANHO
      previsoes.append(transformar_pontos(predicao, cv2.invertAffineTransform(M)))
    
    return np.array(previsoes, dtype=np.float32).reshape(-1, 98, 2)


def detectar_faces(imagem_rgb):
    """Caixas (x0, y0, x1, y1) das faces encontradas, da maior para a menor."""
    
    bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)  # o YuNet espera BGR
    altura, largura = bgr.shape[:2]
    detector.setInputSize((largura, altura))           # precisa ser o tamanho da imagem
    _, faces = detector.detect(bgr)
    
    if faces is None:                                  # None quando não há nenhuma face
        return []
    
    caixas = [np.array([x, y, x + w, y + h]) for x, y, w, h in faces[:, :4]]

    return sorted(caixas, key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))

def transformar_pontos(pts, M):
    """Aplica a matriz afim 2x3 M a pontos (K, 2) -- a mesma conta que o warpAffine faz."""
    return pts @ M[:, :2].T + M[:, 2]

def matriz_recorte(caixa, margem=1.25):
  """
  Matriz afim 2x3 que leva a imagem original ao recorte TAMANHO x TAMANHO.
  Implementa p' = s (p - c) + (TAMANHO/2, TAMANHO/2).
  """
  caixa = np.asarray(caixa)
  if caixa.ndim == 1:
    if caixa.size < 4:
      raise ValueError(f"Caixa inválida: shape={caixa.shape}")
    caixa = caixa[:4]
  else:
    caixa = caixa.reshape(-1, caixa.shape[-1])[0, :4]

  x0, y0, x1, y1 = map(float, caixa)
  c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])  # centro da caixa
  lado = margem * max(x1 - x0, y1 - y0)         # lado do quadrado
  s = TAMANHO / lado               # escala
  t = TAMANHO / 2 - s * c          # translação

  return np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float32)