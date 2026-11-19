import numpy as np
import torch

from model import FaceNet
from train import MEDIA, DESVIO


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

    entrada = self.para_tensor(np.expand_dims(image, axis=0))
    with torch.no_grad():
      predicao = self.model(entrada)

    return predicao.detach().cpu().numpy()