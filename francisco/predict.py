import numpy as np


class LandmarkModel:
  
  def __init__(self, weights: str, device: str = "cpu"):
    """
    Carrega o detector de faces e o modelo de landmarks.
    weights: caminho dos pesos (pesos/modelo.pt);
    device: "cpu" ou "cuda".
    """
    ...

  def predict(self, image: np.ndarray) -> np.ndarray:
    """
    Detecta as faces da imagem e prevê os 98 landmarks de cada uma.
    image: imagem inteira, RGB, uint8, formato (altura, largura, 3)
    retorna: array float (N, 98, 2), uma linha por face detectada,
    com (x, y) em pixels da imagem inteira, na ordem dos
    landmarks do WFLW; N = 0 se nenhuma face for encontrada
    """
    ...