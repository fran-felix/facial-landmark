import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from typing import TypedDict

DATA = Path("data")
ANNOTATIONS = DATA / "WFLW_annotations/list_98pt_rect_attr_train_test"
IMAGES = DATA / "WFLW_images"
ESPELHO = DATA / "WFLW_annotations" / "Mirror98.txt"

TAMANHO = 256

REGIOES = {
"contorno": range(0, 33), "sobrancelhas": range(33, 51), "nariz": range(51, 60),
"olho direito": range(60, 68), "olho esquerdo": range(68, 76),
"boca": range(76, 96), "pupilas": range(96, 98),
} 

CORES = {
"contorno": "tab:blue", "sobrancelhas": "tab:brown", "nariz": "tab:green",
"olho direito": "red", "olho esquerdo": "deepskyblue", "boca": "violet",
"pupilas": "orange",
}

rng = np.random.default_rng(0)

class Face(TypedDict):
  landmarks: np.ndarray
  caixa: np.ndarray
  atributos: np.ndarray
  imagem: Path

class DatasetAnotacoesFaces(Dataset):
  def __init__(self, split="train", annotations=ANNOTATIONS, img_dir=IMAGES, transform=True):
    """transform: default 'True', applies data augmentation on images. Set to 'False' to get unmodified images through __getitem__"""
    self.annotations = Path(annotations)
    self.img_dir = Path(img_dir)
    self.tamanho_recorte = TAMANHO
    self.transform = transform

    # [0, 7499] train [7500, 9999] test
    # self.faces: list[Face] = (self.ler_anotacoes("train") + self.ler_anotacoes("test"))
    self.faces: list[Face] = self.ler_anotacoes(split)

  def ler_anotacoes(self, split: str) -> list[Face]:
    arquivo = self.annotations / f"list_98pt_rect_attr_{split}.txt"
    faces: list[Face] = []

    for linha in arquivo.read_text().splitlines():
      campos = linha.split()
      faces.append(
        {
          "landmarks": np.array(campos[:196], dtype=np.float32).reshape(98, 2),
          "caixa": np.array(campos[196:200], dtype=np.float32),
          "atributos": np.array(campos[200:206], dtype=int),
          "imagem": self.img_dir / campos[206],
        }
      )

    return faces

  def __len__(self):
    return len(self.faces)

  def ler_imagem_rgb(self, caminho):
    """Lê uma imagem com OpenCV e converte BGR para RGB."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
      raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

  def matriz_recorte(self, caixa, margem=1.25):
    """
    Matriz afim 2x3 que leva a imagem original ao recorte TAMANHO x TAMANHO.
    Implementa p' = s (p - c) + (TAMANHO/2, TAMANHO/2).
    """
    x0, y0, x1, y1 = caixa
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])  # centro da caixa
    lado = margem * max(x1 - x0, y1 - y0)         # lado do quadrado
    s = self.tamanho_recorte / lado               # escala
    t = self.tamanho_recorte / 2 - s * c          # translação

    return np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float32)

  def transformar_pontos(self, pts, M):
      """Aplica a matriz afim 2x3 M a pontos (K, 2) -- a mesma conta que o warpAffine faz."""
      return pts @ M[:, :2].T + M[:, 2]
  
  def recortar(self, face, M=None):
    """Recorte RGB TAMANHO x TAMANHO da face e seus landmarks no sistema do recorte."""

    if M is None:
      M = self.matriz_recorte(face["caixa"])
    recorte = cv2.warpAffine(self.ler_imagem_rgb(face["imagem"]), M, (self.tamanho_recorte, self.tamanho_recorte))

    return recorte, self.transformar_pontos(face["landmarks"], M)

  def para_3x3(self, M):
    return np.vstack([M, [0, 0, 1]])

  def matriz_aumentada(self, caixa, angulo=0.0, escala=1.0, deslocamento=(0.0, 0.0)):
    """
    Recorte + rotação (graus) + escala + deslocamento (fração do recorte) em uma só matriz.
    escala > 1 aumenta o lado do recorte (afasta a face); escala < 1 aproxima.
    """
    R = cv2.getRotationMatrix2D((TAMANHO / 2, TAMANHO / 2), angulo, 1 / escala)
    R[:, 2] += np.array(deslocamento) * TAMANHO
    return (self.para_3x3(R) @ self.para_3x3(self.matriz_recorte(caixa)))[:2].astype(np.float32)

  def espelhar(self, recorte, pts):
    espelho = np.array(ESPELHO.read_text().strip().split(","), dtype=int)

    # A inversão cria um array com stride negativo; o PyTorch não aceita isso.
    recorte_esp = recorte[:, ::-1].copy()          # espelha e cria memória contígua
    pts_esp = pts.copy()
    pts_esp[:, 0] = TAMANHO - 1 - pts[:, 0]        # espelha só a coordenada x
    pts_certo = pts_esp[espelho]                   # ... e troca cada ponto pelo seu par

    return recorte_esp, pts_certo

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.item()
    if idx is None:
      raise RuntimeError(f"Index inválido, não foi possível acessar a posição: {idx}")

    face = self.faces[int(idx)]

    # Augmentation
    m = None
    if self.transform:
      params = {"angulo": rng.uniform(-15, 15), "escala": rng.uniform(0.85, 1.2), "deslocamento": rng.uniform(-0.06, 0.06, size=2)}
      m = self.matriz_aumentada(face["caixa"], **params)
    imagem, landmarks = self.recortar(face, m)

    if self.transform and rng.random() < 0.5:
      imagem, landmarks = self.espelhar(imagem, landmarks)

    sample = {
      "landmarks": landmarks,
      "caixa": face["caixa"],
      "atributos": face["atributos"],
      "imagem": imagem, # Retornando apenas a FACE, não a imagem inteira
    }
    
    return sample