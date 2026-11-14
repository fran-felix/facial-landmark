import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset import DatasetAnotacoesFaces


# Definindo o backbone do modelo
pesos_imagenet = torchvision.models.ResNet18_Weights.DEFAULT


class ModeloTeste(nn.Module):
  """
  Exemplo mínimo: ResNet-18 + pooling global + uma camada linear -> 98 pontos.
  É a opção mais simples e a menos precisa. Serve só para entender o treino.
  """

  def __init__(self):
    super().__init__()
    resnet = torchvision.models.resnet18(weights=pesos_imagenet)
    self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    self.cabeca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 98 * 2))

  def forward(self, x):
    # (B, 3, 256, 256) -> (B, 98, 2), coordenadas em fração do recorte
    return self.cabeca(self.backbone(x)).view(-1, 98, 2)


def contar(parametros):
  return sum(p.numel() for p in parametros)


modelo = ModeloTeste()
print(f"backbone:   {contar(modelo.backbone.parameters()):>10,} parâmetros (pré-treinados, ImageNet)")
print(f"cabeça:     {contar(modelo.cabeca.parameters()):>10,} parâmetros (aleatórios, treinados no WFLW)")
print(f"treináveis: {contar(p for p in modelo.parameters() if p.requires_grad):>10,}")


def erros_por_ponto(pred, gt):
  """Erro de cada landmark / distância entre os cantos externos dos olhos (60 e 72).

  Aceita (98, 2) para uma face ou (N, 98, 2) para várias.
  """
  d = np.linalg.norm(gt[..., 60, :] - gt[..., 72, :], axis=-1)
  return np.linalg.norm(pred - gt, axis=-1) / d[..., None]


def nme(pred, gt):
  """NME de cada face: média dos erros normalizados dos 98 pontos."""
  return erros_por_ponto(pred, gt).mean(-1)


MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # normalização da ImageNet
DESVIO = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dispositivo.type == "cuda":
  N_TREINO, EPOCAS, LOTE, LR, PASSOS_SANIDADE = 2000, 10, 32, 3e-4, 100
else:  # sem GPU: treino bem menor (e resultado bem pior), para levar poucos minutos;
  # lotes menores e lr maior compensam um pouco o número pequeno de passos
  N_TREINO, EPOCAS, LOTE, LR, PASSOS_SANIDADE = 400, 4, 16, 1e-3, 40
N_VAL = 300


def para_tensor(recortes):
  """(N, 256, 256, 3) uint8 RGB -> tensor (N, 3, 256, 256) normalizado, no dispositivo."""
  if isinstance(recortes, np.ndarray):
    recortes = torch.from_numpy(recortes)
  x = recortes.to(dispositivo).permute(0, 3, 1, 2).float() / 255
  return (x - MEDIA.to(dispositivo)) / DESVIO.to(dispositivo)

# ----- TESTE DE TREINO -----
torch.manual_seed(0)
modelo = ModeloTeste().to(dispositivo)
otimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
dataset = DatasetAnotacoesFaces()

faces = dataset.faces
ordem = np.random.default_rng(0).permutation(len(faces))
faces_treino = [faces[i] for i in ordem[:N_TREINO]]
faces_val = [faces[i] for i in ordem[N_TREINO:N_TREINO + N_VAL]]  # validação: do TREINO

loader_treino = DataLoader(
  Subset(dataset, ordem[:N_TREINO].tolist()),
  batch_size=16,
  shuffle=True,
)

# Um lote fixo permite verificar se o modelo consegue decorar poucos exemplos.
lote = next(iter(loader_treino))
x = para_tensor(lote["imagem"])
y = lote["landmarks"].to(dispositivo).float() / 256

print(x.shape)
print(y.shape)

for p in modelo.backbone.parameters(): p.requires_grad = False

for passo in range(1, PASSOS_SANIDADE + 1):
  modelo.train()
  perda = (modelo(x) - y).abs().mean()  # L1 nas coordenadas
  otimizador.zero_grad()
  perda.backward()
  otimizador.step()
  if passo % 20 == 0:
      print(f"passo {passo:3d} | perda {perda.item():.4f}")
print(f"perda final: {perda.item():.4f}")
