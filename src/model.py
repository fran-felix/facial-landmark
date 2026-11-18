import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset import DatasetAnotacoesFaces, TAMANHO, rng
import time
import matplotlib.pyplot as plt

MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # normalização da ImageNet
DESVIO = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dispositivo.type == "cuda":
  N_TREINO, EPOCAS, LOTE, LR, PASSOS_SANIDADE = 2000, 10, 32, 3e-4, 100
else:
  # sem GPU: treino bem menor (e resultado bem pior), para levar poucos minutos;
  # lotes menores e lr maior compensam um pouco o número pequeno de passos
  N_TREINO, EPOCAS, LOTE, LR, PASSOS_SANIDADE = 400, 4, 16, 1e-3, 40
N_VAL = 300

PASSO = 4                       # pixels do recorte por célula do mapa
LADO_MAPA = TAMANHO // PASSO    # 64

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


class CabecaFaceNet(nn.Module):

  def __init__(self):
    super().__init__()
    # 8x8x512 -> 16x16x256
    self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(256)
    # 16x16x256 -> 32x32x128
    self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    # 32x32x128 -> 64x64x64
    self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.last_conv = nn.Conv2d(64, 98, 1)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.last_conv(x)

    return x # Retorna os 98 heatmaps previstos


class FaceNet(nn.Module):

  def __init__(self):
    super().__init__()
    resnet = torchvision.models.resnet18(weights=pesos_imagenet)
    self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    self.cabeca = CabecaFaceNet()

  def forward(self, x):
    return self.soft_argmax_2d(self.cabeca(self.backbone(x))) # Retorna as coodernadas previstas para cada ponto

  # Gerado por IA v
  def soft_argmax_2d(self, logits):
    # logits: [batch, pontos, altura, largura]
    batch, pontos, altura, largura = logits.shape

    probabilidades = torch.softmax(logits.flatten(start_dim=2), dim=2).view(batch, pontos, altura, largura)

    xs = torch.linspace(0, 1, largura, device=logits.device, dtype=logits.dtype).view(1, 1, 1, largura)

    ys = torch.linspace(0, 1, altura, device=logits.device, dtype=logits.dtype).view(1, 1, altura, 1)

    coordenada_x = (probabilidades * xs).sum(dim=(2, 3))
    coordenada_y = (probabilidades * ys).sum(dim=(2, 3))

    return torch.stack((coordenada_x, coordenada_y), dim=-1)

def contar(parametros):
  return sum(p.numel() for p in parametros)

def para_tensor(recortes):
  """(N, 256, 256, 3) uint8 RGB -> tensor (N, 3, 256, 256) normalizado, no dispositivo."""
  if isinstance(recortes, np.ndarray):
    recortes = torch.from_numpy(recortes)
  
  x = recortes.to(dispositivo).permute(0, 3, 1, 2).float() / 255
  return (x - MEDIA.to(dispositivo)) / DESVIO.to(dispositivo)

# ----- Cálculo do Erro -----
def erros_por_ponto(pred, gt):
  """
  Erro de cada landmark / distância entre os cantos externos dos olhos (60 e 72).
  Aceita (98, 2) para uma face ou (N, 98, 2) para várias.
  """
  d = np.linalg.norm(gt[..., 60, :] - gt[..., 72, :], axis=-1)
  return np.linalg.norm(pred - gt, axis=-1) / d[..., None]

def nme(pred, gt):
  """NME de cada face: média dos erros normalizados dos 98 pontos."""
  return erros_por_ponto(pred, gt).mean(-1)

# ----- EAR -----
PONTOS_EAR = {"direito": [60, 61, 63, 64, 65, 67],    # p1..p6 no olho direito da pessoa
              "esquerdo": [72, 71, 69, 68, 75, 73]}   # equivalente, por simetria

def ear(pts, olho):
    """EAR de um olho. pts: (98, 2) ou (N, 98, 2)."""
    p1, p2, p3, p4, p5, p6 = (pts[..., i, :] for i in PONTOS_EAR[olho])
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

def ear_medio(pts):
    """Média dos dois olhos (reduz o ruído)."""
    return (ear(pts, "direito") + ear(pts, "esquerdo")) / 2

# ----- Heatmaps -----
def heatmap(ponto, sigma=2.0):
  """Gaussiana LADO_MAPA x LADO_MAPA centrada no ponto (dado em pixels do recorte)."""
  x, y = np.asarray(ponto) / PASSO            # posição em células do mapa
  ys, xs = np.mgrid[0:LADO_MAPA, 0:LADO_MAPA]
  return np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))

def soft_argmax_2d(logits):
  p = np.exp(logits - logits.max())
  p /= p.sum()                                     # softmax espacial: soma 1
  ys, xs = np.mgrid[0:LADO_MAPA, 0:LADO_MAPA]
  return np.array([(xs * p).sum(), (ys * p).sum()]) * PASSO  # valor esperado


# ----- Predição X Ground Truth -----
def avaliar(modelo, loader, tamanho):
  modelo.eval()
  nmes = []

  with torch.no_grad():
    for lote in loader:
      x = para_tensor(lote["imagem"])
      pred = (modelo(x)).cpu().numpy() 
      pred = pred * tamanho # Precisamos dos pixels, não dos valores normalizados
      gt = lote["landmarks"].cpu().numpy() # Tambem não normalizado, diferente do treino
      nmes.append(nme(pred, gt))

  return np.concatenate(nmes).mean()


# ----- TREINO -----
def treinar(modelo, max_epocas, batch_size, lr, tamanho, completo=True):
  torch.manual_seed(0)

  modelo.to(dispositivo)
  otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr)
  agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, max_epocas)  # lr cai até 0

  dataset_treino = DatasetAnotacoesFaces("train")
  dataset_val = DatasetAnotacoesFaces("test")
  ordem = rng

  if completo:
    dataset_treino.faces = [dataset_treino.faces[i] for i in ordem.permutation(len(dataset_val))]
    dataset_val.faces = [dataset_val.faces[i] for i in ordem.permutation(len(dataset_val))]  # validação do TREINO
  else:
    idx_treino = ordem.permutation(len(dataset_treino))[:N_TREINO]
    idx_val = ordem.permutation(len(dataset_val))[:N_VAL]
    dataset_treino = Subset(dataset_treino, idx_treino.tolist())
    dataset_val = Subset(dataset_val, idx_val.tolist())

  loader_treino = DataLoader(
    dataset=dataset_treino,
    batch_size=batch_size,
    shuffle=True,
  )
  loader_val = DataLoader(
    dataset=dataset_val,
    batch_size=batch_size,
  )

  historico = []

  for epoca in range(1, max_epocas + 1):
    inicio = time.time()

    modelo.train()
    perdas = []
    
    for lote in loader_treino:
      x = para_tensor(lote["imagem"])
      y = lote["landmarks"].to(dispositivo).float() / tamanho

      otimizador.zero_grad()
      perda = (modelo(x) - y).abs().mean()
      perda.backward()
      otimizador.step()

      perdas.append(perda.item())
    
    agendador.step()
    nme_val = avaliar(modelo, loader_val, tamanho)
    
    historico.append((np.mean(perdas), nme_val))
    print(f"época {epoca:2d} | perda {np.mean(perdas):.4f} | NME validação {100 * nme_val:5.2f}% |"
          f"tempo: {time.time() - inicio:.0f} s")

# modelo = ModeloTeste()
"""
face_model = FaceNet()
for p in face_model.backbone.parameters(): p.requires_grad = False
print(f"backbone:   {contar(face_model.backbone.parameters()):>10,} parâmetros (pré-treinados, ImageNet)")
print(f"cabeça:     {contar(face_model.cabeca.parameters()):>10,} parâmetros (aleatórios, treinados no WFLW)")
print(f"treináveis: {contar(p for p in face_model.parameters() if p.requires_grad):>10,}")

#print(face_model)

#treinar(face_model, EPOCAS, LOTE, LR, TAMANHO, completo=False)

#torch.save(modelo.cabeca.state_dict(), "/home/francisco/trabalho_final_TPI/weights/cabeca.pt")
#torch.save(face_model.cabeca.state_dict(), "/home/francisco/trabalho_final_TPI/weights/modelo.pt")
"""