import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import torch

from torch.utils.data import DataLoader
from dataset import DatasetAnotacoesFaces, TAMANHO, REGIOES, CORES, rng
from train import para_tensor, TAMANHO, nme, PASSO, LADO_MAPA, heatmap, avaliar
from model import FaceNet


def desenhar_caixa(ax, caixa, cor, **kwargs):
  x0, y0, x1, y1 = caixa
  ax.add_patch(pat.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cor, **kwargs))

def desenhar_landmarks(ax, pts, s=10, legenda=False):
  """Desenha os 98 pontos, com uma cor por região."""

  for nome, indices in REGIOES.items():
    ax.scatter(*pts[list(indices)].T, s=s, c=CORES[nome], label=nome)
  if legenda:
    ax.legend(fontsize=8, loc="lower right", markerscale=1.5)

def buscar(faces, nome_arquivo):
  """Maior face anotada na imagem com esse nome de arquivo."""
  candidatas = [f for f in faces if f["imagem"].name == nome_arquivo]
  return max(candidatas, key=lambda f: f["caixa"][2] - f["caixa"][0])

def test1(dataset):
  """Testando o __getitem__ implementado"""

  print(len(dataset))
  for i in range(5):
    sample = dataset[i]
    print(i, sample['landmarks'].shape, sample['caixa'].shape, sample['atributos'].shape, sample['imagem'].shape)

# Inversão. Nota para o futuro
def test3(dataset, exemplo, M, pts_recorte):
  M_inversa = cv2.invertAffineTransform(M)
  de_volta = dataset.transformar_pontos(pts_recorte, M_inversa)
  print(f"erro máximo na ida e volta: {np.abs(de_volta - exemplo['landmarks']).max():.6f} pixels")

def test2(dataset):
  """Testando se as funções de recorte e manipulação (adaptadas) funcionam na estrutura de dados montada"""

  exemplo = next(f for f in dataset.faces if np.allclose(f["caixa"], [692, 90, 854, 326]))
  M = dataset.matriz_recorte(exemplo["caixa"])
  recorte, pts_recorte = dataset.recortar(exemplo, M)

  print("matriz M:\n", M.round(3))
  print("landmark 60 na imagem original:", exemplo["landmarks"][60].round(1))
  print("landmark 60 no recorte:        ", pts_recorte[60].round(1))

  imagem = dataset.ler_imagem_rgb(exemplo["imagem"])
  x0, y0, x1, y1 = exemplo["caixa"]
  c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
  lado = 1.25 * max(x1 - x0, y1 - y0)

  fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
  axs[0].imshow(imagem)
  desenhar_caixa(axs[0], exemplo["caixa"], "lime", lw=2)
  desenhar_caixa(axs[0], [*(c - lado / 2), *(c + lado / 2)], "yellow", lw=2, ls="--")
  axs[0].set_title(f"(a) original {imagem.shape[1]}x{imagem.shape[0]}\n"
                  "verde: caixa anotada | amarelo: quadrado com margem de 25%")
  axs[1].imshow(recorte)
  desenhar_landmarks(axs[1], pts_recorte, s=8)
  axs[1].set_title("(b) recorte 256x256\nlandmarks transformados com a mesma M")
  plt.show()

  test3(dataset, exemplo, M, pts_recorte)

def test4(modelo, dataset):
  """Visualizando previsões do modelo e comparando com os landmarks corretos"""

  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  modelo.to(dispositivo)
  modelo.eval()

  fig, axs = plt.subplots(3, 4, figsize=(14, 7.2))
  for ax, sample in zip(axs.flat, dataset):
    img = sample["imagem"]
    entrada = para_tensor(np.expand_dims(img, axis=0))

    with torch.no_grad():
      pred = modelo(entrada).cpu().numpy()[0] * TAMANHO
      gt = sample["landmarks"]

    ax.imshow(img)
    ax.scatter(*gt.T, s=12, c="lime", label="verdade")
    ax.scatter(*pred.T, s=12, c="red", marker="x", label="previsto")
    ax.set_title(f"Landmarks - NME: {100 * nme(pred, gt):.2f}%")
    ax.legend(loc="lower right")
    ax.axis("off")
  fig.suptitle("Previsões para rostos.")
  plt.tight_layout()
  plt.show()

def test5():
  dataset_treino = DatasetAnotacoesFaces("train")
  amostra = dataset_treino[1]
  recorte = amostra["imagem"]
  pts = amostra["landmarks"]
  mapas = np.stack([heatmap(p) for p in pts])  # (98, 64, 64): um mapa por landmark
  # extent: posiciona o mapa 64x64 sobre o recorte 256x256 (cada célula cobre 4x4 pixels)
  extent = (-PASSO / 2, TAMANHO - PASSO / 2, TAMANHO - PASSO / 2, -PASSO / 2)

  fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
  axs[0].imshow(recorte)
  axs[0].scatter(*pts[60], s=40, c="red")
  axs[0].set_title("(a) recorte, landmark 60")
  axs[1].imshow(mapas[60], cmap="hot")
  axs[1].set_title("(b) heatmap do 60 (64x64, σ = 2)")
  axs[2].imshow(recorte)
  axs[2].imshow(mapas.max(0), cmap="hot", alpha=0.6, extent=extent)
  axs[2].set_title("(c) máximo dos 98 heatmaps")

  axs[3].imshow(recorte)
  centros = np.arange(LADO_MAPA) * PASSO  # posição de cada célula do mapa, em pixels do recorte
  for i, cor in [(62, "red"), (66, "deepskyblue")]:
    axs[3].contour(centros, centros, mapas[i], levels=[0.25, 0.5, 0.75], colors=cor, linewidths=1.2)
    axs[3].scatter(*pts[i], s=40, c=cor, edgecolors="white", zorder=3)
  axs[3].scatter(*pts[[60, 61, 63, 64, 65, 67]].T, s=12, c="yellow", zorder=3)
  axs[3].set_xticks(np.arange(-PASSO / 2, TAMANHO, PASSO), minor=True)
  axs[3].set_yticks(np.arange(-PASSO / 2, TAMANHO, PASSO), minor=True)
  axs[3].grid(which="minor", color="white", lw=0.4, alpha=0.5)
  #aproximar(axs[3], pts[60:68], margem=0.25)
  distancia = np.linalg.norm(pts[62] - pts[66])
  axs[3].set_title(f"(d) mapas do 62 e do 66 (grade = células)\n"
                  f"distância {distancia:.1f} px = {distancia / PASSO:.1f} células")
  for ax in axs:
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
  plt.show()

def test6(dataset: DatasetAnotacoesFaces, path):
  """Visualização da Data Augmentation do dataset"""
  
  fig, axs = plt.subplots(2, 4, figsize=(14, 7.2))
  for ax in axs.flat:
    sample = dataset[500]
    img = sample["imagem"]
    pts = sample["landmarks"]
    ax.imshow(img)
    desenhar_landmarks(ax, pts, s=4)
    ax.axis("off")
  fig.suptitle("Recortes aumentados: os pontos acompanham a imagem em todos eles", y=0.98)
  plt.show()

def test7(modelo, batch_size, tamanho):
  """Validação fora do loop de treinamento"""

  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  modelo.to(dispositivo)
  modelo.eval()
  
  loader = DataLoader(
      dataset=DatasetAnotacoesFaces(split="test", transform=False),
      batch_size=batch_size,
    )

  print(f"NME médio para o teste: {100*avaliar(modelo, loader, tamanho):5.2f}%")



modelo = FaceNet()
modelo.cabeca.load_state_dict(torch.load("francisco/pesos/modelo.pt", weights_only=True))
dataset_val = DatasetAnotacoesFaces(split="test", transform=False)

test4(modelo, dataset_val)

#dataset6 = DatasetAnotacoesFaces()
#test6(dataset6, "41_Swimming_Swimmer_41_701.jpg")

test7(modelo, 32, TAMANHO)