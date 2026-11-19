import torchvision
import torch
from torch import nn


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
    self.apply(self._inicializar_pesos)

  @staticmethod
  def _inicializar_pesos(modulo: nn.Module) -> None:
    if isinstance(modulo, nn.ConvTranspose2d):
      nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
      if modulo.bias is not None:
        nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, nn.Conv2d):
      nn.init.normal_(modulo.weight, mean=0.0, std=0.01)
      if modulo.bias is not None:
        nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, nn.BatchNorm2d):
      if modulo.weight is not None:
        nn.init.ones_(modulo.weight)
      if modulo.bias is not None:
        nn.init.zeros_(modulo.bias)

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