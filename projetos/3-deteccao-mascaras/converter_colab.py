
from ultralytics import YOLO
import shutil
import os

print("=== INICIANDO EXPORTAÇÃO NATIVA ULTRALYTICS NO COLAB ===")
# Carrega o modelo PyTorch que obteve 0.529 de mAP
model = YOLO("model.pt")

# Exporta usando o formato unificado do LiteRT do Google
model.export(format="litert")

# Localiza o arquivo .tflite ou .litert gerado dentro da subpasta e move para a raiz do projeto
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".tflite") or file.endswith(".litert"):
            origem = os.path.join(root, file)
            destinho = "model.tflite"
            shutil.copy2(origem, destinho)
            print(f"✅ Sucesso! Modelo copiado para: {destinho}")
