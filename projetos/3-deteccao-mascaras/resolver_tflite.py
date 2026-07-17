
from ultralytics import YOLO
import shutil
import os

print("--- RUNGING NATIVE EXPORT ON LINUX SERVER ---")
model = YOLO("projetos/3-deteccao-mascaras/model.pt")

# Exporta usando o motor oficial da Ultralytics no Linux
output_path = model.export(format="tflite")
print(f"Modelo exportado para: {output_path}")

# Move o arquivo gerado para a pasta correta do projeto substituindo o zerado
if os.path.exists("projetos/3-deteccao-mascaras/model.tflite"):
    os.remove("projetos/3-deteccao-mascaras/model.tflite")

shutil.move("projetos/3-deteccao-mascaras/model_saved_model/model_float32.tflite", "projetos/3-deteccao-mascaras/model.tflite")
print("✅ model.tflite atualizado com sucesso no servidor!")
