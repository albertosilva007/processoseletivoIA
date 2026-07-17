import os
import sys
from ultralytics import YOLO

def main():
    print("============================================================")
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("============================================================")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define os caminhos possíveis baseados no que descobrimos no log
    caminho_pasta_yolo = os.path.join(current_dir, "model_tflite", "model_float32.tflite")
    caminho_pasta_yolo_alt = os.path.join(current_dir, "model_tflite", "model.tflite")
    caminho_direto = os.path.join(current_dir, "model.tflite")
    
    # Escolhe o caminho que realmente existir no servidor
    if os.path.exists(caminho_pasta_yolo):
        model_path = caminho_pasta_yolo
    elif os.path.exists(caminho_pasta_yolo_alt):
        model_path = caminho_pasta_yolo_alt
    else:
        model_path = caminho_direto

    print(f"🎬 Carregando modelo do caminho validado: {model_path}")
    
    try:
        model = YOLO(model_path, task="detect")
        print("✅ SUCESSO! Modelo carregado perfeitamente no Ultralytics!")
    except Exception as e:
        print(f"❌ Erro ao carregar: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
