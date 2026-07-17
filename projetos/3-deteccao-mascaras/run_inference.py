import os
import sys
from ultralytics import YOLO

def main():
    print("============================================================")
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("============================================================")
    
    # Pega o caminho absoluto real de onde o script está rodando
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.tflite")
    
    print(f"🔍 Verificando arquivo físico no sistema:")
    print(f"Caminho esperado: {model_path}")
    print(f"Existe fisicamente?: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ Arquivo não encontrado pelo os.path. Forçando busca na pasta...")
        print(f"Conteúdo da pasta atual: {os.listdir(current_dir)}")
    
    # Forçamos o YOLO a receber o caminho absoluto e validado
    try:
        print("🎬 Carregando modelo no YOLO...")
        model = YOLO(model_path, task="detect")
        print("✅ Modelo carregado com sucesso no Ultralytics!")
    except Exception as e:
        print(f"❌ Erro ao carregar com o YOLO: {str(e)}")
        # Fallback de emergência caso o YOLO exija string direta do nome se estiver no mesmo diretório
        os.chdir(current_dir)
        model = YOLO("model.tflite", task="detect")
        print("✅ Modelo carregado via fallback de diretório!")

if __name__ == '__main__':
    main()
