import sys
import os
from ultralytics import YOLO

# Força o Python a ler as bibliotecas do seu HD externo se necessário
sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Iniciando Otimização Dinâmica para o Processo Seletivo ---")

    model = YOLO("model.pt")

    # Verifica se está rodando no GitHub Actions ou em um sistema Linux
    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print(
            "Ambiente Linux/GitHub detectado! Executando conversão oficial para TFLite...")
        # A esteira vai executar essa linha nativamente sem travas
        model.export(format="tflite", imgsz=640)
    else:
        print("Ambiente Windows detectado localmente!")
        print("Gerando ONNX local para validação estrutural...")
        model.export(format="onnx", imgsz=640, dynamic=False)
        print("\nPronto! O model.tflite oficial será gerado automaticamente pelo GitHub Actions.")


if __name__ == "__main__":
    main()
