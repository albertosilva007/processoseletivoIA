import sys
from ultralytics import YOLO

# Força o Python a ler as bibliotecas do seu HD externo (letra E:) se necessário
sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Iniciando Otimização e Exportação via ONNX/TFLite ---")

    # 1. Carrega o modelo PyTorch treinado
    model = YOLO("model.pt")

    # 2. Exportação intermediária estável que funciona em qualquer Windows
    print("Exportando para formato estruturado aberto...")
    model.export(format="onnx", imgsz=640, dynamic=False)

    print("\n--- Processo local concluído! ---")
    print("Nota: O servidor Linux do GitHub Actions fará a validação nativa.")


if __name__ == "__main__":
    main()
