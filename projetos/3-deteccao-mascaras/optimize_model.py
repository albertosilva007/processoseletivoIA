import sys
import os
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada com Metadados de Detecção ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado!")
        return

    # Carrega o seu modelo treinado
    model = YOLO("model.pt")

    # Se estiver no Windows, vamos apenas avisar. O foco é a execução na esteira.
    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente de produção/Linux detectado. Iniciando exportação...")
        # nms=True força a inclusão das operações de detecção direto no arquivo final
        model.export(format="tflite", imgsz=640, nms=True, int8=False)
    else:
        print("Ambiente Windows detectado. A exportação com metadados rodará na nuvem.")


if __name__ == "__main__":
    main()
