import sys
import os
from ultralytics import YOLO

# Força o Python a ler as bibliotecas do seu HD externo se necessário
sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Iniciando Otimização e Exportação Oficial no Servidor ---")

    # 1. Carrega o modelo PyTorch treinado
    if not os.path.exists("model.pt"):
        print("Erro: O arquivo model.pt não foi encontrado na raiz do projeto!")
        return

    model = YOLO("model.pt")

    # 2. Exporta diretamente para TFLite usando o ambiente Linux do GitHub
    print("Exportando model.pt para model.tflite nativamente...")
    try:
        # Tenta a exportação padrão exigida pelo desafio
        model.export(format="tflite", imgsz=640)
        print("Exportação direta concluída com sucesso!")
    except Exception as e:
        print(f"Aviso na exportação direta: {e}")
        print("Tentando alternativa compatível...")
        # Alternativa caso o ambiente precise do formato atualizado LiteRT
        model.export(format="litert", imgsz=640)

        # Garante que o arquivo final se chamará exatamente model.tflite
        if os.path.exists("model.litert") and not os.path.exists("model.tflite"):
            os.rename("model.litert", "model.tflite")
            print("Renomeado model.litert para model.tflite com sucesso.")

    print("\n--- Processo concluído! ---")


if __name__ == "__main__":
    main()
