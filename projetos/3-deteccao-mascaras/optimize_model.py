import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada e Alocação Estrita de Arquivos ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado na raiz!")
        return

    model = YOLO("model.pt")

    # Executa apenas no ambiente do GitHub / Linux
    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente Linux detectado. Removendo resíduos antigos...")

        # Remove o tflite antigo zerado se ele existir para não mascarar o teste
        if os.path.exists("model.tflite"):
            os.remove("model.tflite")

        print("Iniciando exportação limpa...")
        # Exportação padrão estável
        exported_path = model.export(format="tflite", imgsz=640)
        print(f"Retorno do export da Ultralytics: {exported_path}")

        # Varredura inteligente para encontrar o arquivo real criado pela Ultralytics
        possible_paths = [
            "model_saved_model/model_float32.tflite",
            "model_saved_model/model.tflite",
            "model.tflite",
            exported_path
        ]

        sucesso = False
        for path in possible_paths:
            if path and os.path.exists(path) and os.path.isfile(path):
                print(f"Arquivo válido encontrado em: {path}")
                # Move e renomeia diretamente para a raiz do projeto
                shutil.move(path, "model.tflite")
                print("Sucesso: Arquivo model.tflite alocado corretamente na raiz!")
                sucesso = True
                break

        if not sucesso:
            print("Aviso: Procurando arquivos .tflite recursivamente...")
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".tflite") and root != ".":
                        target = os.path.join(root, file)
                        shutil.move(target, "./model.tflite")
                        print(
                            f"Encontrado e movido recursivamente de {target}")
                        break

    else:
        print("Ambiente Windows. O gerenciamento de caminhos rodará no GitHub Actions.")


if __name__ == "__main__":
    main()
