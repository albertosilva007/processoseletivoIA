import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada: Reconstrução Completa do Grafo ---")

    if not os.path.exists("model.pt"):
        print("Erro crítico: model.pt não foi encontrado na raiz do projeto!")
        return

    # Carrega o seu modelo treinado (com mAP de 0.529)
    model = YOLO("model.pt")

    # Execução na esteira Linux do GitHub
    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente do GitHub Actions detectado. Iniciando exportação limpa...")

        # Garante que nenhum resíduo físico local atrapalhe
        for f in ["model.tflite", "model_float32.tflite"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        if os.path.exists("model_saved_model"):
            shutil.rmtree("model_saved_model", ignore_errors=True)

        print("Exportando via Ultralytics com formato estável...")
        # exportando sem NMS embutido para evitar incompatibilidade de versão do TF Runtime da esteira
        exported_path = model.export(
            format="tflite", imgsz=640, half=False, int8=False)
        print(f"Retorno do exportador: {exported_path}")

        # Caça o arquivo gerado e força a entrega na raiz
        if exported_path and os.path.exists(exported_path):
            if os.path.isdir(exported_path):
                for file in os.listdir(exported_path):
                    if file.endswith(".tflite"):
                        shutil.move(os.path.join(
                            exported_path, file), "model.tflite")
                        print(
                            f"Sucesso: {file} movido para a raiz como model.tflite")
            else:
                shutil.move(exported_path, "model.tflite")
                print("Sucesso: arquivo movido direto para a raiz.")

        # Fallback definitivo caso a esteira use caminhos estáticos da estrutura interna do TF
        if os.path.exists("model_saved_model/model_float32.tflite"):
            shutil.move("model_saved_model/model_float32.tflite",
                        "model.tflite")
        elif os.path.exists("model_saved_model/model.tflite"):
            shutil.move("model_saved_model/model.tflite", "model.tflite")

        if os.path.exists("model.tflite"):
            print(
                f"Concluído! Arquivo pronto para o validador. Tamanho: {os.path.getsize('model.tflite')} bytes")
    else:
        print("Ambiente Windows local. O processo real rodará no servidor do GitHub.")


if __name__ == "__main__":
    main()
