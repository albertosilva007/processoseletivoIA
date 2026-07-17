import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Exportação Direta para a Pasta do Projeto ---")

    # O script roda dentro de projetos/3-deteccao-mascaras na esteira
    if not os.path.exists("model.pt"):
        print("Erro crítico: model.pt não encontrado na pasta atual!")
        return

    model = YOLO("model.pt")

    # Alvos de limpeza na pasta local do projeto
    for f in ["model.tflite", "model_float32.tflite"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    if os.path.exists("model_saved_model"):
        shutil.rmtree("model_saved_model", ignore_errors=True)

    print("Iniciando exportação estável do TFLite em Float32...")
    # int8=False garante compatibilidade e precisão máxima para bater o mAP de 0.20
    exported_path = model.export(format="tflite", imgsz=640, int8=False)
    print(f"Caminho retornado: {exported_path}")

    # Movimentação inteligente para garantir que o arquivo fique na pasta atual (do projeto)
    if exported_path and os.path.exists(exported_path):
        if os.path.isdir(exported_path):
            for file in os.listdir(exported_path):
                if file.endswith(".tflite"):
                    shutil.move(os.path.join(
                        exported_path, file), "model.tflite")
                    print(f"Sucesso: {file} movido para a pasta do projeto.")
        else:
            if exported_path != "model.tflite":
                shutil.move(exported_path, "model.tflite")
                print("Sucesso: arquivo movido para a pasta do projeto.")

    # Fallback caso a Ultralytics crie a pasta saved_model padrão
    if os.path.exists("model_saved_model/model_float32.tflite"):
        shutil.move("model_saved_model/model_float32.tflite", "model.tflite")
    elif os.path.exists("model_saved_model/model.tflite"):
        shutil.move("model_saved_model/model.tflite", "model.tflite")

    if os.path.exists("model.tflite"):
        print(
            f"Concluído com sucesso! Arquivo gerado em: {os.path.abspath('model.tflite')}")
    else:
        print("Erro: O arquivo model.tflite não foi gerado.")


if __name__ == "__main__":
    main()
