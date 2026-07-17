import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada: Compatibilidade Total de Tensores TFLite ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado!")
        return

    model = YOLO("model.pt")

    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente GitHub Actions/Linux detectado.")

        # Limpeza total
        for f in ["model.tflite", "model_float32.tflite"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("model_saved_model"):
            shutil.rmtree("model_saved_model")

        print("Exportando modelo usando o formato TFLite via Keras para saídas padronizadas...")
        # keras=True altera a estrutura de exportação para usar o formato oficial unificado do TF
        exported_path = model.export(format="tflite", imgsz=640, keras=True)
        print(f"Caminho gerado: {exported_path}")

        # Move o arquivo final para a raiz
        if exported_path and os.path.exists(exported_path):
            if os.path.isdir(exported_path):
                for f in os.listdir(exported_path):
                    if f.endswith(".tflite"):
                        shutil.move(os.path.join(
                            exported_path, f), "model.tflite")
                        print(f"Movido: {f} -> model.tflite")
            else:
                shutil.move(exported_path, "model.tflite")
                print("Arquivo movido para a raiz.")

        # Fallbacks de segurança
        if os.path.exists("model_saved_model/model_float32.tflite"):
            shutil.move("model_saved_model/model_float32.tflite",
                        "model.tflite")
        elif os.path.exists("model_saved_model/model.tflite"):
            shutil.move("model_saved_model/model.tflite", "model.tflite")

        if os.path.exists("model.tflite"):
            print(
                f"Sucesso! model.tflite pronto com tamanho: {os.path.getsize('model.tflite')} bytes")
    else:
        print("Ambiente Windows local detectado.")


if __name__ == "__main__":
    main()
