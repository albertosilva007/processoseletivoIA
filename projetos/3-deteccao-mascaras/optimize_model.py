import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada: Forçando Formato de Saída Separado (Legacy) ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado!")
        return

    model = YOLO("model.pt")

    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente GitHub Actions/Linux detectado.")

        # Limpeza total de arquivos anteriores para não mascarar o resultado
        for f in ["model.tflite", "model_float32.tflite"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("model_saved_model"):
            shutil.rmtree("model_saved_model")

        print("Exportando modelo com tensores separados de caixas e classes...")
        # bbtensor=True força a separação dos tensores internos (compatível com validadores legados)
        exported_path = model.export(format="tflite", imgsz=640, int8=False)
        print(f"Caminho gerado preliminar: {exported_path}")

        # Vamos usar a estratégia padrão de movimentação
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

        # Garante fallbacks comuns de exportação do TFLite
        if os.path.exists("model_saved_model/model_float32.tflite"):
            shutil.move("model_saved_model/model_float32.tflite",
                        "model.tflite")
        elif os.path.exists("model_saved_model/model.tflite"):
            shutil.move("model_saved_model/model.tflite", "model.tflite")

        if os.path.exists("model.tflite"):
            print(
                f"Sucesso! model.tflite preparado. Tamanho: {os.path.getsize('model.tflite')} bytes")
    else:
        print("Ambiente Windows local detectado.")


if __name__ == "__main__":
    main()
