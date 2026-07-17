import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Dinâmica: Modo de Compatibilidade Estrita TFLite ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado!")
        return

    model = YOLO("model.pt")

    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente GitHub Actions/Linux detectado.")

        # Limpeza total de execuções anteriores
        for f in ["model.tflite", "model_float32.tflite"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("model_saved_model"):
            shutil.rmtree("model_saved_model")

        print("Exportando modelo com precisão Float32 padrão (sem quantização)...")
        # Força o formato puro float32 e tamanho de imagem padrão exigido pelo edital
        exported_path = model.export(format="tflite", imgsz=640, int8=False)
        print(f"Caminho gerado: {exported_path}")

        # Mapeamento e movimentação para a raiz com o nome exato esperado
        if exported_path and os.path.exists(exported_path):
            if os.path.isdir(exported_path):
                for f in os.listdir(exported_path):
                    if f.endswith(".tflite"):
                        shutil.move(os.path.join(
                            exported_path, f), "model.tflite")
                        print(f"Movido do diretório: {f} -> model.tflite")
            else:
                shutil.move(exported_path, "model.tflite")
                print("Arquivo movido para a raiz.")

        # Fallback de segurança para o nome padrão float32 da Ultralytics
        if os.path.exists("model_saved_model/model_float32.tflite"):
            shutil.move("model_saved_model/model_float32.tflite",
                        "model.tflite")
            print("Fallback: movido model_float32.tflite para a raiz.")

        if os.path.exists("model.tflite"):
            print("Sucesso: model.tflite pronto para avaliação.")
    else:
        print("Ambiente Windows local detectado.")


if __name__ == "__main__":
    main()
