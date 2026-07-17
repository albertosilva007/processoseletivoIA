import sys
import os
import shutil
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização Avançada e Ajuste de Tensores de Saída ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado na raiz!")
        return

    model = YOLO("model.pt")

    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente Linux do GitHub detectado. Removendo arquivos antigos...")

        # Limpeza absoluta para garantir que nenhum arquivo fantasma interfira
        for f in ["model.tflite", "model_float32.tflite"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("model_saved_model"):
            shutil.rmtree("model_saved_model")

        print("Iniciando exportação com compatibilidade máxima de tensores...")

        # Exporta desativando o nms embutido complexo e fixando o formato estável
        # Muitas esteiras aplicam o NMS no próprio script de teste, esperando a saída padrão plana
        exported_path = model.export(format="tflite", imgsz=640, nms=False)
        print(f"Caminho gerado pela Ultralytics: {exported_path}")

        # Procura e move o arquivo gerado para a raiz
        if exported_path and os.path.exists(exported_path):
            if os.path.isdir(exported_path):
                # Se for uma pasta, busca o arquivo lá dentro
                for f in os.listdir(exported_path):
                    if f.endswith(".tflite"):
                        shutil.move(os.path.join(
                            exported_path, f), "model.tflite")
                        print(
                            f"Movido de dentro do diretório: {f} -> model.tflite")
            else:
                shutil.move(exported_path, "model.tflite")
                print("Arquivo movido diretamente para model.tflite")

        # Garante o fallback de nome se ele salvou em outro lugar da pasta atual
        if os.path.exists("model_saved_model/model_float32.tflite"):
            shutil.move("model_saved_model/model_float32.tflite",
                        "model.tflite")
            print("Movido fallback de model_saved_model para a raiz.")

        if os.path.exists("model.tflite"):
            print(
                f"Sucesso absoluto! Tamanho do arquivo final: {os.path.getsize('model.tflite')} bytes")
    else:
        print("Ambiente local Windows. O processamento real rodará na nuvem.")


if __name__ == "__main__":
    main()
