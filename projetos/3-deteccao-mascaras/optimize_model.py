import sys
import os
import shutil
import subprocess
from ultralytics import YOLO

sys.path.insert(0, r"E:\pip_packages")


def main():
    print("--- Otimização e Auto-Commit na Esteira Linux ---")

    if not os.path.exists("model.pt"):
        print("Erro: model.pt não encontrado!")
        return

    model = YOLO("model.pt")

    # Se estiver rodando na esteira do GitHub (Linux)
    if os.getenv("GITHUB_ACTIONS") == "true" or sys.platform != "win32":
        print("Ambiente Linux da esteira detectado. Exportando via LiteRT...")

        # Limpa resíduos
        if os.path.exists("model.tflite"):
            os.remove("model.tflite")

        # Exporta usando o formato unificado exigido pela versão nova da Ultralytics
        exported_path = model.export(format="litert", imgsz=640)
        print(f"Exportado para: {exported_path}")

        # Move o arquivo gerado para a raiz com o nome que o validador quer (model.tflite)
        if exported_path and os.path.exists(exported_path):
            shutil.move(exported_path, "model.tflite")
            print("Arquivo model.tflite gerado com sucesso no Linux!")

            # --- TRUQUE DO AUTO-COMMIT ---
            # Configura o Git dentro da máquina virtual para devolver o arquivo pro seu repositório
            print("Forçando auto-commit do arquivo gerado para a raiz do seu GitHub...")
            try:
                subprocess.run(["git", "config", "--global",
                               "user.name", "github-actions[bot]"], check=True)
                subprocess.run(["git", "config", "--global", "user.email",
                               "github-actions[bot]@users.noreply.github.com"], check=True)
                subprocess.run(["git", "add", "model.tflite"], check=True)
                # O [skip ci] impede que o GitHub entre num loop infinito de execuções
                subprocess.run(
                    ["git", "commit", "-m", "Bot: Adicionando model.tflite oficial gerado no Linux [skip ci]"], check=True)
                subprocess.run(["git", "push"], check=True)
                print(
                    "Sucesso absoluto! O arquivo foi injetado de volta no seu repositório.")
            except Exception as git_err:
                print(
                    f"Aviso no auto-commit (pode ser falta de permissão de escrita no token): {git_err}")
    else:
        print("Rodando no Windows local. O processo real de exportação e commit ocorrerá na nuvem Linux.")


if __name__ == "__main__":
    main()
