import sys
import os

# Força o Python a ler as bibliotecas do seu HD externo (Letra E:)
sys.path.insert(0, r"E:\pip_packages")

try:
    import onnx2tf
    import tensorflow as tf
    print("=== Módulos do HD Externo carregados com sucesso! ===")
    
    # Executa a conversão de forma direta
    print("\nIniciando conversão de model.onnx para TFLite...")
    onnx2tf.convert(
        input_onnx_file_path="model.onnx",
        output_folder_path="model_tflite",
    )
    print("\n=== Conversão executada! Verifique a pasta 'model_tflite' ===")

except ModuleNotFoundError as e:
    print(f"\nErro ao carregar os módulos: {e}")
    print("Dica: Verifique se o seu HD externo continua mapeado na letra E: hoje.")