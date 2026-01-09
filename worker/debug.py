import os
import traceback
import sys

from AletheiaEngine import AletheiaEngine


def main():
    ULTRAFACE_MODEL = "./models/ultraface.onnx"
    ARCFACE_MODEL = "./models/arcface.onnx"
    CLASSIFIER_MODEL = "./models/document_classifier.onnx"

    SELFIE_PATH = "./Examples/foto_1.jpeg"
    DOC_PATH = "./Examples/foto_2.jpg"

    for path in [
        ULTRAFACE_MODEL,
        ARCFACE_MODEL,
        CLASSIFIER_MODEL,
        SELFIE_PATH,
        DOC_PATH,
    ]:
        if not os.path.exists(path):
            print(f"ERRO: Arquivo não encontrado em: {path}")
            sys.exit(1)

    print("Inicializando Aletheia Engine...")
    try:
        engine = AletheiaEngine(
            classifier_path=CLASSIFIER_MODEL,
            debug_dir="debug_output",
        )
    except Exception:
        print("FALHA CRÍTICA NA INICIALIZAÇÃO:")
        traceback.print_exc()
        sys.exit(1)

    print("\nAnalisando biometria...")
    try:
        resultado = engine.verify(SELFIE_PATH, DOC_PATH)

        print("\n" + "=" * 40)
        print("        RESULTADO DA VALIDAÇÃO")
        print("=" * 40)

        if resultado["status"] == "error":
            print("STATUS: FALHA NA LÓGICA")
            print(f"DETALHE: {resultado.get('message')}")
        else:
            print(f"STATUS: {resultado['status']}")
            print(f"SCORE: {resultado.get('similarity_score', 0.0):.4f}")

    except Exception:
        print("\nERRO DE EXECUÇÃO (CRASH):")
        traceback.print_exc()

    print("=" * 40)


if __name__ == "__main__":
    main()
