"""
Classificador binário Nivea x Senna.

    python main.py                 abre a interface (treina sozinho se não houver modelo salvo)
    python main.py train           treina, mostra as métricas de treino/teste e salva o modelo
    python main.py predict F1 F2   classifica uma ou mais fotos pelo terminal
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kai"))

from binary_classifier.pipeline import CatClassifier, format_report, train  # noqa: E402

DATA_ROOT = HERE.parent
NEGATIVE_FOLDER = DATA_ROOT / "Nivea"   # label 0
POSITIVE_FOLDER = DATA_ROOT / "Senna"   # label 1
MODEL_PATH = HERE / "kai" / "models" / "model.npz"


def cmd_train(args):
    classifier, report = train(NEGATIVE_FOLDER, POSITIVE_FOLDER, test_size=args.test_size, seed=args.seed)
    print(format_report(report, classifier.class_names))
    classifier.save(MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")


def cmd_predict(args):
    if not MODEL_PATH.exists():
        sys.exit("Nenhum modelo salvo. Rode 'python main.py train' primeiro.")
    classifier = CatClassifier.load(MODEL_PATH)
    for photo in args.photos:
        name, confidence = classifier.predict_image(Path(photo))
        print(f"{photo}: {name} ({confidence:.1%})")


def cmd_gui(_args):
    from binary_classifier.gui import run
    run(NEGATIVE_FOLDER, POSITIVE_FOLDER, MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="treina e avalia o modelo")
    p_train.add_argument("--test-size", type=float, default=0.2, help="fração de fotos de cada gata separada para teste")
    p_train.add_argument("--seed", type=int, default=0, help="semente do sorteio treino/teste")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="classifica fotos pelo terminal")
    p_predict.add_argument("photos", nargs="+")
    p_predict.set_defaults(func=cmd_predict)

    sub.add_parser("gui", help="abre a interface").set_defaults(func=cmd_gui)

    args = parser.parse_args()
    getattr(args, "func", cmd_gui)(args)


if __name__ == "__main__":
    main()
