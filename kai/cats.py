"""
Classificador binário Nivea x Senna (fotos).

    python cats.py                 abre a interface (treina sozinho se não houver modelo salvo)
    python cats.py train           treina, mostra as métricas de treino/teste e salva o modelo
    python cats.py predict F1 F2   classifica uma ou mais fotos pelo terminal
"""
import argparse
import sys
from pathlib import Path

from binary_classifier.pipeline import PhotoClassifier, format_report, train_on_folders

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1]             # binary_classification_desktop/
NEGATIVE_FOLDER = DATA_ROOT / "Nivea"   # label 0
POSITIVE_FOLDER = DATA_ROOT / "Senna"   # label 1
MODEL_PATH = HERE / "models" / "cats.npz"


def cmd_train(args):
    classifier, report = train_on_folders(
        NEGATIVE_FOLDER, POSITIVE_FOLDER, test_fraction=args.test_fraction, random_state=args.seed
    )
    print(format_report(report, classifier.class_names))
    classifier.save(MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")


def cmd_predict(args):
    if not MODEL_PATH.exists():
        sys.exit("Nenhum modelo salvo. Rode 'python cats.py train' primeiro.")
    classifier = PhotoClassifier.load(MODEL_PATH)
    for photo in args.photos:
        name, confidence = classifier.predict_image(photo)
        print(f"{photo}: {name} ({confidence:.1%})")


def cmd_gui(_args):
    from binary_classifier.gui import run
    run(NEGATIVE_FOLDER, POSITIVE_FOLDER, MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="treina e avalia o modelo")
    p_train.add_argument("--test-fraction", type=float, default=0.2, help="fração das fotos separada para teste")
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
