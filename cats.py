"""
Classificador binário de fotos de duas gatas (Nivea x Senna).

    python cats.py                 abre a interface (treina sozinho se não houver modelo salvo)
    python cats.py train           avalia com validação cruzada, treina com todas as fotos e salva
    python cats.py predict F1 F2   classifica uma ou mais fotos pelo terminal

As fotos não fazem parte do repositório. Espera-se uma pasta por classe:

    <data-dir>/Nivea/*.jpg
    <data-dir>/Senna/*.jpg

onde <data-dir> é, por ordem: --data-dir, a variável de ambiente CATS_DATA_DIR,
ou a pasta acima deste repositório. Use --classes para outros nomes de pasta.
"""
import argparse
import os
import sys
from pathlib import Path

from binary_classifier.pipeline import PhotoClassifier, TrainingConfig, format_report, train_on_folders

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("CATS_DATA_DIR", HERE.parent))
DEFAULT_CLASSES = ("Nivea", "Senna")  # label 0, label 1
MODEL_PATH = HERE / "models" / "cats_pca.npz"


def class_folders(args) -> tuple[Path, Path]:
    negative, positive = (args.data_dir / name for name in args.classes)
    for folder in (negative, positive):
        if not folder.is_dir():
            sys.exit(
                f"Pasta não encontrada: {folder}\n"
                f"Aponte --data-dir para a pasta que contém {args.classes[0]}/ e {args.classes[1]}/."
            )
    return negative, positive


def cmd_train(args):
    config = TrainingConfig(
        n_components=args.components, delta=args.delta, n_splits=args.folds, random_state=args.seed
    )
    classifier, report = train_on_folders(*class_folders(args), config)
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


def cmd_gui(args):
    from binary_classifier.gui import run
    run(*class_folders(args), MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"pasta que contém as pastas das classes (padrão: {DEFAULT_DATA_DIR})")
    parser.add_argument("--classes", nargs=2, metavar=("NEGATIVA", "POSITIVA"), default=list(DEFAULT_CLASSES),
                        help=f"nomes das duas pastas (padrão: {' '.join(DEFAULT_CLASSES)})")
    sub = parser.add_subparsers(dest="command")

    defaults = TrainingConfig()
    p_train = sub.add_parser("train", help="avalia com validação cruzada e treina o modelo final")
    p_train.add_argument("--folds", type=int, default=defaults.n_splits, help="número de folds da validação cruzada")
    p_train.add_argument("--components", type=int, default=defaults.n_components, help="componentes do PCA")
    p_train.add_argument("--delta", type=float, default=defaults.delta, help="força da penalidade ridge")
    p_train.add_argument("--seed", type=int, default=defaults.random_state, help="semente do sorteio dos folds")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="classifica fotos pelo terminal")
    p_predict.add_argument("photos", nargs="+")
    p_predict.set_defaults(func=cmd_predict)

    sub.add_parser("gui", help="abre a interface").set_defaults(func=cmd_gui)

    args = parser.parse_args()
    getattr(args, "func", cmd_gui)(args)


if __name__ == "__main__":
    main()
