"""
Simple Tkinter window: pick a photo, see which cat the model thinks it is.
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk

from binary_classifier.pipeline import PhotoClassifier, format_report, train_on_folders

PREVIEW_SIZE = 380


class App(tk.Tk):
    def __init__(self, negative_folder, positive_folder, model_path):
        super().__init__()
        self.negative_folder = Path(negative_folder)
        self.positive_folder = Path(positive_folder)
        self.model_path = Path(model_path)
        self.classifier = None
        self._preview = None  # keeps a reference, otherwise Tk drops the image

        self.title("Classificador de gatas")
        self.resizable(False, False)
        self._build()

        if self.model_path.exists():
            self.classifier = PhotoClassifier.load(self.model_path)
            self._set_status(f"Modelo carregado: {self.model_path.name}")
        else:
            self.retrain()

    def _build(self):
        buttons = tk.Frame(self, padx=10, pady=10)
        buttons.pack(fill="x")
        self.open_button = tk.Button(buttons, text="Abrir foto...", width=16, command=self.open_photo)
        self.open_button.pack(side="left")
        self.train_button = tk.Button(buttons, text="Treinar novamente", width=16, command=self.retrain)
        self.train_button.pack(side="left", padx=8)

        self.canvas = tk.Label(self, width=PREVIEW_SIZE, height=PREVIEW_SIZE, bg="#222",
                               text="Nenhuma foto", fg="#aaa")
        self.canvas.pack(padx=10)

        self.result = tk.Label(self, text="", font=("Segoe UI", 20, "bold"), pady=6)
        self.result.pack()
        self.confidence = tk.Label(self, text="", font=("Segoe UI", 11))
        self.confidence.pack()

        self.report = tk.Text(self, width=58, height=14, font=("Consolas", 9), state="disabled")
        self.report.pack(padx=10, pady=(8, 4))

        self.status = tk.Label(self, text="", anchor="w", fg="#555", padx=10, pady=4)
        self.status.pack(fill="x")

    def _set_status(self, text):
        self.status.config(text=text)

    def _set_report(self, text):
        self.report.config(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.config(state="disabled")

    def open_photo(self):
        if self.classifier is None:
            messagebox.showinfo("Aguarde", "O modelo ainda está sendo treinado.")
            return
        path = filedialog.askopenfilename(
            title="Escolha uma foto",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            name, confidence = self.classifier.predict_image(path)
            with Image.open(path) as opened:
                preview = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Erro", f"Não consegui abrir a imagem:\n{exc}")
            return

        preview.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self._preview = ImageTk.PhotoImage(preview)
        self.canvas.config(image=self._preview, text="", width=PREVIEW_SIZE, height=PREVIEW_SIZE)

        self.result.config(text=name)
        self.confidence.config(text=f"confiança: {confidence:.1%}")
        self._set_status(Path(path).name)

    def retrain(self):
        self.open_button.config(state="disabled")
        self.train_button.config(state="disabled")
        self._set_status("Treinando... (carregando as fotos das duas pastas)")

        def work():
            try:
                classifier, report = train_on_folders(self.negative_folder, self.positive_folder)
                classifier.save(self.model_path)
                self.after(0, self._training_done, classifier, report, None)
            except Exception as exc:
                self.after(0, self._training_done, None, None, exc)

        threading.Thread(target=work, daemon=True).start()

    def _training_done(self, classifier, report, error):
        self.open_button.config(state="normal")
        self.train_button.config(state="normal")
        if error is not None:
            self._set_status("Falha no treino")
            messagebox.showerror("Erro no treino", str(error))
            return
        self.classifier = classifier
        self._set_report(format_report(report, classifier.class_names))
        self._set_status(f"Modelo treinado e salvo em {self.model_path}")


def run(negative_folder, positive_folder, model_path):
    App(negative_folder, positive_folder, model_path).mainloop()
