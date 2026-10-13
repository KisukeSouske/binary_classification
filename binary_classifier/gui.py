"""
Simple Tkinter window: pick a photo, see which cat the model thinks it is.
"""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk

from binary_classifier.pipeline import PhotoClassifier, format_report, train_on_folders

PREVIEW_MAX = 380          # preview side in pixels on a large screen
PREVIEW_SCREEN_SHARE = 0.45  # never taller than this fraction of the screen
REPORT_LINES = 22            # visible lines of the report; the rest scrolls
TRAINING_POLL_MS = 100       # how often the window checks if training finished


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
        self.preview_size = min(PREVIEW_MAX, int(self.winfo_screenheight() * PREVIEW_SCREEN_SHARE))
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

        body = tk.Frame(self, padx=10)
        body.pack()

        # A Label's width/height are in text units until it holds an image, so
        # the fixed pixel size lives on a frame that doesn't resize to its child.
        left = tk.Frame(body)
        left.pack(side="left", anchor="n")
        box = tk.Frame(left, width=self.preview_size, height=self.preview_size, bg="#222")
        box.pack_propagate(False)
        box.pack()
        self.canvas = tk.Label(box, bg="#222", text="Nenhuma foto", fg="#aaa")
        self.canvas.pack(fill="both", expand=True)

        self.result = tk.Label(left, text="", font=("Segoe UI", 18, "bold"), pady=4)
        self.result.pack()
        self.confidence = tk.Label(left, text="", font=("Segoe UI", 10))
        self.confidence.pack()

        # The cross-validation report is longer than the preview is tall, so it scrolls
        report_box = tk.Frame(body)
        report_box.pack(side="left", anchor="n", fill="y", padx=(10, 0))
        scrollbar = tk.Scrollbar(report_box)
        scrollbar.pack(side="right", fill="y")
        self.report = tk.Text(report_box, width=66, height=REPORT_LINES, font=("Consolas", 9),
                              state="disabled", wrap="none", yscrollcommand=scrollbar.set)
        self.report.pack(side="left", fill="y")
        scrollbar.config(command=self.report.yview)

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

        preview.thumbnail((self.preview_size, self.preview_size))
        self._preview = ImageTk.PhotoImage(preview)
        self.canvas.config(image=self._preview, text="")

        self.result.config(text=name)
        self.confidence.config(text=f"confiança: {confidence:.1%}")
        self._set_status(Path(path).name)

    def retrain(self):
        self.open_button.config(state="disabled")
        self.train_button.config(state="disabled")
        self._set_status("Treinando... (validação cruzada + modelo final)")

        # Tk may only be touched from the main thread, so the worker just drops
        # its result in a queue and the main loop polls for it.
        results = queue.Queue()

        def work():
            try:
                classifier, report = train_on_folders(self.negative_folder, self.positive_folder)
                classifier.save(self.model_path)
                results.put((classifier, report, None))
            except Exception as exc:
                results.put((None, None, exc))

        threading.Thread(target=work, daemon=True).start()
        self._poll_training(results)

    def _poll_training(self, results):
        try:
            outcome = results.get_nowait()
        except queue.Empty:
            self.after(TRAINING_POLL_MS, self._poll_training, results)
            return
        self._training_done(*outcome)

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
