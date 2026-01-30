import os
import random
import glob
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QFrame, QSizePolicy, QMessageBox
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QFrame, QSizePolicy, QMessageBox
from PyQt6.QtGui import QPixmap, QColor, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject

from pipeline import run_restoration_pipeline

class PipelineWorker(QObject):
    """Uruchamia pipeline w osobnym wątku, aby nie blokować UI."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            # To jest blokujące wywołanie, ale działa w osobnym wątku
            result_path = run_restoration_pipeline(self.image_path)
            self.finished.emit(str(result_path)) # Upewniamy się, że to string
        except Exception as e:
            self.error.emit(str(e))

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class InpaintingActivity(QWidget):
    back_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.main_pixmap = None
        self.small_pixmap = None
        self.image_path = None
        self.restored_pixmap_cache = None
        self.worker = None
        self.thread = None
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Left Panel (Controls)
        left_panel = QFrame()
        left_panel.setFixedWidth(250)
        left_panel.setStyleSheet("background-color: #333; border-right: 1px solid #444; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 20, 15, 20)

        # Back Button
        self.btn_back = QPushButton("← Back")
        self.btn_back.clicked.connect(self.back_clicked.emit)
        self.btn_back.setStyleSheet("text-align: left; padding: 10px; border: none; color: #aaa; font-size: 14px;")
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)

        # Load Button
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.open_image_dialog)
        self.btn_load.setMinimumHeight(45)
        self.btn_load.setStyleSheet("background-color: #007ACC; color: white; border-radius: 5px; font-weight: bold;")
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)

        # Restore Button
        self.btn_restore = QPushButton("Restore Image")
        self.btn_restore.clicked.connect(self.restore_image)
        self.btn_restore.setMinimumHeight(45)
        self.btn_restore.setEnabled(False)
        self.btn_restore.setStyleSheet("""
            QPushButton { background-color: #388E3C; color: white; border-radius: 5px; font-weight: bold; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_restore.setCursor(Qt.CursorShape.PointingHandCursor)

        # Save Button
        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setMinimumHeight(45)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #444; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { color: #777; }
        """)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.progress.setStyleSheet("QProgressBar { height: 6px; background: #444; border: none; border-radius: 3px; } QProgressBar::chunk { background: #388E3C; border-radius: 3px; }")

        left_layout.addWidget(self.btn_back)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_restore)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_save)

        # Right Panel (Images)
        self.right_container = QWidget()
        self.right_container.setStyleSheet("background-color: #222; border-radius: 10px; border: 1px solid #444;")
        self.right_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.lbl_main = QLabel(self.right_container)
        self.lbl_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_main.setText("No image selected")
        self.lbl_main.setStyleSheet("color: #666; font-size: 16px;")
        
        self.lbl_small = ClickableLabel(self.right_container)
        self.lbl_small.setFixedSize(150, 150)
        self.lbl_small.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_small.setText("No\nOriginal")
        self.lbl_small.setStyleSheet("QLabel { background-color: #111; border: 2px solid #666; color: #555; font-size: 12px; } QLabel:hover { border-color: #888; }")
        self.lbl_small.clicked.connect(self.swap_images)
        self.lbl_small.hide()

        layout.addWidget(left_panel)
        layout.addWidget(self.right_container, stretch=1)

    def resizeEvent(self, event):
        if self.right_container:
            m = 10
            w = self.right_container.width()
            h = self.right_container.height()
            self.lbl_main.setGeometry(m, m, w - 2*m, h - 2*m)
            
            sm_m = 20
            sm_w = self.lbl_small.width()
            sm_h = self.lbl_small.height()
            self.lbl_small.move(sm_m + m, h - sm_h - sm_m - m)
            
            self.update_displays()
        super().resizeEvent(event)

    def open_image_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            # TODO: cropping image to right size
            self.set_image(fname)

    def set_image(self, damaged_path):
        self.image_path = damaged_path

        # Display the damaged image
        self.main_pixmap = QPixmap(damaged_path)
        self.small_pixmap = None
        self.restored_pixmap_cache = None
        self.lbl_small.hide()
        self.btn_restore.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.update_displays()

    def restore_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "Missing Image", "Cannot restore image without image :3")
            return

        self.btn_restore.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.show()

        # Ustawienie i uruchomienie pipeline'u w osobnym wątku
        self.thread = QThread()
        self.worker = PipelineWorker(self.image_path)
        self.worker.moveToThread(self.thread)

        # Połączenie sygnałów z workera do slotów w tej klasie
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.error.connect(self.on_pipeline_error)
        
        # Sprzątanie po zakończeniu pracy wątku
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)

        # Start wątku
        self.thread.start()

    def on_pipeline_finished(self, final_image_path: str):
        """Slot obsługujący pomyślne zakończenie pipeline'u."""
        restored_pixmap = None
        if os.path.exists(final_image_path):
            restored_pixmap = QPixmap(final_image_path)
        else:
            self.on_pipeline_error(f"Pipeline zakończony, ale nie znaleziono pliku wyjściowego: {final_image_path}")
            return

        # --- Update UI after pipeline finishes (or fails) ---
        self.progress.hide()
        self.btn_load.setEnabled(True)

        if restored_pixmap and not restored_pixmap.isNull():
            self.small_pixmap = self.main_pixmap
            self.main_pixmap = restored_pixmap
            self.restored_pixmap_cache = restored_pixmap
            self.lbl_small.show()
            self.lbl_small.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_save.setEnabled(True)
            self.btn_restore.setEnabled(True)
        else:
            self.btn_restore.setEnabled(True)

        self.update_displays()

    def on_pipeline_error(self, error_message: str):
        """Slot obsługujący błędy z pipeline'u."""
        print(f"An error occurred during the restoration pipeline: {error_message}")
        QMessageBox.critical(self, "Błąd w trakcie przetwarzania", f"Wystąpił błąd podczas przywracania obrazu:\n{error_message}")
        self.progress.hide()
        self.btn_load.setEnabled(True)
        self.btn_restore.setEnabled(True)

    def swap_images(self):
        if not self.small_pixmap or not self.main_pixmap: return
        self.main_pixmap, self.small_pixmap = self.small_pixmap, self.main_pixmap
        self.update_displays()

    def update_displays(self):
        if self.main_pixmap and not self.main_pixmap.isNull():
            scaled = self.main_pixmap.scaled(self.lbl_main.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_main.setPixmap(scaled)
            self.lbl_main.setText("")
        else:
            self.lbl_main.clear()
            self.lbl_main.setText("No image selected")

        if self.small_pixmap and not self.small_pixmap.isNull():
            scaled_small = self.small_pixmap.scaled(self.lbl_small.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            w, h = self.lbl_small.width(), self.lbl_small.height()
            x = (scaled_small.width() - w) // 2
            y = (scaled_small.height() - h) // 2
            self.lbl_small.setPixmap(scaled_small.copy(x, y, w, h))
        else:
            self.lbl_small.clear()
            self.lbl_small.setText("No\nOriginal")

    def save_image(self):
        pixmap_to_save = self.restored_pixmap_cache if self.restored_pixmap_cache else self.main_pixmap
        if not pixmap_to_save: return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "restored.png", "PNG (*.png);;JPEG (*.jpg)")
        if save_path: pixmap_to_save.save(save_path)