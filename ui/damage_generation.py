import random
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QProgressBar, QFrame, QSizePolicy,
                             QGraphicsOpacityEffect)
from PyQt6.QtGui import QPixmap, QColor, QPainter
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer

class DamageGenerationActivity(QWidget):
    back_clicked = pyqtSignal()
    proceed_clicked = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.original_pixmap = None
        self.mask_geometry = None
        self.saved_path = None
        
        self.init_ui()

    def init_ui(self):
        # Main Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left Panel (Controls)
        left_panel = QFrame()
        left_panel.setFixedWidth(250)
        left_panel.setStyleSheet("background-color: #333; border-right: 1px solid #444;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Back Button
        self.btn_back = QPushButton("← Back")
        self.btn_back.clicked.connect(self.back_clicked.emit)
        self.btn_back.setStyleSheet("text-align: left; padding: 10px; border: none; color: #aaa;")
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Load Button
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.open_image_dialog)
        self.btn_load.setMinimumHeight(40)
        self.btn_load.setStyleSheet("background-color: #007ACC; color: white; border-radius: 5px;")
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)

        # Generate Damage Button
        self.btn_damage = QPushButton("Generate Damage")
        self.btn_damage.clicked.connect(self.generate_damage)
        self.btn_damage.setMinimumHeight(40)
        self.btn_damage.setEnabled(False)
        self.btn_damage.setStyleSheet("""
            QPushButton { background-color: #D32F2F; color: white; border-radius: 5px; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_damage.setCursor(Qt.CursorShape.PointingHandCursor)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.progress.setStyleSheet("QProgressBar { height: 4px; background: #444; border: none; } QProgressBar::chunk { background: #D32F2F; }")

        # Save Button
        self.btn_save = QPushButton("Save Damaged Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #444; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { color: #777; }
        """)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Proceed Button
        self.btn_proceed = QPushButton("Proceed to Inpainting →")
        self.btn_proceed.clicked.connect(self.on_proceed)
        self.btn_proceed.setEnabled(False)
        self.btn_proceed.setMinimumHeight(50)
        self.btn_proceed.setStyleSheet("""
            QPushButton { background-color: #388E3C; color: white; border-radius: 5px; font-weight: bold; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_proceed.setCursor(Qt.CursorShape.PointingHandCursor)

        left_layout.addWidget(self.btn_back)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_damage)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(self.btn_proceed)

        # Right Panel (Image Display)
        right_layout = QVBoxLayout()
        
        # Container for the image
        self.image_container = QWidget()
        self.image_container_layout = QVBoxLayout(self.image_container)
        self.image_container_layout.setContentsMargins(0,0,0,0)
        
        self.lbl_image = QLabel("Load an image to start")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #222; color: #666; font-size: 14px;")
        self.lbl_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Mask Widget
        self.mask_widget = QWidget(self.lbl_image)
        self.mask_widget.setStyleSheet("background-color: black;")
        self.mask_widget.hide()
        
        self.opacity_effect = QGraphicsOpacityEffect(self.mask_widget)
        self.mask_widget.setGraphicsEffect(self.opacity_effect)
        
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(2000)
        self.anim.setLoopCount(-1)
        self.anim.setKeyValueAt(0, 0.0)
        self.anim.setKeyValueAt(0.5, 1.0)
        self.anim.setKeyValueAt(1, 0.0)

        self.image_container_layout.addWidget(self.lbl_image)
        right_layout.addWidget(self.image_container)

        main_layout.addWidget(left_panel)
        main_layout.addLayout(right_layout)

    def open_image_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.load_image(fname)

    def load_image(self, path):
        self.current_image_path = path
        self.original_pixmap = QPixmap(path)
        if self.original_pixmap.isNull():
            return
        
        self.mask_widget.hide()
        self.anim.stop()
        self.mask_geometry = None
        self.saved_path = None
        
        self.btn_damage.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_proceed.setEnabled(False)
        
        self.update_image_display()

    def update_image_display(self):
        if not self.original_pixmap:
            return
            
        size = self.lbl_image.size()
        scaled_pixmap = self.original_pixmap.scaled(
            size, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_image.setPixmap(scaled_pixmap)
        
        if self.mask_geometry:
            self.update_mask_display(scaled_pixmap)

    def generate_damage(self):
        if not self.original_pixmap:
            return

        self.progress.show()
        self.btn_damage.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_proceed.setEnabled(False)
        
        QTimer.singleShot(600, self._perform_damage_generation)

    def _perform_damage_generation(self):
        w = self.original_pixmap.width()
        h = self.original_pixmap.height()
        
        self.mask_geometry = self._create_mask_rect(w, h)
        
        self.update_image_display()
        
        self.mask_widget.show()
        self.anim.start()
        
        self.progress.hide()
        self.btn_damage.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_proceed.setEnabled(True)
        self.saved_path = None

    def _create_mask_rect(self, w, h):
        size = min(w, h) // 4
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return (x, y, size, size)

    def update_mask_display(self, displayed_pixmap):
        lbl_w = self.lbl_image.width()
        lbl_h = self.lbl_image.height()
        pm_w = displayed_pixmap.width()
        pm_h = displayed_pixmap.height()
        
        off_x = (lbl_w - pm_w) // 2
        off_y = (lbl_h - pm_h) // 2
        
        scale_x = pm_w / self.original_pixmap.width()
        scale_y = pm_h / self.original_pixmap.height()
        
        mx, my, mw, mh = self.mask_geometry
        
        final_x = off_x + int(mx * scale_x)
        final_y = off_y + int(my * scale_y)
        final_w = int(mw * scale_x)
        final_h = int(mh * scale_y)
        
        self.mask_widget.setGeometry(final_x, final_y, final_w, final_h)

    def save_image(self):
        if not self.original_pixmap or not self.mask_geometry:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Damaged Image", "damaged.png", "PNG (*.png);;JPEG (*.jpg)")
        if save_path:
            self._save_to_disk(save_path)
            self.saved_path = save_path

    def _save_to_disk(self, path):
        result = self.original_pixmap.copy()
        painter = QPainter(result)
        painter.setBrush(QColor("black"))
        painter.setPen(Qt.PenStyle.NoPen)
        mx, my, mw, mh = self.mask_geometry
        painter.drawRect(mx, my, mw, mh)
        painter.end()
        result.save(path)

    def on_proceed(self):
        path_to_pass = self.saved_path
        
        # If not saved by user, save to temp
        if not path_to_pass:
            temp_path = os.path.abspath("temp_damaged_image.png")
            self._save_to_disk(temp_path)
            path_to_pass = temp_path
        
        self.proceed_clicked.emit(path_to_pass)
