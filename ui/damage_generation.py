import random
import os
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QProgressBar, QFrame, QSizePolicy,
                             QGraphicsOpacityEffect)
from PyQt6.QtGui import QPixmap, QColor, QPainter, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer

from .image_utils import process_image_for_display
from hole_generator.holes_generator import ImageHoleGenerator

class DamageGenerationActivity(QWidget):
    back_clicked = pyqtSignal()
    proceed_clicked = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.original_pixmap = None
        self.damage_mask = None # numpy array
        self.saved_path = None
        self.damage_mask_pixmap = None # QPixmap for blinking overlay
        
        self.hole_gen = ImageHoleGenerator(recreate_output_dir=False, holes=1)

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
        
        # Mask Overlay Label
        self.lbl_mask_overlay = QLabel(self.lbl_image)
        self.lbl_mask_overlay.setStyleSheet("background-color: transparent;")
        self.lbl_mask_overlay.hide()
        
        self.opacity_effect = QGraphicsOpacityEffect(self.lbl_mask_overlay)
        self.lbl_mask_overlay.setGraphicsEffect(self.opacity_effect)
        
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
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self.original_pixmap = process_image_for_display(pixmap, target_size=256)
        
        self.lbl_mask_overlay.hide()
        self.anim.stop()
        self.damage_mask = None
        self.saved_path = None
        self.damage_mask_pixmap = None
        
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
        
        if self.damage_mask_pixmap:
            # The overlay label should be the same size as the scaled pixmap
            # and positioned correctly within lbl_image
            lbl_w, lbl_h = self.lbl_image.width(), self.lbl_image.height()
            pm_w, pm_h = scaled_pixmap.width(), scaled_pixmap.height()
            
            off_x = (lbl_w - pm_w) // 2
            off_y = (lbl_h - pm_h) // 2
            
            self.lbl_mask_overlay.setGeometry(off_x, off_y, pm_w, pm_h)
            
            # Scale the mask pixmap to fit the new geometry
            scaled_mask = self.damage_mask_pixmap.scaled(
                pm_w, pm_h,
                Qt.AspectRatioMode.IgnoreAspectRatio, # We want it to stretch to fit
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_mask_overlay.setPixmap(scaled_mask)
        if self.damage_mask_pixmap:
            # The overlay label should be the same size as the scaled pixmap
            # and positioned correctly within lbl_image
            lbl_w, lbl_h = self.lbl_image.width(), self.lbl_image.height()
            pm_w, pm_h = scaled_pixmap.width(), scaled_pixmap.height()
            
            off_x = (lbl_w - pm_w) // 2
            off_y = (lbl_h - pm_h) // 2
            
            self.lbl_mask_overlay.setGeometry(off_x, off_y, pm_w, pm_h)
            
            # Scale the mask pixmap to fit the new geometry
            scaled_mask = self.damage_mask_pixmap.scaled(
                pm_w, pm_h,
                Qt.AspectRatioMode.IgnoreAspectRatio, # We want it to stretch to fit
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_mask_overlay.setPixmap(scaled_mask)

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
        
        # Use the existing generator by providing a dummy image of the correct size
        self.hole_gen.image = np.zeros((h, w, 3), dtype=np.uint8)
        _, mask = self.hole_gen.generate_holes(0.01)
        self.damage_mask = (mask * 255).astype(np.uint8)
        
        # Create a pixmap for the blinking overlay
        rgba_buffer = np.zeros((h, w, 4), dtype=np.uint8)
        mask_indices = self.damage_mask > 0
        rgba_buffer[mask_indices, 0:3] = 0  # R, G, B = 0 (black)
        rgba_buffer[mask_indices, 3] = 255 # Alpha = 255 (opaque)
        
        overlay_image = QImage(rgba_buffer.data, w, h, QImage.Format.Format_RGBA8888)
        self.damage_mask_pixmap = QPixmap.fromImage(overlay_image)

        self.update_image_display()
        
        self.lbl_mask_overlay.show()
        self.anim.start()
        
        self.progress.hide()
        self.btn_damage.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_proceed.setEnabled(True)
        self.saved_path = None

    def save_image(self):
        if not self.original_pixmap or self.damage_mask is None:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Damaged Image", "damaged.png", "PNG (*.png);;JPEG (*.jpg)")
        if save_path:
            self._save_to_disk(save_path)
            self.saved_path = save_path

    def _save_to_disk(self, path):
        result = self.original_pixmap.copy()
        
        h, w = self.damage_mask.shape
        
        # Create a black image with an alpha channel from the mask
        rgba_buffer = np.zeros((h, w, 4), dtype=np.uint8)
        mask_indices = self.damage_mask > 0
        rgba_buffer[mask_indices, 0:3] = 0  # R, G, B = 0 (black)
        rgba_buffer[mask_indices, 3] = 255 # Alpha = 255 (opaque)
        
        overlay_image = QImage(rgba_buffer.data, w, h, QImage.Format.Format_RGBA8888)
        
        # Draw the overlay on top of the original image
        painter = QPainter(result)
        painter.drawImage(0, 0, overlay_image)
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
