import glob
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedLayout, QGraphicsOpacityEffect
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

class SlideshowWidget(QWidget):
    def __init__(self, image_folder="assets/arts", interval=3000):
        super().__init__()
        self.interval = interval
        self.current_idx = 0
        
        self.current_path = None
        self.next_path = None
        
        self.image_paths = sorted(glob.glob(f"{image_folder}/*.png") + glob.glob(f"{image_folder}/*.jpg"))
    
        self.layout = QStackedLayout(self)
        self.layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.lbl_bg = QLabel()
        self.lbl_fg = QLabel()
        
        for lbl in (self.lbl_bg, self.lbl_fg):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #1e1e1e;")
            lbl.setScaledContents(False) 
            self.layout.addWidget(lbl)
        
        self.layout.setCurrentWidget(self.lbl_fg)

        self.opacity_effect = QGraphicsOpacityEffect(self.lbl_fg)
        self.lbl_fg.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(1000)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.anim.finished.connect(self.on_fade_finished)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_slide)

        if self.image_paths:
            self.current_path = self.image_paths[0]
            self.set_image(self.lbl_fg, self.current_path)
            self.timer.start(self.interval)
        else:
            self.lbl_fg.setText("Assets not found.")
            self.lbl_fg.setStyleSheet("color: #888; font-size: 16px; background-color: #222;")

    def resizeEvent(self, event):
        if self.current_path:
            self.set_image(self.lbl_fg, self.current_path)
        if self.next_path:
            self.set_image(self.lbl_bg, self.next_path)
        super().resizeEvent(event)

    def set_image(self, label, path):
        if not path or self.width() <= 0 or self.height() <= 0:
            return
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            cropped = scaled.copy(x, y, self.width(), self.height())
            
            label.setPixmap(cropped)

    def next_slide(self):
        if not self.image_paths or len(self.image_paths) < 2:
            return
        
        next_idx = (self.current_idx + 1) % len(self.image_paths)
        self.next_path = self.image_paths[next_idx]
        
        self.set_image(self.lbl_bg, self.next_path)
        
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()
        
        self.current_idx = next_idx

    def on_fade_finished(self):
        self.current_path = self.next_path
        self.next_path = None
        self.lbl_fg.setPixmap(self.lbl_bg.pixmap())
        self.opacity_effect.setOpacity(1.0)

class MainMenuWidget(QWidget):
    def __init__(self, on_generate_damage, on_restore_image):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slideshow = SlideshowWidget()
        layout.addWidget(self.slideshow, stretch=4)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(20, 20, 20, 20)
        buttons_layout.setSpacing(20)

        btn_style = """
            QPushButton {
                background-color: #444; color: white; font-size: 16px; 
                padding: 15px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #555; }
        """

        self.btn_damage = QPushButton("Generate Damage")
        self.btn_damage.setStyleSheet(btn_style)
        self.btn_damage.clicked.connect(on_generate_damage)
        
        self.btn_restore = QPushButton("Restore Image")
        self.btn_restore.setStyleSheet(btn_style)
        self.btn_restore.clicked.connect(on_restore_image)

        buttons_layout.addWidget(self.btn_damage)
        buttons_layout.addWidget(self.btn_restore)
        
        layout.addLayout(buttons_layout, stretch=1)