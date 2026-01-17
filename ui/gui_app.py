from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from ui.main_menu import MainMenuWidget
from ui.damage_generation import DamageGenerationActivity
from ui.inpainting_activity import InpaintingActivity

class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Art Restoration App")
        self.setFixedSize(1000, 700)
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Main Menu
        self.menu_view = MainMenuWidget(
            on_generate_damage=lambda: self.stack.setCurrentIndex(1),
            on_restore_image=lambda: self.stack.setCurrentIndex(2)
        )
        
        # Damage Generation
        self.damage_view = DamageGenerationActivity()
        self.damage_view.back_clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.damage_view.proceed_clicked.connect(self.go_to_inference)

        # Inpainting
        self.inpainting_view = InpaintingActivity()
        self.inpainting_view.back_clicked.connect(lambda: self.stack.setCurrentIndex(1)) # Back to Damage

        self.stack.addWidget(self.menu_view)
        self.stack.addWidget(self.damage_view)
        self.stack.addWidget(self.inpainting_view)
        
        
        self.setStyleSheet("background-color: #2b2b2b; color: #ddd;")

    def go_to_inference(self, path):
        self.inpainting_view.set_image(path)
        self.stack.setCurrentIndex(2)
