import sys
from PyQt6.QtWidgets import QApplication
from ui.gui_app import MainAppWindow

class AppLauncher:
    def run(self):
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        window = MainAppWindow()
        window.show()
        
        sys.exit(app.exec())

if __name__ == "__main__":
    AppLauncher().run()