import sys
import os
from PyQt6.QtWidgets import QApplication


project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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