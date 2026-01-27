from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

def process_image_for_display(pixmap: QPixmap, target_size: int = 256) -> QPixmap:
    if pixmap.isNull():
        return pixmap

    # Crop to square (center crop)
    w, h = pixmap.width(), pixmap.height()
    size = min(w, h)
    x = (w - size) // 2
    y = (h - size) // 2
    cropped_pixmap = pixmap.copy(x, y, size, size)
    
    # Resize to target_size x target_size
    return cropped_pixmap.scaled(
        target_size, target_size, 
        Qt.AspectRatioMode.IgnoreAspectRatio, 
        Qt.TransformationMode.SmoothTransformation
    )
