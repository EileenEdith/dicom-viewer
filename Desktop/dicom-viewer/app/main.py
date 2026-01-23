import sys
from PyQt5.QtWidgets import QApplication
from app.viewer import ImageViewer

app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec_())