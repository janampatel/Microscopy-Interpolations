"""Launch the interpolation GUI (`microinterp gui`)."""
import sys


def launch_gui(argv=None):
    from PySide6.QtWidgets import QApplication, QMainWindow
    from .interpolation_view import InterpolationView

    app = QApplication.instance() or QApplication(argv or sys.argv)
    window = QMainWindow()
    window.setWindowTitle("microinterp - Frame Interpolation (RIFE / FILM / CDFI / LDMVFI)")

    view = InterpolationView()
    window.setCentralWidget(view)
    window.resize(1280, 860)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
