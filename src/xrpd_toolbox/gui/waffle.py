import sys

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xrpd_toolbox.gui.bad_pixel_gui import run_bad_pixel_gui

# Constants
BEAMLINES: list[str] = ["i11", "i15-1"]


class WaffleGUI(QWidget):
    """Main waffle GUI with square buttons that resize dynamically"""

    def __init__(self, beamline: str, grid_size: int = 3) -> None:
        super().__init__()
        self.beamline: str = beamline
        self.grid_size: int = grid_size
        self.setWindowTitle(f"{beamline} Modules")
        self.setMinimumSize(300, 300)  # Prevent uncontrolled expansion

        self.layout: QGridLayout = QGridLayout()  # type: ignore
        self.layout.setSpacing(5)
        self.buttons: list[QPushButton] = []

        # Create NxN waffle buttons
        for row in range(grid_size):
            for col in range(grid_size):
                idx: int = row * grid_size + col
                btn: QPushButton = QPushButton(f"Module {idx}")
                btn.clicked.connect(lambda checked: self.launch_module(idx))  # type: ignore # noqa
                self.layout.addWidget(btn, row, col)
                self.buttons.append(btn)

        self.setLayout(self.layout)

        # Force initial square sizing
        self.adjust_button_sizes()

    def resizeEvent(self, event) -> None:  # type: ignore # noqa
        """Keep buttons square when window resizes"""
        self.adjust_button_sizes()
        super().resizeEvent(event)

    def adjust_button_sizes(self) -> None:
        """Compute max square size that fits the grid"""
        if not self.buttons:
            return
        # total spacing between buttons
        spacing: int = self.layout.spacing() * (self.grid_size - 1)
        available_width: int = self.width() - spacing
        available_height: int = self.height() - spacing
        button_size: int = min(available_width, available_height) // self.grid_size
        for btn in self.buttons:
            btn.setFixedSize(QSize(button_size, button_size))

    def launch_module(self, idx: int) -> None:
        """Open submodule GUI (placeholder)"""
        # sub_gui: SubmoduleGUI = SubmoduleGUI(f"Module {idx}")
        # sub_gui.show()
        # # Keep a reference so it doesn't get garbage collected
        # sub_gui.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        data_file = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1406731.nxs"

        run_bad_pixel_gui(data_file)


class BeamlineSelector(QWidget):
    """Initial beamline selection GUI"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Select Beamline")
        self.setGeometry(100, 100, 300, 150)

        self.waffle_gui: WaffleGUI | None = None

        layout: QVBoxLayout = QVBoxLayout()
        label: QLabel = QLabel("Select beamline:")
        self.combo: QComboBox = QComboBox()
        self.combo.addItems(BEAMLINES)

        self.confirm_btn: QPushButton = QPushButton("Confirm")
        self.confirm_btn.clicked.connect(self.confirm_selection)

        layout.addWidget(label)
        layout.addWidget(self.combo)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)

    def confirm_selection(self) -> None:
        beamline: str = self.combo.currentText()
        if beamline not in BEAMLINES:
            QMessageBox.warning(self, "Error", "Please select a valid beamline.")
            return

        # Open waffle GUI
        self.waffle_gui = WaffleGUI(beamline)
        self.waffle_gui.show()
        self.close()  # close the selector


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    selector: BeamlineSelector = BeamlineSelector()
    selector.show()
    sys.exit(app.exec())
