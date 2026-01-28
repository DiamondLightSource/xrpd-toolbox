import sys
from pathlib import Path
from typing import List, Optional, Set

import h5py
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QWidget,
    QLabel,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from xrpd_toolbox.utils.utils import load_int_array_from_file

DATASET_PATH = "/entry/mythen_nx/data"
MODULE_COUNT = 28
MODULE_SIZE = 1280
UNDO_LIMIT = 10
COUNTER = 0


def load_mythen_data(filepath: str) -> np.ndarray:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as file:
        if DATASET_PATH not in file:
            raise KeyError(f"Dataset not found: {DATASET_PATH}")
        
        data = np.asarray(file.get(DATASET_PATH, []))

    if data.ndim < 1:
        raise ValueError("Dataset must have at least one dimension")
    else:
        data = data[:, :, COUNTER]

    return data


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        data: np.ndarray,
        global_selected_indices: Set[int],
        selection_callback,
        parent: Optional[QWidget] = None,
    ) -> None:
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)

        self._pan_start = None  # new attribute


        self.ax = self.figure.add_subplot(111)

        self.raw_data = data
        self.current_module = 0
        self.global_selected_indices = global_selected_indices
        self.selection_callback = selection_callback

        self.x = np.arange(MODULE_SIZE)
        self.scatter = None
        self.reference_curve = None

        self._connect_events()
        self._plot_module(0)
        self._enable_rectangle_zoom()

    def _connect_events(self) -> None:
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("button_release_event", self._on_mouse_release)

    def _plot_module(self, module: int) -> None:
        self.ax.cla()

        start = module * MODULE_SIZE
        end = start + MODULE_SIZE

        curves = self.raw_data[:, start:end]

        for curve in curves:
            self.ax.plot(self.x, curve, alpha=0.3, linewidth=1)

        self.reference_curve = curves[0]

        self.ax.set_title(f"Mythen Data — Module {module}")
        self.ax.set_xlabel("Pixel (within module)")
        self.ax.set_ylabel("Intensity (Arb. Units.)")

        self.ax.relim()
        self.ax.autoscale()

        self.scatter = self.ax.scatter([], [], color="red", zorder=5)
        self._update_selected_points()

        self._enable_rectangle_zoom()
        self.draw_idle()

    def _enable_rectangle_zoom(self) -> None:
        if hasattr(self, "rect_selector"):
            self.rect_selector.disconnect_events()

        self.rect_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1], #type: ignore
            interactive=False,
            props=dict(facecolor="blue", alpha=0.2),
        )

    def reset_zoom(self) -> None:
        self.ax.relim()
        self.ax.autoscale()
        self.draw_idle()

    def set_module(self, module: int) -> None:
        self.current_module = module
        self._plot_module(module)

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return

        zoom_factor = 1.05
        factor = 1 / zoom_factor if event.button == "up" else zoom_factor

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        x_range = (x_max - x_min) * factor
        y_range = (y_max - y_min) * factor

        self.ax.set_xlim(event.xdata - x_range / 2, event.xdata + x_range / 2)
        self.ax.set_ylim(event.ydata - y_range / 2, event.ydata + y_range / 2)
        self.draw_idle()

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return

        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.draw_idle()


    def _on_click(self, event) -> None:
        # Right-click toggle
        if event.button == 3 and event.xdata is not None:
            index = int(round(event.xdata))
            if 0 <= index < MODULE_SIZE:
                global_index = self.current_module * MODULE_SIZE + index
                self.selection_callback(global_index)
        # Middle-button press starts panning
        elif event.button == 2 and event.xdata is not None and event.ydata is not None:
            self._pan_start = {
                "x": event.xdata,
                "y": event.ydata,
                "xlim": self.ax.get_xlim(),
                "ylim": self.ax.get_ylim(),
            }

    def _on_mouse_move(self, event) -> None:
        if self._pan_start is None or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._pan_start["x"]
        dy = event.ydata - self._pan_start["y"]
        xlim_start, xlim_end = self._pan_start["xlim"]
        ylim_start, ylim_end = self._pan_start["ylim"]
        self.ax.set_xlim(xlim_start - dx, xlim_end - dx)
        self.ax.set_ylim(ylim_start - dy, ylim_end - dy)
        self.draw_idle()

    def _on_mouse_release(self, event) -> None:
        if event.button == 2:
            self._pan_start = None

    def _update_selected_points(self) -> None:
        if self.scatter is None or self.reference_curve is None:
            return

        module_start = self.current_module * MODULE_SIZE
        module_end = module_start + MODULE_SIZE

        local_indices = [
            idx - module_start
            for idx in self.global_selected_indices
            if module_start <= idx < module_end
        ]

        if local_indices:
            x = self.x[local_indices]
            y = self.reference_curve[local_indices]
            offsets = np.column_stack((x, y))
        else:
            offsets = np.empty((0, 2))

        self.scatter.set_offsets(offsets)
        self.draw_idle()


class MainWindow(QWidget):
    def __init__(self, data: np.ndarray, initial_indices: Set[int]) -> None:
        super().__init__()

        self.setWindowTitle("Mythen NXS Viewer")

        self.global_selected_indices = initial_indices

        self.undo_stack: List[Set[int]] = []
        self.redo_stack: List[Set[int]] = []

        self.canvas = PlotCanvas(
            data, self.global_selected_indices, self._toggle_index
        )
        self.list_widget = QListWidget()

        self.module_slider = QSlider(Qt.Orientation.Horizontal)
        self.module_slider.setRange(0, MODULE_COUNT - 1)
        self.module_slider.valueChanged.connect(self.canvas.set_module)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.canvas.reset_zoom)

        self.save_button = QPushButton("Save Selected Indices")
        self.save_button.clicked.connect(self._save_indices)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, MODULE_SIZE // 2)
        self.n_spin.setValue(5)

        self.add_edges_button = QPushButton("Add First/Last N per Module")
        self.add_edges_button.clicked.connect(self._add_edge_indices)

        self._setup_layout()
        self._timer_id = self.startTimer(200)

    def _setup_layout(self) -> None:
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Selected Global Indices"))
        right_layout.addWidget(self.list_widget)
        right_layout.addWidget(QLabel("N (edge points per module)"))
        right_layout.addWidget(self.n_spin)
        right_layout.addWidget(self.add_edges_button)

        # Undo/Redo buttons side by side
        undo_redo_layout = QHBoxLayout()
        undo_redo_layout.addWidget(self.undo_button)
        undo_redo_layout.addWidget(self.redo_button)
        right_layout.addLayout(undo_redo_layout)

        right_layout.addWidget(self.save_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.module_slider)
        left_layout.addWidget(self.reset_zoom_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        self.setLayout(main_layout)

    def timerEvent(self, event) -> None: #type: ignore
        self._sync_list()

    def _sync_list(self) -> None:
        indices = sorted(self.global_selected_indices)
        if self.list_widget.count() == len(indices):
            return

        self.list_widget.clear()
        for idx in indices:
            self.list_widget.addItem(QListWidgetItem(str(idx)))

    def _record_state(self) -> None:
        self.undo_stack.append(self.global_selected_indices.copy())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _toggle_index(self, idx: int) -> None:
        self._record_state()
        if idx in self.global_selected_indices:
            self.global_selected_indices.remove(idx)
        else:
            self.global_selected_indices.add(idx)
        self.canvas._update_selected_points()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self.global_selected_indices.copy())
        prev_state = self.undo_stack.pop()
        self.global_selected_indices.clear()
        self.global_selected_indices.update(prev_state)
        self.canvas._update_selected_points()

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self.global_selected_indices.copy())
        next_state = self.redo_stack.pop()
        self.global_selected_indices.clear()
        self.global_selected_indices.update(next_state)
        self.canvas._update_selected_points()

    def _add_edge_indices(self) -> None:
        self._record_state()
        n = self.n_spin.value()
        for module in range(MODULE_COUNT):
            base = module * MODULE_SIZE
            for i in range(n):
                self.global_selected_indices.add(base + i)
                self.global_selected_indices.add(base + MODULE_SIZE - 1 - i)
        self.canvas._update_selected_points()

    def _save_indices(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Indices",
            "",
            "Text Files (*.txt)",
        )
        if not path_str:
            return
        path = Path(path_str)
        with path.open("w", encoding="utf-8") as f:
            for idx in sorted(self.global_selected_indices):
                f.write(f"{idx}\n")


def run_gui(filepath: str, indices_file: Optional[str] = None) -> None:
    data = load_mythen_data(filepath)
    initial_indices: Set[int] = set()
    if indices_file is not None:
        initial_indices = set(load_int_array_from_file(indices_file))

    app = QApplication(sys.argv)
    window = MainWindow(data, initial_indices)
    window.resize(1450, 900)
    window.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    # Example usage:
    run_gui("/Users/akz63626/cm44155-1/1407178.nxs", indices_file="/Users/akz63626/cm44155-1/combined_bad_channels.txt")
