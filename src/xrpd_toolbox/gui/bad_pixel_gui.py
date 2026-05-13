import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from xrpd_toolbox.i11.mythen import MythenDataLoader
from xrpd_toolbox.utils.utils import load_int_array_from_file

# =========================
# Plot canvas
# =========================


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        data: MythenDataLoader,
        global_selected_indices: set[int],
        selection_callback,
        pixels_per_modules: int = 1280,
        parent: QWidget | None = None,
    ) -> None:
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)

        self.ax = self.figure.add_subplot(111)
        self._pan_start = None

        self.data = data
        self.raw_data = data.data
        self.pixels_per_modules = pixels_per_modules

        self.current_module = 0
        self.global_selected_indices = global_selected_indices
        self.selection_callback = selection_callback

        self.x = np.arange(self.pixels_per_modules)
        self.scatter = None
        self.reference_curve = None

        self._connect_events()
        self._plot_module(0)

    def _connect_events(self) -> None:
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("button_release_event", self._on_mouse_release)

    def _plot_module(self, module: int) -> None:
        self.ax.cla()

        start = module * self.pixels_per_modules
        end = start + self.pixels_per_modules
        curves = self.raw_data[:, start:end]

        for curve in curves:
            self.ax.plot(self.x, curve, alpha=0.3, linewidth=1)

        self.reference_curve = curves[0]

        self.ax.set_title(f"Mythen Data — Module {module}")
        self.ax.set_xlabel("Pixel (within module)")
        self.ax.set_ylabel("Intensity (a.u.)")

        self.scatter = self.ax.scatter([], [], color="red", zorder=5)
        self._update_selected_points()

        self._enable_rectangle_zoom()
        self.ax.relim()
        self.ax.autoscale()
        self.draw_idle()

    def _enable_rectangle_zoom(self) -> None:
        if hasattr(self, "rect_selector"):
            self.rect_selector.disconnect_events()

        self.rect_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1],  # type: ignore
            interactive=False,
            props={"facecolor": "blue", "alpha": 0.2},
        )

    def set_module(self, module: int) -> None:
        self.current_module = module
        self._plot_module(module)

    def reset_zoom(self) -> None:
        self.ax.relim()
        self.ax.autoscale()
        self.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return

        zoom = 1.05
        factor = 1 / zoom if event.button == "up" else zoom

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        self.ax.set_xlim(
            event.xdata - (x1 - x0) * factor / 2,
            event.xdata + (x1 - x0) * factor / 2,
        )
        self.ax.set_ylim(
            event.ydata - (y1 - y0) * factor / 2,
            event.ydata + (y1 - y0) * factor / 2,
        )
        self.draw_idle()

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        self.ax.set_xlim(sorted((eclick.xdata, erelease.xdata)))  # type: ignore
        self.ax.set_ylim(sorted((eclick.ydata, erelease.ydata)))  # type: ignore
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.button == 3 and event.xdata is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < self.data.pixels_per_module:
                global_idx = self.current_module * self.data.pixels_per_module + idx
                self.selection_callback(global_idx)

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

        x0, x1 = self._pan_start["xlim"]
        y0, y1 = self._pan_start["ylim"]

        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self.draw_idle()

    def _on_mouse_release(self, event) -> None:
        if event.button == 2:
            self._pan_start = None

    def _update_selected_points(self) -> None:
        if self.scatter is None or self.reference_curve is None:
            return

        start = self.current_module * self.pixels_per_modules
        end = start + self.pixels_per_modules

        local = [
            idx - start for idx in self.global_selected_indices if start <= idx < end
        ]

        if local:
            x = self.x[local]
            y = self.reference_curve[local]
            self.scatter.set_offsets(np.column_stack((x, y)))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

        self.draw_idle()


# =========================
# Main window
# =========================


class MainWindow(QMainWindow):
    def __init__(self, data: MythenDataLoader, initial_indices: set[int]) -> None:
        super().__init__()

        self.setWindowTitle("Mythen NXS Viewer")

        self.data = data
        self.global_selected_indices = initial_indices
        self._current_save_path: Path | None = None

        # undo / redo
        self.undo_stack: list[set[int]] = []
        self.redo_stack: list[set[int]] = []
        self.undo_limit = 20

        self.canvas = PlotCanvas(
            self.data,
            self.global_selected_indices,
            self._toggle_index,
        )

        self.list_widget = QListWidget()

        self.module_bad_pixels_box = QTextEdit()
        self.module_bad_pixels_box.setReadOnly(True)

        self.module_count_label = QLabel()
        self.global_count_label = QLabel()

        self.module_slider = QSlider(Qt.Orientation.Horizontal)
        self.module_slider.setRange(0, self.data.n_modules_in_data - 1)
        self.module_slider.valueChanged.connect(self._on_module_changed)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.canvas.reset_zoom)

        self.save_button = QPushButton("Save Selected Indices")
        self.save_button.clicked.connect(self._save)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, self.data.pixels_per_module // 2)
        self.n_spin.setValue(5)

        self.add_edges_button = QPushButton("Add First/Last N per Module")
        self.add_edges_button.clicked.connect(self._add_edge_indices)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)  # type: ignore
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)  # type: ignore

        self._setup_layout()
        self._setup_menu()
        self._sync_all()

    # ---------- UI ----------

    def _setup_layout(self) -> None:
        right = QVBoxLayout()
        right.addWidget(QLabel("Selected Global Indices"))
        right.addWidget(self.list_widget)

        right.addWidget(QLabel("Bad Pixels (Current Module, Global Indices)"))
        right.addWidget(self.module_bad_pixels_box)
        right.addWidget(self.module_count_label)
        right.addWidget(self.global_count_label)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("N edge points per module"))
        edge_row.addWidget(self.n_spin)
        edge_row.addWidget(self.add_edges_button)
        edge_row.addStretch()
        right.addLayout(edge_row)

        undo_row = QHBoxLayout()
        undo_row.addWidget(self.undo_button)
        undo_row.addWidget(self.redo_button)
        undo_row.addStretch()
        right.addLayout(undo_row)

        right.addWidget(self.save_button)

        left = QVBoxLayout()
        left.addWidget(self.canvas)
        left.addWidget(self.module_slider)
        left.addWidget(self.reset_zoom_button)

        central = QWidget()
        main = QHBoxLayout(central)
        main.addLayout(left, 3)
        main.addLayout(right, 1)

        self.setCentralWidget(central)

    def _setup_menu(self) -> None:
        menu = self.menuBar()

        if menu is None:
            raise Exception("Menu has broken")
        file_menu = menu.addMenu("File")
        if file_menu is None:
            raise Exception("file_menu has broken")
        file_menu.addAction("Save", self._save)
        file_menu.addAction("Save As...", self._save_as)

        help_menu = menu.addMenu("Help")
        if help_menu is None:
            raise Exception("help_menu has broken")

        help_menu.addAction("Controls", self._show_controls)

    # ---------- syncing ----------

    def _sync_all(self) -> None:
        self.list_widget.clear()
        for idx in sorted(self.global_selected_indices):
            self.list_widget.addItem(QListWidgetItem(str(idx)))

        self._update_module_bad_pixels()
        self.global_count_label.setText(
            f"Total bad pixels (global): {len(self.global_selected_indices)}"
        )

    def _update_module_bad_pixels(self) -> None:
        m = self.canvas.current_module
        ppm = self.data.pixels_per_module
        start = m * ppm
        end = start + ppm

        module_globals = sorted(
            idx for idx in self.global_selected_indices if start <= idx < end
        )

        self.module_bad_pixels_box.setPlainText(
            ", ".join(map(str, module_globals)) if module_globals else "(none)"
        )
        self.module_count_label.setText(f"Bad pixels in module: {len(module_globals)}")

    # ---------- undo / redo ----------

    def _record_state(self) -> None:
        self.undo_stack.append(self.global_selected_indices.copy())
        if len(self.undo_stack) > self.undo_limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _undo(self) -> None:
        if not self.undo_stack:
            return

        self.redo_stack.append(self.global_selected_indices.copy())
        prev = self.undo_stack.pop()

        self.global_selected_indices.clear()
        self.global_selected_indices.update(prev)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _redo(self) -> None:
        if not self.redo_stack:
            return

        self.undo_stack.append(self.global_selected_indices.copy())
        nxt = self.redo_stack.pop()

        self.global_selected_indices.clear()
        self.global_selected_indices.update(nxt)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    # ---------- actions ----------

    def _toggle_index(self, idx: int) -> None:
        self._record_state()

        if idx in self.global_selected_indices:
            self.global_selected_indices.remove(idx)
        else:
            self.global_selected_indices.add(idx)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _add_edge_indices(self) -> None:
        self._record_state()

        n = self.n_spin.value()
        for m in range(self.data.n_modules_in_data):
            base = m * self.data.pixels_per_module
            for i in range(n):
                self.global_selected_indices.add(base + i)
                self.global_selected_indices.add(
                    base + self.data.pixels_per_module - 1 - i
                )

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    # ---------- save ----------

    def _save(self) -> None:
        if self._current_save_path is None:
            self._save_as()
            return

        with self._current_save_path.open("w", encoding="utf-8") as f:
            for idx in sorted(self.global_selected_indices):
                f.write(f"{idx}\n")

    def _save_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Indices", "", "Text Files (*.txt)"
        )
        if not path_str:
            return

        self._current_save_path = Path(path_str)
        self._save()

    # ---------- help ----------

    def _show_controls(self) -> None:
        QMessageBox.information(
            self,
            "Controls",
            "Right click: toggle bad pixel\n"
            "Middle click + drag: pan\n"
            "Scroll wheel: zoom\n"
            "Left click + drag: rectangle zoom\n"
            "Slider: change module\n"
            "Ctrl+Z / Ctrl+Y: undo / redo",
        )

    def _on_module_changed(self, module: int) -> None:
        self.canvas.set_module(module)
        self._update_module_bad_pixels()


# =========================
# Entrypoint
# =========================


def run_bad_pixel_gui(filepath: str, indices_file: str | None = None) -> None:
    data = MythenDataLoader(filepath)

    if indices_file:
        initial_indices = set(load_int_array_from_file(indices_file))
    else:
        initial_indices = set()

    app = QApplication(sys.argv)
    win = MainWindow(data, initial_indices)
    win.resize(1500, 900)
    win.show()
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)  # optional
    sys.exit(app.exec())


if __name__ == "__main__":
    DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1410286.nxs"

    run_bad_pixel_gui(
        DATA_FILE,
    )
