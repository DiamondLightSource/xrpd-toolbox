import datetime
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFileSystemModel, QKeySequence, QShortcut
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

from xrpd_toolbox.gui.fast_icons import FastIconProvider
from xrpd_toolbox.i11.mythen import MythenDataLoader
from xrpd_toolbox.utils.utils import load_int_array_from_file

CURRENT_YEAR = datetime.datetime.now().year

DEFAULT_BAD_CHANNEL_FILEPATH: str = "/dls_sw/i11/software/mythen/badchannels.txt"
DEFAULT_DATA_FOLDER: str = f"/dls/i11/data/{CURRENT_YEAR}"
try:
    CWD = Path.cwd()
except Exception:
    CWD = Path.home()
# =========================
# Plot canvas
# =========================


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        data: MythenDataLoader,
        global_selected_indices: set[int],
        selection_callback,
        parent: QWidget | None = None,
    ) -> None:
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)

        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twiny()

        self._pan_start = None

        self.current_module = 0
        self.global_selected_indices = global_selected_indices
        self.selection_callback = selection_callback

        self.bad_pixel_lines = None
        self.show_bad_pixels = True

        self._connect_events()
        self.set_data(data)

    @property
    def pixels_per_modules(self) -> int:
        # Always sourced from the loaded data — never a separate value that
        # can drift out of sync with it (that drift was the cause of bad
        # pixel markers landing away from where you actually clicked).
        return self.data.pixels_per_module

    def set_data(self, data: MythenDataLoader) -> None:
        self.data = data
        self.raw_data = data.data
        self.x = np.arange(self.pixels_per_modules)
        self.current_module = 0
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
        intensities = self.raw_data[:, start:end]

        # module_channel_number = np.arange(start, end).tolist()

        for intensity in intensities:
            self.ax.plot(self.x, intensity, alpha=0.3, linewidth=1)

        self._sync_ax2_xlim()  # 0 .. pixels_per_modules

        # choose some local tick positions (reuse ax's ticks, clipped to range)
        tick_positions = self.ax.get_xticks()
        tick_positions = tick_positions[
            (tick_positions >= 0) & (tick_positions <= self.pixels_per_modules)
        ]

        self.ax2.set_xticks(tick_positions)
        self.ax2.set_xticklabels((tick_positions + start).astype(int))
        self.ax2.set_xlabel(r"Detector Channel Number (global)")

        self.ax.set_title(f"Mythen Data — Module {module}")
        self.ax.set_xlabel("Module Pixel Channel")
        self.ax.set_ylabel("Intensity (a.u.)")

        # bad-pixel markers are (re)created fresh for this module
        self.bad_pixel_lines = None
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
            useblit=False,
            button=[1],  # type: ignore
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            props={"facecolor": "blue", "alpha": 0.2},
        )

    def set_module(self, module: int) -> None:
        self.current_module = module
        self._plot_module(module)

    def _sync_ax2_xlim(self) -> None:
        # ax2 (the twiny "global channel" axis) sits on top of ax and is
        # what matplotlib resolves mouse events against. If its x-limits
        # aren't kept in sync with ax's current (zoomed/panned) limits,
        # event.xdata on a click gets computed from ax2's stale range,
        # so clicks land on the wrong pixel whenever the view is zoomed.
        self.ax2.set_xlim(self.ax.get_xlim())

    def reset_zoom(self) -> None:
        self.ax.relim()
        self.ax.autoscale()
        self._sync_ax2_xlim()
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
        self._sync_ax2_xlim()
        self.draw_idle()

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return

        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))

        # A plain click (no real drag) fires this callback with an almost
        # zero-size box. Ignore it instead of collapsing the axes limits,
        # which triggered the "identical low and high ylims" warning and
        # zoomed the view down to a sliver that hid newly toggled pixels.
        if x0 == x1 or y0 == y1:
            return

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self._sync_ax2_xlim()
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
        self._sync_ax2_xlim()
        self.draw_idle()

    def _on_mouse_release(self, event) -> None:
        if event.button == 2:
            self._pan_start = None

    def set_bad_pixels_visible(self, visible: bool) -> None:
        self.show_bad_pixels = visible
        self._update_selected_points()

    def _update_selected_points(self) -> None:
        # Bad pixels are drawn as full-height vertical lines using the
        # x-axis transform (x in data coords, y in axes-fraction coords),
        # so they always span the plot regardless of intensity values or
        # the current zoom/pan state.
        if self.bad_pixel_lines is not None:
            self.bad_pixel_lines.remove()
            self.bad_pixel_lines = None

        if not self.show_bad_pixels:
            self.draw_idle()
            return

        start = self.current_module * self.pixels_per_modules
        end = start + self.pixels_per_modules

        local = [
            idx - start for idx in self.global_selected_indices if start <= idx < end
        ]

        if local:
            self.bad_pixel_lines = self.ax.vlines(
                local,
                0,
                1,
                transform=self.ax.get_xaxis_transform(),
                colors="red",
                linewidth=1,
                alpha=0.6,
                zorder=5,
            )

        self.draw_idle()


# =========================
# Main window
# =========================

EMPTY_BAD_CHANNELS: set[int] = set()


class BadModuleMainWindow(QMainWindow):
    def __init__(
        self,
        data: MythenDataLoader | None = None,
        bad_channels: set[int] = EMPTY_BAD_CHANNELS,
    ) -> None:
        super().__init__()

        self.setWindowTitle("Mythen NXS Viewer")

        if data is None:
            filepath = self.nexus_file_dialog()

            data = MythenDataLoader(filepath)

        else:
            data = data

        self.data = data
        self.global_selected_indices = bad_channels
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

        self.file_label = QLabel(f"File: {self.data.filepath}")
        self.status = self.statusBar()
        if self.status is not None:
            self.status.addPermanentWidget(self.file_label)

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

        self.toggle_bad_pixels_button = QPushButton("Hide Bad Pixels", self.canvas)
        self.toggle_bad_pixels_button.setCheckable(True)
        self.toggle_bad_pixels_button.toggled.connect(self._on_toggle_bad_pixels)
        self.toggle_bad_pixels_button.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(255, 255, 255, 200);"
            "  border: 1px solid #888;"
            "  border-radius: 4px;"
            "  padding: 3px 8px;"
            "}"
            "QPushButton:checked {"
            "  background-color: rgba(220, 220, 220, 220);"
            "}"
        )
        self.toggle_bad_pixels_button.adjustSize()
        self.toggle_bad_pixels_button.move(8, 8)
        self.toggle_bad_pixels_button.raise_()

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

    def nexus_file_dialog(self) -> str:
        data_folder = Path(DEFAULT_DATA_FOLDER).parent

        if data_folder.exists() and data_folder.is_dir():
            folder = str(data_folder)
        else:
            folder = str(CWD)

        dialog = QFileDialog(
            self, "Mythen Nexus File", folder, "NeXus Files (*.nxs *.nexus)"
        )
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

        # Must use the Qt (non-native) dialog for the performance options
        # below to have any effect at all — the native OS dialog doesn't
        # go through QFileSystemModel/QFileIconProvider.
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.ViewMode.List)
        dialog.setIconProvider(FastIconProvider())

        model = dialog.findChild(QFileSystemModel)
        if model is not None:
            model.setOption(QFileSystemModel.Option.DontWatchForChanges, True)
            model.setOption(QFileSystemModel.Option.DontResolveSymlinks, True)
            model.setOption(QFileSystemModel.Option.DontUseCustomDirectoryIcons, True)

        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return ""

        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def load_nexus_file_from_dialog(self) -> None:

        filepath = self.nexus_file_dialog()

        if filepath:
            self.load_nexus_file(filepath)

    def load_nexus_file(self, filepath: str) -> None:

        try:
            self.data = MythenDataLoader(filepath)
            self.canvas.set_data(self.data)
            self.module_slider.setRange(0, self.data.n_modules_in_data - 1)
            self._update_bad_channel_canvas()
            self.file_label.setText(f"File: {self.data.filepath}")

        except ValueError as e:
            QMessageBox.information(self, title="error", text=f"{e}")

    def load_initial_bad_channels(self) -> None:

        bad_channel_folder = Path(DEFAULT_BAD_CHANNEL_FILEPATH).parent

        if bad_channel_folder.exists() and bad_channel_folder.is_dir():
            folder = str(bad_channel_folder)
        else:
            folder = str(CWD)

        bad_channels_file, _ = QFileDialog.getOpenFileName(
            self, "Bad Channel File", folder, "Text Files (*.txt)"
        )
        if not bad_channels_file:
            return

        try:
            new_global_selected_indices = set(
                load_int_array_from_file(bad_channels_file)
            )
            self.global_selected_indices.update(new_global_selected_indices)

        except Exception as e:
            print(f"Error occurred while loading initial indices: {e}")
            pass

        self._update_bad_channel_canvas()

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

        file_menu.addAction("Load Nexus File", self.load_nexus_file_from_dialog)

        file_menu.addAction("Save", self._save)
        file_menu.addAction("Save As...", self._save_as)
        file_menu.addAction("Load Bad Channels...", self.load_initial_bad_channels)

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

    def _update_bad_channel_canvas(self) -> None:
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

        else:
            reply = QMessageBox.question(
                self,
                "Save to badchannels file?",
                "This will overwrite the existing badchannels file. Are you sure?",
            )

            if reply != QMessageBox.StandardButton.Yes:
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

    def _on_toggle_bad_pixels(self, hidden: bool) -> None:
        self.canvas.set_bad_pixels_visible(not hidden)
        self.toggle_bad_pixels_button.setText(
            "Show Bad Pixels" if hidden else "Hide Bad Pixels"
        )
        self.toggle_bad_pixels_button.adjustSize()


# =========================
# Entrypoint
# =========================
def run_bad_pixel_gui(
    filepath: str | None = None,
    bad_channel_file: str | None = None,
) -> None:

    if filepath is not None:
        data = MythenDataLoader(filepath)
    else:
        data = None

    if bad_channel_file:
        initial_indices = set(load_int_array_from_file(bad_channel_file))
    else:
        try:
            initial_indices = set(
                load_int_array_from_file(DEFAULT_BAD_CHANNEL_FILEPATH)
            )
        except Exception as e:
            print(f"Error occurred while loading initial indices: {e}")
            initial_indices = set()

    app = QApplication(sys.argv)
    win = BadModuleMainWindow(data, initial_indices)
    win.resize(1500, 900)
    win.show()
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)  # optional
    sys.exit(app.exec())


if __name__ == "__main__":
    DATA_FILE = "/workspaces/outputs/step_scan/1410286.nxs"

    run_bad_pixel_gui(
        DATA_FILE,
    )
