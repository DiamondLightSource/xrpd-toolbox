import datetime
import shutil
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
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
BAD_CHANNEL_BACKUP_FOLDER: str = "/dls_sw/i11/software/mythen/badchannels"
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

        self.current_module = 0

        self.ax = self.figure.add_subplot(111)

        # ax2 (the "global channel" axis on top) is created fresh inside
        # _plot_module every time, not here. It's a secondary axis (NOT
        # ax.twiny()): twiny() makes a second, fully independent Axes
        # stacked on top of ax, which both steals mouse events from ax
        # (clicks/drags resolve against whichever Axes is topmost) and is
        # invisible to the navigation toolbar's pan/zoom, which only ever
        # acts on the Axes that actually holds the data. secondary_xaxis
        # avoids both problems — but it only stays linked to `ax` until
        # the next `ax.cla()`, which severs that link entirely (not just
        # the tick cache). Since every module switch calls `ax.cla()`,
        # ax2 has to be rebuilt after each one, or the global-channel
        # numbers freeze/break after the very first draw.
        self.ax2 = None

        self.global_selected_indices = global_selected_indices
        self.selection_callback = selection_callback

        self.bad_pixel_lines = None
        self.show_bad_pixels = True

        # Set by the owning window once the toolbar exists, so click
        # handling can check whether Pan/Zoom mode is currently active.
        self.toolbar: NavigationToolbar2QT | None = None

        self._connect_events()
        self.set_data(data)

    @property
    def pixels_per_modules(self) -> int:
        # Always sourced from the loaded data — never a separate value that
        # can drift out of sync with it (that drift was the cause of bad
        # pixel markers landing away from where you actually clicked).
        return self.data.pixels_per_module

    def _module_to_global(self, x):
        start = self.current_module * self.pixels_per_modules
        return np.asarray(x) + start

    def _global_to_module(self, x):
        start = self.current_module * self.pixels_per_modules
        return np.asarray(x) - start

    def set_data(self, data: MythenDataLoader) -> None:
        self.data = data
        self.raw_data = data.data
        self.x = np.arange(self.pixels_per_modules)
        self.current_module = 0
        self._plot_module(0)

    def _connect_events(self) -> None:
        self.mpl_connect("button_press_event", self._on_click)

    def _plot_module(self, module: int) -> None:
        self.ax.cla()

        # Rebuild the secondary "global channel" axis immediately after
        # cla() — cla() severs the link a secondary axis needs to stay
        # synced with ax, so it must be recreated on every module switch
        # (see the note on self.ax2 = None in __init__).
        self.ax2 = self.ax.secondary_xaxis(
            "top", functions=(self._module_to_global, self._global_to_module)
        )
        self.ax2.set_xlabel("Detector Channel Number (global)")

        start = module * self.pixels_per_modules
        end = start + self.pixels_per_modules
        intensities = self.raw_data[:, start:end]

        for intensity in intensities:
            self.ax.plot(self.x, intensity, alpha=0.3, linewidth=1)

        self.ax.set_title(f"Mythen Data — Module {module}")
        self.ax.set_xlabel("Module Pixel Channel")
        self.ax.set_ylabel("Intensity (a.u.)")

        # bad-pixel markers are (re)created fresh for this module
        self.bad_pixel_lines = None
        self._update_selected_points()

        self.ax.relim()
        self.ax.autoscale()
        self.draw_idle()

        # Reset the toolbar's Home/Back/Forward history so "Home" points
        # at this module's freshly autoscaled view, not the previous
        # module's. Without this, Home after switching modules would
        # jump back to wherever the old module happened to be zoomed.
        if self.toolbar is not None:
            self.toolbar.update()

    def set_module(self, module: int) -> None:
        self.current_module = module
        self._plot_module(module)

    def _on_click(self, event) -> None:
        # A stationary right click (press+release with no movement) is
        # a no-op for the toolbar's Pan/Zoom tools — they only act on a
        # drag — so toggling a bad pixel here never conflicts with them,
        # even while Pan or Zoom mode is switched on.
        if event.button == 3 and event.xdata is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < self.data.pixels_per_module:
                global_idx = self.current_module * self.data.pixels_per_module + idx
                self.selection_callback(global_idx)

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
        bad_channels_save_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.setWindowTitle("Mythen Bad Pixel GUI")

        if data is None:
            filepath = self.nexus_file_dialog()

            data = MythenDataLoader(filepath)

        else:
            data = data

        self.data = data
        self.global_bad_channels = bad_channels

        if bad_channels_save_path is not None:
            bad_channels_save_path = Path(bad_channels_save_path)

        self._current_save_path: Path | None = bad_channels_save_path

        # undo / redo
        self.undo_stack: list[set[int]] = []
        self.redo_stack: list[set[int]] = []
        self.undo_limit = 20

        self.canvas = PlotCanvas(
            self.data,
            self.global_bad_channels,
            self._toggle_index,
        )

        # Standard matplotlib navigation toolbar: Home, Back, Forward,
        # Pan, Zoom-to-rect, Configure Subplots, Save. Pan/Zoom now work
        # correctly because PlotCanvas uses secondary_xaxis (not
        # twiny()) for the global-channel axis, so there's no second
        # overlapping Axes for the toolbar or mouse events to get
        # confused by.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.toolbar = self.toolbar

        self.file_label = QLabel(f"File: {self.data.filepath}")
        self.save_path_label = QLabel()
        self._update_save_path_label()

        # QStatusBar arranges permanent widgets left-to-right, so to get
        # the two labels stacked on top of each other we group them in
        # our own container with a vertical layout and add that single
        # container as the (one) permanent widget instead.
        status_labels = QWidget()
        status_labels_layout = QVBoxLayout(status_labels)
        status_labels_layout.setContentsMargins(0, 0, 0, 0)
        status_labels_layout.setSpacing(0)
        status_labels_layout.addWidget(self.file_label)
        status_labels_layout.addWidget(self.save_path_label)

        self.status = self.statusBar()
        if self.status is not None:
            self.status.addPermanentWidget(status_labels)

        self.list_widget = QListWidget()

        self.module_bad_pixels_box = QTextEdit()
        self.module_bad_pixels_box.setReadOnly(True)

        self.module_count_label = QLabel()
        self.global_count_label = QLabel()

        self.module_slider = QSlider(Qt.Orientation.Horizontal)
        self.module_slider.setRange(0, self.data.n_modules_in_data - 1)
        self.module_slider.valueChanged.connect(self._on_module_changed)

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

        # Start with the toolbar's Zoom tool active, as if the user had
        # already clicked it. This is deferred with a 0ms QTimer rather
        # than called directly here: at this point in __init__ the
        # window hasn't been shown yet and the canvas hasn't done its
        # first real paint/resize, so the zoom tool's mouse-event
        # handling doesn't fully engage until something (e.g. a click)
        # forces Qt to finish that first paint. Scheduling it to run
        # right after the event loop starts sidesteps that entirely.
        QTimer.singleShot(0, self.toolbar.zoom)

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

    def load_bad_channels(self) -> None:

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
            self.global_bad_channels.update(new_global_selected_indices)

        except Exception as e:
            print(f"Error occurred while loading bad channels: {e}")
            pass

        self._update_bad_channel_canvas()

    def _update_save_path_label(self) -> None:
        if self._current_save_path is None:
            self.save_path_label.setText("Save path: (not set — will prompt on Save)")
        else:
            self.save_path_label.setText(f"Save path: {self._current_save_path}")

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
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)
        left.addWidget(self.module_slider)

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
        file_menu.addAction("Load Bad Channels...", self.load_bad_channels)

        help_menu = menu.addMenu("Help")
        if help_menu is None:
            raise Exception("help_menu has broken")

        help_menu.addAction("Controls", self._show_controls)

    # ---------- syncing ----------

    def _sync_all(self) -> None:
        self.list_widget.clear()
        for idx in sorted(self.global_bad_channels):
            self.list_widget.addItem(QListWidgetItem(str(idx)))

        self._update_module_bad_pixels()
        self.global_count_label.setText(
            f"Total bad pixels (global): {len(self.global_bad_channels)}"
        )

    def _update_module_bad_pixels(self) -> None:
        m = self.canvas.current_module
        ppm = self.data.pixels_per_module
        start = m * ppm
        end = start + ppm

        module_globals = sorted(
            idx for idx in self.global_bad_channels if start <= idx < end
        )

        self.module_bad_pixels_box.setPlainText(
            ", ".join(map(str, module_globals)) if module_globals else "(none)"
        )
        self.module_count_label.setText(f"Bad pixels in module: {len(module_globals)}")

    # ---------- undo / redo ----------

    def _record_state(self) -> None:
        self.undo_stack.append(self.global_bad_channels.copy())
        if len(self.undo_stack) > self.undo_limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _undo(self) -> None:
        if not self.undo_stack:
            return

        self.redo_stack.append(self.global_bad_channels.copy())
        prev = self.undo_stack.pop()

        self.global_bad_channels.clear()
        self.global_bad_channels.update(prev)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _update_bad_channel_canvas(self) -> None:
        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _redo(self) -> None:
        if not self.redo_stack:
            return

        self.undo_stack.append(self.global_bad_channels.copy())
        nxt = self.redo_stack.pop()

        self.global_bad_channels.clear()
        self.global_bad_channels.update(nxt)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    # ---------- actions ----------

    def _toggle_index(self, idx: int) -> None:
        self._record_state()

        if idx in self.global_bad_channels:
            self.global_bad_channels.remove(idx)
        else:
            self.global_bad_channels.add(idx)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _add_edge_indices(self) -> None:
        self._record_state()

        n = self.n_spin.value()
        for m in range(self.data.n_modules_in_data):
            base = m * self.data.pixels_per_module
            for i in range(n):
                self.global_bad_channels.add(base + i)
                self.global_bad_channels.add(base + self.data.pixels_per_module - 1 - i)

        self.canvas._update_selected_points()  # noqa
        self._sync_all()

    def _backup(self):

        if (
            (self._current_save_path is not None)
            and self._current_save_path.exists()
            and (str(self._current_save_path) == DEFAULT_BAD_CHANNEL_FILEPATH)
        ):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = Path(BAD_CHANNEL_BACKUP_FOLDER)
            backup_name = f"{self._current_save_path.stem}_{timestamp}{self._current_save_path.suffix}"  # noqa
            backup_path = backup_folder / backup_name

            backup_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy2(self._current_save_path, backup_path)

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

            if reply == QMessageBox.StandardButton.Yes:
                # Try to back up the file before overwriting,
                # if file exists and is the default
                self._backup()

                with self._current_save_path.open("w", encoding="utf-8") as f:
                    for idx in sorted(self.global_bad_channels):
                        f.write(f"{idx}\n")

    def _save_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Indices", "", "Text Files (*.txt)"
        )
        if not path_str:
            return

        self._current_save_path = Path(path_str)
        self._update_save_path_label()
        self._save()

    # ---------- help ----------

    def _show_controls(self) -> None:
        QMessageBox.information(
            self,
            "Controls",
            "Toolbar: Home / Back / Forward, Pan, Zoom-to-rect, Save\n"
            "Right click: toggle bad pixel (only while Pan/Zoom tool is off)\n"
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
            print("Using default badchannels:", DEFAULT_BAD_CHANNEL_FILEPATH)
            initial_indices = set(
                load_int_array_from_file(DEFAULT_BAD_CHANNEL_FILEPATH)
            )

            bad_channel_file = DEFAULT_BAD_CHANNEL_FILEPATH

        except Exception as e:
            print(f"Error occurred while loading initial indices on startup: {e}")
            initial_indices = set()

    app = QApplication(sys.argv)
    win = BadModuleMainWindow(
        data=data, bad_channels=initial_indices, bad_channels_save_path=bad_channel_file
    )
    win.resize(1500, 900)
    win.show()
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)  # optional
    sys.exit(app.exec())


if __name__ == "__main__":
    DATA_FILE = "/workspaces/outputs/step_scan/1410286.nxs"

    run_bad_pixel_gui(
        DATA_FILE,
    )
