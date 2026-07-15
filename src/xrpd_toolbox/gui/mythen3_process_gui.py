import datetime
import sys
from pathlib import Path
from types import UnionType
from typing import Any, Literal, cast, get_args, get_origin

from matplotlib.backends.backend_qt import (
    NavigationToolbar2QT,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from pydantic import ValidationError
from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFileSystemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.gui.fast_icons import FastIconProvider
from xrpd_toolbox.i11.mythen import MythenDetector, MythenSettings

# from xrpd_toolbox.i11.mythen3_reduction_legacy import I11Reduction

CURRENT_YEAR: int = datetime.datetime.now().year
DEFAULT_DATA_FOLDER: str = "/dls/i11/data"
ROOT: str = str(Path.root)
# ROOT = "/workspaces/outputs"

# Qt.ItemDataRole.UserRole is used to store the input Path on a list item;
# UserRole + 1 stores the .xye output Path once processing finishes
# (or None until then).
OUTPUT_PATH_ROLE: int = int(Qt.ItemDataRole.UserRole) + 1


# =========================
# Worker Thread
# =========================


class ProcessingWorker(QThread):
    file_started = pyqtSignal(Path)
    # Emits (input .nxs path, output .xye path) once a file is done.
    file_finished = pyqtSignal(Path, Path)
    file_failed = pyqtSignal(Path, str)

    def __init__(
        self,
        files: list[Path],
        settings: MythenSettings,
        output_dir: str | None = None,
        suffix: str = "",
    ) -> None:
        super().__init__()
        self.files: list[Path] = files
        self.settings: MythenSettings = settings
        self.output_dir: str | None = output_dir
        self.suffix: str = suffix

        print(self.settings)

    def run(self) -> None:
        for file in self.files:
            print(file)

            self.file_started.emit(file)
            try:
                output_path = self.process_file(file)
                self.file_finished.emit(file, output_path)
            except Exception as exc:
                self.file_failed.emit(file, str(exc))

    def process_file(self, filepath: Path) -> Path:
        reduction = MythenDetector(
            filepath=str(filepath),
            settings=self.settings,
            output_directory=self.output_dir,
            filename_suffix=self.suffix,
        )

        if self.settings.data_reduction_mode == "step_scan":
            reduction.process_step_scan(control=False)
        else:
            raise ValueError("Not implemented yet, speak to R.Dixey")

        return Path(reduction.xye_filepath_out)


# =========================
# Plot Window
# =========================


class PlotWindow(QWidget):
    """Standalone window showing a diffraction pattern from an XYEData object.
    Includes a Matplotlib navigation toolbar for pan/zoom/reset/save."""

    def __init__(self, data: XYEData, title: str = "") -> None:
        super().__init__()
        self.setWindowTitle(title or (data.source or "Diffraction Pattern"))
        self.resize(700, 550)

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        ax = self.figure.add_subplot(111)
        if data.e is not None:
            ax.errorbar(data.x, data.y, yerr=data.e, fmt="-", linewidth=1)
        else:
            ax.plot(data.x, data.y, linewidth=1)

        ax.set_xlabel(data.x_unit)
        ax.set_ylabel(data.y_unit)
        if data.source:
            ax.set_title(Path(data.source).name)

        self.figure.tight_layout()
        self.canvas.draw()


# =========================
# Main Window
# =========================


class MainWindow(QWidget):
    # Fields that get a full-width "path + Browse + Open" row instead of a
    # normal grid cell. Maps field name -> (dialog title, file filter).
    SPECIAL_FILE_FIELDS: dict[str, tuple[str, str]] = {
        "bad_channels_filepath": ("Select Bad Channels File", "All Files (*)"),
        "angcal_filepath": ("Select Angular Calibration File", "All Files (*)"),
        "flatfield_filepath": ("Select Flatfield File", "All Files (*)"),
    }

    # Fields that get a full-width row (spanning all settings columns) but
    # are otherwise rendered with their normal widget type.
    WIDE_FIELDS: set[str] = {
        "active_modules",
        "bad_modules",
        "modules_in_flatfield",
    }

    # Fields that are not shown in the settings grid at all.
    EXCLUDED_FIELDS: set[str] = {
        "send_to_ispyb",
    }

    # This field is rendered inline next to flatfield_filepath's row rather
    # than getting its own grid cell.
    FLATFIELD_TOGGLE_FIELD: str = "apply_flatfield"

    # Widget-dict keys for the non-model fields (suffix + output dir).
    SUFFIX_FIELD: str = "Filename Suffix"
    OUTPUT_DIR_FIELD: str = "Output Directory"
    OUTPUT_DIR_ENABLED_FIELD: str = "Output Directory_enabled"

    def __init__(
        self,
        settings_path: str | Path | None = None,
        settings: MythenSettings | None = None,
        beamline: str = "i11",
        settings_columns: int = 2,
    ) -> None:
        super().__init__()

        # Output directory is blank/None by default; the user must tick the
        # "Enable" checkbox next to the field before a value takes effect.
        self.output_dir: str = ""

        self.beamline = beamline

        self.settings_columns: int = max(1, settings_columns)

        if settings is not None:
            self.settings_model: MythenSettings = settings
        elif settings_path is not None:
            self.settings_path: Path = Path(settings_path)
            self.settings_model: MythenSettings = MythenSettings.load_from_toml(
                settings_path
            )
        else:
            raise ValueError("Either settings or settings_path must be provided.")

        self.selected_files: list[Path] = []
        self.worker: ProcessingWorker | None = None

        # Maps input .nxs path -> reduced .xye output path.
        self.output_paths: dict[Path, Path] = {}
        # Keeps references to open plot windows so PyQt doesn't garbage
        # collect (and silently close) them.
        self.plot_windows: list[PlotWindow] = []

        self.widgets: dict[str, QWidget] = {}

        self.fs_model: QFileSystemModel
        self.tree: QTreeView
        self.file_list: QListWidget
        self.settings_grid: QGridLayout
        self.process_btn: QPushButton

        self.setWindowTitle("Mythen3 Reprocessor")
        self.resize(1200, 650)

        self.init_ui()

    # ---------------------
    # UI setup
    # ---------------------

    def init_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # -------- LEFT PANEL --------
        left_layout = QVBoxLayout()

        self.fs_model = QFileSystemModel()

        # Performance: avoid per-directory file-watchers, per-file shell
        # icon lookups, and symlink resolution. These are what make
        # QFileSystemModel lock up on folders with thousands of entries on
        # a network mount - none of them are needed here, we only care
        # about folder-vs-.nxs-file.
        self.fs_model.setOption(QFileSystemModel.Option.DontWatchForChanges, True)
        self.fs_model.setIconProvider(FastIconProvider())
        self.fs_model.setResolveSymlinks(False)

        if Path(DEFAULT_DATA_FOLDER).exists():
            self.base_path = DEFAULT_DATA_FOLDER
        else:
            self.base_path = ROOT

        self.fs_model.setRootPath(self.base_path)
        self.fs_model.setNameFilters(["*.nxs"])
        self.fs_model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self.base_path))
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Performance: skip per-row height recalculation and expand/collapse
        # animation, both of which add up at thousands of rows.
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(False)
        self.tree.setSortingEnabled(False)

        for col in range(1, self.fs_model.columnCount()):
            self.tree.hideColumn(col)

        selection_model = self.tree.selectionModel()
        if selection_model:
            selection_model.selectionChanged.connect(self.on_selection_changed)

        self.file_list = QListWidget()
        self.file_list.setToolTip("Double-click a finished file to plot it")
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)

        left_layout.addWidget(self.tree)
        left_layout.addWidget(self.file_list)

        # -------- RIGHT PANEL --------
        right_layout = QVBoxLayout()

        self.settings_grid = QGridLayout()
        self.settings_grid.setHorizontalSpacing(20)
        self.settings_grid.setVerticalSpacing(10)

        self.build_settings_grid()

        right_layout.addLayout(self.settings_grid)
        right_layout.addStretch()

        self.process_btn = QPushButton("PROCESS")
        self.process_btn.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 12px;"
        )
        self.process_btn.clicked.connect(self.process)

        right_layout.addWidget(self.process_btn)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

    # ---------------------
    # Settings grid
    # ---------------------

    def build_settings_grid(self) -> None:
        normal_settings: list[tuple[str, QWidget]] = []
        wide_settings: list[tuple[str, QWidget]] = []
        file_settings: list[tuple[str, QWidget]] = []

        model_class = type(self.settings_model)

        # Grab apply_flatfield's value up front so its checkbox can be built
        # and slotted in beside flatfield_filepath, regardless of field
        # declaration order in the model.
        apply_flatfield_val = getattr(self.settings_model, self.FLATFIELD_TOGGLE_FIELD)

        for setting_name, field in model_class.model_fields.items():
            if setting_name in self.EXCLUDED_FIELDS:
                continue

            if setting_name == self.FLATFIELD_TOGGLE_FIELD:
                # Handled inline alongside flatfield_filepath below.
                continue

            setting_val = getattr(self.settings_model, setting_name)
            annotation = field.annotation

            if setting_name in self.SPECIAL_FILE_FIELDS:
                dialog_title, file_filter = self.SPECIAL_FILE_FIELDS[setting_name]

                extra_widget: QWidget | None = None
                if setting_name == "flatfield_filepath":
                    extra_widget = self.make_checkbox(
                        self.FLATFIELD_TOGGLE_FIELD, apply_flatfield_val
                    )

                widget = self.make_file_widget(
                    setting_name,
                    str(setting_val),
                    dialog_title,
                    file_filter,
                    extra_widget=extra_widget,
                )
                file_settings.append((setting_name, widget))
                continue

            if setting_name in self.WIDE_FIELDS:
                widget = self.make_setting(setting_name, setting_val, annotation)
                wide_settings.append((setting_name, widget))
                continue

            widget = self.make_setting(setting_name, setting_val, annotation)

            normal_settings.append((setting_name, widget))

        span = self.settings_columns * 2 - 1
        row = 0

        # Full-width rows for list settings (active_modules, bad_modules,
        # modules_in_flatfield) go first, one per row.
        for setting_name, widget in wide_settings:
            self.settings_grid.addWidget(QLabel(setting_name), row, 0)
            self.settings_grid.addWidget(widget, row, 1, 1, span)
            row += 1

        # Then the normal settings, packed into the column grid.
        normal_start_row = row
        for i, (label, widget) in enumerate(normal_settings):
            col = i % self.settings_columns
            grid_row = normal_start_row + i // self.settings_columns

            label_col = col * 2
            field_col = label_col + 1

            self.settings_grid.addWidget(QLabel(label), grid_row, label_col)
            self.settings_grid.addWidget(widget, grid_row, field_col)

        next_row = (
            normal_start_row
            + (len(normal_settings) + self.settings_columns - 1)
            // self.settings_columns
        )

        # Free-text suffix appended to the output filename. Not part of the
        # MythenSettings model, so it's read directly from self.widgets in
        # process() rather than via collect_settings().
        self.settings_grid.addWidget(QLabel(self.SUFFIX_FIELD), next_row, 0)
        self.settings_grid.addWidget(
            self.make_suffix_widget(),
            next_row,
            1,
            1,
            span,
        )
        next_row += 1

        # Full-width rows for file-path settings (bad_channels_filepath,
        # angcal_filepath, flatfield_filepath), one per row. flatfield_filepath
        # also carries the apply_flatfield checkbox inline.
        for setting_name, widget in file_settings:
            self.settings_grid.addWidget(QLabel(setting_name), next_row, 0)
            self.settings_grid.addWidget(widget, next_row, 1, 1, span)
            next_row += 1

        self.settings_grid.addWidget(QLabel(self.OUTPUT_DIR_FIELD), next_row, 0)
        self.settings_grid.addWidget(
            self.make_dir_widget(self.OUTPUT_DIR_FIELD),
            next_row,
            1,
            1,
            span,
        )

    # ---------------------
    # Widget builders
    # ---------------------

    def make_setting(
        self, setting_name: str, setting_val: Any, annotation: Any
    ) -> QWidget:
        if (get_origin(annotation) is list) and (get_args(annotation) == (int,)):
            w = QLineEdit()
            w.setText(str(setting_val))
            self.widgets[setting_name] = w
            return w
        elif (get_origin(annotation) is UnionType) and (Path in get_args(annotation)):
            w = QLineEdit()
            w.setText(str(setting_val))
            self.widgets[setting_name] = w
            return w
        elif get_origin(annotation) is Literal:
            get_allowed_literals = get_args(annotation)
            w = QComboBox()
            w.addItems(get_allowed_literals)
            w.setCurrentText(setting_val)
            self.widgets[setting_name] = w
            return w
        elif annotation is float:
            w = QDoubleSpinBox()
            w.setRange(0.0, 1e9)
            w.setDecimals(3)
            w.setSingleStep(0.001)
            w.setValue(setting_val)
            self.widgets[setting_name] = w
            return w
        elif annotation is int:
            w = QSpinBox()
            w.setRange(1, 1_000_000)
            w.setValue(setting_val)
            self.widgets[setting_name] = w
            return w
        elif annotation is bool:
            w = QCheckBox()
            w.setChecked(setting_val)
            self.widgets[setting_name] = w
            return w
        else:
            raise ValueError(f"Unknown setting type: {setting_name}")

    def make_checkbox(self, name: str, checked: bool) -> QCheckBox:
        w = QCheckBox(name)
        w.setChecked(checked)
        self.widgets[name] = w
        return w

    def make_suffix_widget(self) -> QWidget:
        """Free-text suffix appended to output filenames, e.g. via
        MythenDetector(..., filename_suffix=suffix). Optional - left blank
        means no suffix."""
        edit = QLineEdit()
        edit.setText("_reprocessed")
        self.widgets[self.SUFFIX_FIELD] = edit
        return edit

    def make_dir_widget(self, name: str) -> QWidget:
        """Output directory row: blank/disabled by default (meaning 'use the
        default output location', i.e. None is passed through). Ticking
        'Enable' turns on the text field + Browse button so the user can
        supply an explicit directory."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        enable_checkbox = QCheckBox("Enable")
        enable_checkbox.setChecked(False)

        edit = QLineEdit(self.output_dir)
        edit.setPlaceholderText("None (default output location will be used)")
        edit.setEnabled(False)

        browse = QPushButton("Browse…")
        browse.setEnabled(False)
        browse.clicked.connect(self.browse_output_dir)

        def on_toggle(checked: bool) -> None:
            edit.setEnabled(checked)
            browse.setEnabled(checked)
            if not checked:
                edit.clear()

        enable_checkbox.toggled.connect(on_toggle)

        layout.addWidget(enable_checkbox)
        layout.addWidget(edit)
        layout.addWidget(browse)

        self.widgets[name] = edit
        self.widgets[self.OUTPUT_DIR_ENABLED_FIELD] = enable_checkbox
        return container

    def make_file_widget(
        self,
        name: str,
        current_value: str,
        dialog_title: str,
        file_filter: str = "All Files (*)",
        extra_widget: QWidget | None = None,
    ) -> QWidget:
        """Build a 'path + Browse… + Open [+ extra_widget]' row for a
        file-path setting. `extra_widget`, if given, is appended to the end
        of the row (e.g. the apply_flatfield checkbox next to
        flatfield_filepath)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        edit = QLineEdit(current_value)

        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda: self.browse_file_path(name, dialog_title, file_filter)
        )

        open_btn = QPushButton("Open")
        open_btn.clicked.connect(lambda: self.open_file_path(name))

        layout.addWidget(edit)
        layout.addWidget(browse)
        layout.addWidget(open_btn)

        if extra_widget is not None:
            layout.addWidget(extra_widget)

        self.widgets[name] = edit
        return container

    # ---------------------
    # Output directory picker
    # ---------------------

    def browse_output_dir(self) -> None:
        edit = cast(QLineEdit, self.widgets[self.OUTPUT_DIR_FIELD])
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            edit.text() or str(Path.home()),
        )
        if directory:
            edit.setText(directory)

    # ---------------------
    # File path picker / opener (bad_channels_filepath, angcal_filepath,
    # flatfield_filepath)
    # ---------------------

    def browse_file_path(self, name: str, dialog_title: str, file_filter: str) -> None:
        edit = cast(QLineEdit, self.widgets[name])
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            edit.text() or str(Path.home()),
            file_filter,
        )
        if filepath:
            edit.setText(filepath)

    def open_file_path(self, name: str) -> None:
        """Open the file currently in the given field with the OS default
        application (e.g. the system's default text editor)."""
        edit = cast(QLineEdit, self.widgets[name])
        text = edit.text().strip()

        if not text:
            QMessageBox.warning(self, "No file", "No file path has been set.")
            return

        path = Path(text)
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"Could not find file:\n{path}")
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not opened:
            QMessageBox.critical(
                self, "Could not open file", f"Failed to open:\n{path}"
            )

    # ---------------------
    # File selection
    # ---------------------

    def on_selection_changed(self, *_: object) -> None:
        self.selected_files.clear()
        self.file_list.clear()

        selection_model = self.tree.selectionModel()
        if selection_model is None:
            return
        indexes = selection_model.selectedIndexes()
        seen: set[Path] = set()

        for index in indexes:
            if index.column() != 0:
                continue

            path = Path(self.fs_model.filePath(index))
            if path.is_file() and path.suffix == ".nxs" and path not in seen:
                seen.add(path)
                self.selected_files.append(path)

                item = QListWidgetItem(f"⏳ {path.name}")
                item.setData(int(Qt.ItemDataRole.UserRole), path)
                item.setData(OUTPUT_PATH_ROLE, None)
                self.file_list.addItem(item)

    # ---------------------
    # Collect settings
    # ---------------------

    @staticmethod
    def _parse_int_list(text: str) -> list[int]:
        """Parse a list[int] field's text back out of its QLineEdit.

        Accepts either Python-list-repr style ("[0, 1, 2]") or a plain
        comma/space separated list ("0, 1, 2" / "0 1 2").
        """
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]

        if not text:
            return []

        parts = [p.strip() for p in text.replace(",", " ").split()]

        try:
            return [int(p) for p in parts if p]
        except ValueError as exc:
            raise ValueError(
                f"Could not parse a list of integers from '{text}': {exc}"
            ) from exc

    def collect_settings(self) -> MythenSettings:
        model_class = type(self.settings_model)
        kwargs: dict[str, Any] = {}

        for setting_name, field in model_class.model_fields.items():
            if setting_name in self.EXCLUDED_FIELDS:
                # Not shown in the UI; let the model fall back to its
                # default (or whatever self.settings_model already has).
                continue

            widget = self.widgets.get(setting_name)
            if widget is None:
                # Wasn't rendered for some reason - don't silently drop the
                # value, keep whatever the model currently holds.
                continue

            annotation = field.annotation

            if (get_origin(annotation) is list) and (get_args(annotation) == (int,)):
                kwargs[setting_name] = self._parse_int_list(
                    cast(QLineEdit, widget).text()
                )
            elif (get_origin(annotation) is UnionType) and (
                Path in get_args(annotation)
            ):
                kwargs[setting_name] = cast(QLineEdit, widget).text().strip()
            elif get_origin(annotation) is Literal:
                kwargs[setting_name] = cast(QComboBox, widget).currentText()
            elif annotation is float:
                kwargs[setting_name] = cast(QDoubleSpinBox, widget).value()
            elif annotation is int:
                kwargs[setting_name] = cast(QSpinBox, widget).value()
            elif annotation is bool:
                kwargs[setting_name] = cast(QCheckBox, widget).isChecked()
            else:
                raise ValueError(
                    f"Unknown setting type for '{setting_name}': {annotation}"
                )

        return MythenSettings(**kwargs)

    def collect_output_dir(self) -> str | None:
        """Returns None unless the user has ticked 'Enable' and supplied a
        non-empty path, in which case that path is returned."""
        checkbox = cast(QCheckBox, self.widgets[self.OUTPUT_DIR_ENABLED_FIELD])
        if not checkbox.isChecked():
            return None

        edit = cast(QLineEdit, self.widgets[self.OUTPUT_DIR_FIELD])
        text = edit.text().strip()
        return text or None

    def collect_suffix(self) -> str:
        edit = cast(QLineEdit, self.widgets[self.SUFFIX_FIELD])
        return edit.text().strip()

    # ---------------------
    # Process
    # ---------------------

    def process(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No files", "Select at least one .nxs file.")
            return

        try:
            settings = self.collect_settings()
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        output_dir = self.collect_output_dir()
        suffix = self.collect_suffix()

        self.process_btn.setEnabled(False)

        self.worker = ProcessingWorker(
            self.selected_files, settings, output_dir=output_dir, suffix=suffix
        )
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.file_failed.connect(self.on_file_failed)
        self.worker.finished.connect(self.on_all_done)
        self.worker.start()

    # ---------------------
    # Worker callbacks
    # ---------------------

    def find_item(self, path: Path) -> QListWidgetItem | None:
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is not None and item.data(int(Qt.ItemDataRole.UserRole)) == path:
                return item
        return None

    def on_file_started(self, path: Path) -> None:
        item = self.find_item(path)
        if item:
            item.setText(f"🔄 {path.name}")

    def on_file_finished(self, path: Path, output_path: Path) -> None:
        self.output_paths[path] = output_path

        item = self.find_item(path)
        if item:
            item.setData(OUTPUT_PATH_ROLE, output_path)
            item.setText(f"✅ {path.name}  →  {output_path.name}")

    def on_file_failed(self, path: Path, error: str) -> None:
        item = self.find_item(path)
        if item:
            item.setText(f"❌ {path.name}")

        print(f"Error processing {path}: {error}")

    def on_all_done(self) -> None:
        self.process_btn.setEnabled(True)
        QMessageBox.information(self, "Done", "All files processed.")

    # ---------------------
    # Plotting finished files
    # ---------------------

    def on_file_double_clicked(self, item: QListWidgetItem) -> None:
        output_path = item.data(OUTPUT_PATH_ROLE)
        if not output_path:
            QMessageBox.information(
                self,
                "Not ready",
                (
                    "This file hasn't finished processing yet, "
                    "or no output file could be found."
                ),
            )
            return

        try:
            data = XYEData.from_csv(output_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not load data", f"Failed to load {output_path}:\n{exc}"
            )
            return

        data.x_unit = "tth"

        plot_window = PlotWindow(data, title=Path(output_path).name)
        self.plot_windows.append(plot_window)
        plot_window.show()


def run_mythen_process():
    app = QApplication(sys.argv)
    settings = MythenSettings()
    window = MainWindow(settings=settings, settings_columns=2)
    window.show()
    sys.exit(app.exec())


# =========================
# App entry point
# =========================

if __name__ == "__main__":
    run_mythen_process()
