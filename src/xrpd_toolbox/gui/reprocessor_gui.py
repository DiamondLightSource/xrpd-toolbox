import sys
import time
from pathlib import Path
from types import UnionType
from typing import Any, Literal, cast, get_args, get_origin

from pydantic import ValidationError
from PyQt5.QtCore import QDir, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
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

from xrpd_toolbox.i11.mythen import MythenSettings

# =========================
# Worker Thread
# =========================


class ProcessingWorker(QThread):
    file_started = pyqtSignal(Path)
    file_finished = pyqtSignal(Path)
    file_failed = pyqtSignal(Path, str)

    def __init__(self, files: list[Path], settings: MythenSettings) -> None:
        super().__init__()
        self.files: list[Path] = files
        self.settings: MythenSettings = settings

    def run(self) -> None:
        for file in self.files:
            self.file_started.emit(file)
            try:
                self.process_file(file)
                self.file_finished.emit(file)
            except Exception as exc:
                self.file_failed.emit(file, str(exc))

    def process_file(self, file: Path) -> None:
        # Replace with real processing
        time.sleep(1)


# =========================
# Main Window
# =========================


class MainWindow(QWidget):
    def __init__(
        self,
        settings_path: str | Path | None = None,
        settings: MythenSettings | None = None,
        beamline: str = "i11",
        settings_columns: int = 1,
    ) -> None:
        super().__init__()

        self.output_dir: str = str(Path.home())
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

        self.widgets: dict[str, QWidget] = {}

        self.fs_model: QFileSystemModel
        self.tree: QTreeView
        self.file_list: QListWidget
        self.settings_grid: QGridLayout
        self.process_btn: QPushButton

        self.setWindowTitle("NXS Reprocessor")
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

        self.base_path = f"/dls/{self.beamline}/data/"

        self.fs_model.setRootPath(QDir.rootPath())
        self.fs_model.setNameFilters(["*.nxs"])
        self.fs_model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self.base_path))
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for col in range(1, self.fs_model.columnCount()):
            self.tree.hideColumn(col)

        selection_model = self.tree.selectionModel()
        if selection_model:
            selection_model.selectionChanged.connect(self.on_selection_changed)

        self.file_list = QListWidget()

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

        model_class = type(self.settings_model)

        for setting_name, field in model_class.model_fields.items():
            setting_val = getattr(self.settings_model, setting_name)
            annotation = field.annotation
            # default = field.default

            widget = self.make_setting(setting_name, setting_val, annotation)
            print("\n")

            normal_settings.append((setting_name, widget))

        for i, (label, widget) in enumerate(normal_settings):
            col = i % self.settings_columns
            row = i // self.settings_columns

            label_col = col * 2
            field_col = label_col + 1

            self.settings_grid.addWidget(QLabel(label), row, label_col)
            self.settings_grid.addWidget(widget, row, field_col)

        output_row = (
            len(normal_settings) + self.settings_columns - 1
        ) // self.settings_columns

        output_name = "Output Directory"
        self.settings_grid.addWidget(QLabel(output_name), output_row, 0)
        self.settings_grid.addWidget(
            self.make_dir_widget(output_name),
            output_row,
            1,
            1,
            self.settings_columns * 2 - 1,
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
            w.setText(setting_val)
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

    def make_dir_widget(self, name) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        edit = QLineEdit(self.output_dir)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_output_dir)

        layout.addWidget(edit)
        layout.addWidget(browse)

        self.widgets[name] = edit
        return container

    # ---------------------
    # Output directory picker
    # ---------------------

    def browse_output_dir(self) -> None:
        edit = cast(QLineEdit, self.widgets["output_dir"])
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            edit.text() or str(Path.home()),
        )
        if directory:
            edit.setText(directory)

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
                self.file_list.addItem(item)

    # ---------------------
    # Collect settings
    # ---------------------

    def collect_settings(self) -> MythenSettings:
        return MythenSettings(
            # threshold=cast(QDoubleSpinBox, self.widgets["threshold"]).value(),
            # max_iterations=cast(QSpinBox, self.widgets["max_iterations"]).value(),
            # normalize=cast(QCheckBox, self.widgets["normalize"]).isChecked(),
            # output_dir=cast(QLineEdit, self.widgets["output_dir"]).text(),
        )

    # ---------------------
    # Process
    # ---------------------

    def process(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No files", "Select at least one .nxs file.")
            return

        try:
            settings = self.collect_settings()
        except ValidationError as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.process_btn.setEnabled(False)

        self.worker = ProcessingWorker(self.selected_files, settings)
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
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == path:
                return item
        return None

    def on_file_started(self, path: Path) -> None:
        item = self.find_item(path)
        if item:
            item.setText(f"🔄 {path.name}")

    def on_file_finished(self, path: Path) -> None:
        item = self.find_item(path)
        if item:
            item.setText(f"✅ {path.name}")

    def on_file_failed(self, path: Path, error: str) -> None:
        item = self.find_item(path)
        if item:
            item.setText(f"❌ {path.name}")

        print(f"Error processing {path}: {error}")

    def on_all_done(self) -> None:
        self.process_btn.setEnabled(True)
        QMessageBox.information(self, "Done", "All files processed.")


# =========================
# App entry point
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    settings = MythenSettings()
    window = MainWindow(settings=settings, settings_columns=1)
    window.show()
    sys.exit(app.exec_())
