from __future__ import annotations

from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidgetItem,
    QSpinBox,
    QWidget,
)

# --- adjust to your real module path ---------------------------------
from xrpd_toolbox.gui.mythen3_process_gui import (
    OUTPUT_PATH_ROLE,
    MainWindow,
    ProcessingWorker,
)
from xrpd_toolbox.i11.mythen import MythenSettings

# -----------------------------------------------------------------------

MODULE: str = ProcessingWorker.__module__


def _cast(widget_type: type[QWidget], widget: QWidget) -> Any:
    """Thin wrapper so test code reads the same way the production code's
    own `cast(...)` calls do, and so static analysis is happy."""
    assert isinstance(widget, widget_type), (
        f"expected {widget_type}, got {type(widget)}"
    )
    return widget


# =========================
# Fixtures
# =========================


@pytest.fixture(autouse=True)
def _never_touch_real_data_folder(monkeypatch, tmp_path) -> None:
    """Point DEFAULT_DATA_FOLDER at an empty tmp dir instead of the real
    (potentially huge, network-mounted) data folder for every test."""
    monkeypatch.setattr(f"{MODULE}.DEFAULT_DATA_FOLDER", str(tmp_path))


@pytest.fixture
def settings() -> MythenSettings:
    return MythenSettings()


@pytest.fixture
def window(qtbot, settings) -> MainWindow:
    win = MainWindow(settings=settings, settings_columns=2)
    qtbot.addWidget(win)
    return win


@pytest.fixture
def fake_detector(monkeypatch):
    """Stand-in for MythenDetector that records how it was constructed and
    called, without touching real Nexus/HDF5 data."""

    class FakeDetector:
        last_instance: FakeDetector | None = None

        def __init__(
            self, filepath, settings, output_directory=None, filename_suffix=""
        ):
            self.filepath = filepath
            self.settings = settings
            self.output_directory = output_directory
            self.filename_suffix = filename_suffix
            out_dir = (
                Path(output_directory) if output_directory else Path(filepath).parent
            )
            self.xye_filepath_out = str(
                out_dir / f"{Path(filepath).stem}{filename_suffix}.xye"
            )
            self.process_step_scan_called_with: bool | None = None
            FakeDetector.last_instance = self

        def process_step_scan(self, control: bool = False) -> None:
            self.process_step_scan_called_with = control

    monkeypatch.setattr(f"{MODULE}.MythenDetector", FakeDetector)
    return FakeDetector


# =========================
# Generic field-type dispatch helpers (mirrors make_setting/collect_settings)
# =========================


def _unwrap_optional(annotation: Any) -> Any:
    """If annotation is a Union containing a Literal[...] member (covers
    Optional[Literal[...]], Literal[...] | int, or any other Literal
    combined with something else — a Union involving Literal always
    normalizes to typing.Union rather than types.UnionType, unlike every
    other X | Y combination), return that Literal[...] member so the tests
    dispatch the same way `make_setting`/`collect_settings` do. Otherwise
    return annotation unchanged."""
    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        for arg in get_args(annotation):
            if get_origin(arg) is Literal:
                return arg
    return annotation


def _is_int_list(annotation: Any) -> bool:
    annotation = _unwrap_optional(annotation)
    return get_origin(annotation) is list and get_args(annotation) == (int,)


def _is_path_union(annotation: Any) -> bool:
    annotation = _unwrap_optional(annotation)
    return get_origin(annotation) is UnionType and Path in get_args(annotation)


def _is_literal(annotation: Any) -> bool:
    return get_origin(_unwrap_optional(annotation)) is Literal


def _model_fields_excluding(model_cls, extra_exclude: set[str] = frozenset()):
    return {
        name: field
        for name, field in model_cls.model_fields.items()
        if name not in MainWindow.EXCLUDED_FIELDS and name not in extra_exclude
    }


def _expected_widget_type(annotation: Any) -> type[QWidget]:
    annotation = _unwrap_optional(annotation)
    if _is_int_list(annotation) or _is_path_union(annotation):
        return QLineEdit
    if _is_literal(annotation):
        return QComboBox
    if annotation is float:
        return QDoubleSpinBox
    if annotation is int:
        return QSpinBox
    if annotation is bool:
        return QCheckBox
    raise AssertionError(f"No known widget mapping for annotation {annotation!r}")


def _bump_value(annotation: Any, current: Any) -> Any:
    """A different-but-valid value for `annotation`, used to prove a field
    round-trips through its widget."""
    annotation = _unwrap_optional(annotation)
    if _is_int_list(annotation):
        return [1, 2, 3] if list(current) != [1, 2, 3] else [4, 5, 6]
    if _is_path_union(annotation):
        return "/tmp/bumped_test_path"
    if _is_literal(annotation):
        options = get_args(annotation)
        other = next((o for o in options if o != current), None)
        if other is None:
            pytest.skip(f"Literal field has only one option: {options}")
        return other
    if annotation is float:
        return float(current) + 1.0
    if annotation is int:
        return int(current) + 1
    if annotation is bool:
        return not current
    raise AssertionError(f"No bump strategy for annotation {annotation!r}")


def _set_widget_value(widget: QWidget, annotation: Any, value: Any) -> None:
    annotation = _unwrap_optional(annotation)
    if _is_int_list(annotation):
        _cast(QLineEdit, widget).setText(", ".join(str(v) for v in value))
    elif _is_path_union(annotation):
        _cast(QLineEdit, widget).setText(str(value))
    elif _is_literal(annotation):
        _cast(QComboBox, widget).setCurrentText(value)
    elif annotation is float:
        _cast(QDoubleSpinBox, widget).setValue(value)
    elif annotation is int:
        _cast(QSpinBox, widget).setValue(value)
    elif annotation is bool:
        _cast(QCheckBox, widget).setChecked(value)
    else:
        raise AssertionError(f"No setter for annotation {annotation!r}")


def _first_field_named(model_cls, predicate) -> str | None:
    for name, field in model_cls.model_fields.items():
        if name in MainWindow.EXCLUDED_FIELDS:
            continue
        if predicate(field.annotation):
            return name
    return None


def _literal_options(annotation: Any) -> tuple:
    """get_args() of the Literal itself, unwrapping Optional[Literal[...]]
    (or Literal[...] | int, etc.) first so we don't accidentally include
    the Literal type object or another union member as a spurious 'option'."""
    return get_args(_unwrap_optional(annotation))


# =========================
# _parse_int_list (pure logic — unaffected by real model shape)
# =========================


class TestParseIntList:
    def test_python_list_repr(self) -> None:
        assert MainWindow._parse_int_list("[0, 1, 2]") == [0, 1, 2]

    def test_comma_separated(self) -> None:
        assert MainWindow._parse_int_list("0, 1, 2") == [0, 1, 2]

    def test_space_separated(self) -> None:
        assert MainWindow._parse_int_list("0 1 2") == [0, 1, 2]

    def test_mixed_whitespace_and_commas(self) -> None:
        assert MainWindow._parse_int_list("0,  1 ,2   3") == [0, 1, 2, 3]

    def test_empty_string(self) -> None:
        assert MainWindow._parse_int_list("") == []

    def test_empty_brackets(self) -> None:
        assert MainWindow._parse_int_list("[]") == []

    def test_whitespace_only(self) -> None:
        assert MainWindow._parse_int_list("   ") == []

    def test_single_value(self) -> None:
        assert MainWindow._parse_int_list("5") == [5]

    def test_negative_numbers(self) -> None:
        assert MainWindow._parse_int_list("-1, 0, 1") == [-1, 0, 1]

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse a list of integers"):
            MainWindow._parse_int_list("0, foo, 2")

    def test_trailing_commas_ignored(self) -> None:
        assert MainWindow._parse_int_list("0, 1, 2,") == [0, 1, 2]


# =========================
# ProcessingWorker.process_file — MythenDetector mocked out
# =========================


class TestProcessingWorkerProcessFile:
    def test_step_scan_calls_detector_and_returns_output_path(
        self, tmp_path, settings, fake_detector
    ) -> None:
        settings.data_reduction_mode = "step_scan"
        input_file = tmp_path / "scan0001.nxs"

        worker = ProcessingWorker([input_file], settings, output_dir=None, suffix="")
        out = worker.process_file(input_file)

        inst = fake_detector.last_instance
        assert inst is not None
        assert inst.filepath == str(input_file)
        assert inst.settings is settings
        assert inst.process_step_scan_called_with is False
        assert out == Path(inst.xye_filepath_out)

    def test_step_scan_passes_through_output_dir_and_suffix(
        self, tmp_path, settings, fake_detector
    ) -> None:
        settings.data_reduction_mode = "step_scan"
        input_file = tmp_path / "scan0001.nxs"
        out_dir = tmp_path / "out"

        worker = ProcessingWorker(
            [input_file], settings, output_dir=str(out_dir), suffix="_reprocessed"
        )
        worker.process_file(input_file)

        inst = fake_detector.last_instance
        assert inst.output_directory == str(out_dir)
        assert inst.filename_suffix == "_reprocessed"

    def test_non_step_scan_mode_raises_and_never_calls_detector(
        self, tmp_path, settings, fake_detector
    ) -> None:
        # Any mode other than "step_scan" should hit the NotImplemented branch.
        other_modes = [
            m
            for m in _literal_options(
                MythenSettings.model_fields["data_reduction_mode"].annotation
            )
            if m != "step_scan"
        ]
        if not other_modes:
            pytest.skip("data_reduction_mode has no non-step_scan option to test")
        settings.data_reduction_mode = other_modes[0]
        input_file = tmp_path / "scan0001.nxs"

        worker = ProcessingWorker([input_file], settings)
        with pytest.raises(ValueError, match="Not implemented yet"):
            worker.process_file(input_file)

        # Detector is constructed before the mode check in the source, so
        # we only assert process_step_scan itself was never invoked.
        if fake_detector.last_instance is not None:
            assert fake_detector.last_instance.process_step_scan_called_with is None

    def test_run_emits_finished_signal_on_success(
        self, qtbot, tmp_path, settings, fake_detector
    ) -> None:
        settings.data_reduction_mode = "step_scan"
        input_file = tmp_path / "scan0001.nxs"
        worker = ProcessingWorker([input_file], settings)

        with qtbot.waitSignal(worker.file_finished, timeout=5000) as blocker:
            worker.run()

        emitted_input, emitted_output = blocker.args
        assert emitted_input == input_file
        assert emitted_output == Path(fake_detector.last_instance.xye_filepath_out)

    def test_run_emits_failed_signal_on_exception(
        self, qtbot, tmp_path, settings, fake_detector
    ) -> None:
        other_modes = [
            m
            for m in _literal_options(
                MythenSettings.model_fields["data_reduction_mode"].annotation
            )
            if m != "step_scan"
        ]
        if not other_modes:
            pytest.skip("data_reduction_mode has no non-step_scan option to test")
        settings.data_reduction_mode = other_modes[0]
        input_file = tmp_path / "scan0001.nxs"
        worker = ProcessingWorker([input_file], settings)

        with qtbot.waitSignal(worker.file_failed, timeout=5000) as blocker:
            worker.run()

        emitted_input, error_message = blocker.args
        assert emitted_input == input_file
        assert "Not implemented yet" in error_message

    def test_run_processes_multiple_files_independently(
        self, qtbot, tmp_path, settings, fake_detector
    ) -> None:
        settings.data_reduction_mode = "step_scan"
        f1 = tmp_path / "a.nxs"
        f2 = tmp_path / "b.nxs"

        worker = ProcessingWorker([f1, f2], settings)
        finished = []
        worker.file_finished.connect(lambda i, o: finished.append((i, o)))

        worker.run()

        assert {i for i, _ in finished} == {f1, f2}


class TestKnownProductionBugs:
    """Regression tests for bugs found while writing this suite. Each test
    documents exactly what's broken and why, and should start passing once
    the corresponding source fix is applied."""

    def test_make_setting_handles_literal_combined_with_other_types(
        self, window: MainWindow
    ) -> None:
        """Any Union involving Literal[...] — Optional[Literal[...]],
        Literal[...] | int, etc. — is always normalized by Python's typing
        module to typing.Union rather than types.UnionType, unlike every
        other X | Y combination. `make_setting`/`collect_settings` only
        checked `get_origin(annotation) is UnionType` (types.UnionType), so
        any field typed this way fell through every branch and raised
        "Unknown setting type" — which happened for real during ordinary
        `MainWindow` construction, not just in tests.

        Fix: `_unwrap_optional` scans a Union's members for one whose own
        origin is Literal and uses that, rather than only special-casing
        NoneType.
        """
        annotation = Literal["a", "b", "c"] | int
        widget = window.make_setting("_regression_test_field", "a", annotation)
        assert isinstance(widget, QComboBox)
        assert widget.currentText() == "a"
        items = [widget.itemText(i) for i in range(widget.count())]
        assert items == ["a", "b", "c"]

    def test_make_setting_int_spinbox_allows_zero_and_negative_values(
        self, window: MainWindow
    ) -> None:
        """QSpinBox's range previously had a hardcoded floor of 1, which
        silently clamped any int field whose real value is 0 (or negative)
        up to 1 the instant the widget was built — a real data-corruption
        bug: opening the GUI and hitting Process without touching an
        int field defaulting to 0 would silently send 1 instead."""
        widget = window.make_setting("_regression_zero_field", 0, int)
        assert isinstance(widget, QSpinBox)
        assert widget.value() == 0

        widget2 = window.make_setting("_regression_negative_field", -5, int)
        assert widget2.value() == -5


# =========================
# MainWindow: settings grid construction (dynamic over real model fields)
# =========================


class TestBuildSettingsGrid:
    def test_excluded_fields_not_rendered(self, window: MainWindow) -> None:
        for name in MainWindow.EXCLUDED_FIELDS:
            assert name not in window.widgets

    def test_flatfield_toggle_rendered_as_checkbox(self, window: MainWindow) -> None:
        assert isinstance(window.widgets[MainWindow.FLATFIELD_TOGGLE_FIELD], QCheckBox)

    def test_special_file_fields_rendered_as_line_edits(
        self, window: MainWindow
    ) -> None:
        for name in MainWindow.SPECIAL_FILE_FIELDS:
            assert isinstance(window.widgets[name], QLineEdit)

    def test_wide_fields_rendered_as_line_edits(self, window: MainWindow) -> None:
        for name in MainWindow.WIDE_FIELDS:
            assert isinstance(window.widgets[name], QLineEdit)

    def test_every_field_gets_the_widget_type_its_annotation_implies(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        fields = _model_fields_excluding(
            type(settings), extra_exclude={MainWindow.FLATFIELD_TOGGLE_FIELD}
        )
        for name, field in fields.items():
            widget = window.widgets.get(name)
            assert widget is not None, f"no widget built for field '{name}'"
            expected_type = _expected_widget_type(field.annotation)
            assert isinstance(widget, expected_type), (
                f"field '{name}' ({field.annotation}) expected {expected_type}, "
                f"got {type(widget)}"
            )

    def test_literal_field_options_match_model(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        for name, field in type(settings).model_fields.items():
            if not _is_literal(field.annotation):
                continue
            combo = _cast(QComboBox, window.widgets[name])
            items = [combo.itemText(i) for i in range(combo.count())]
            assert items == list(_literal_options(field.annotation))
            assert combo.currentText() == getattr(settings, name)

    def test_suffix_widget_defaults_to_reprocessed(self, window: MainWindow) -> None:
        w = _cast(QLineEdit, window.widgets[MainWindow.SUFFIX_FIELD])
        assert w.text() == "_reprocessed"

    def test_output_dir_widgets_present_and_disabled_by_default(
        self, window: MainWindow
    ) -> None:
        edit = _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD])
        checkbox = _cast(QCheckBox, window.widgets[MainWindow.OUTPUT_DIR_ENABLED_FIELD])
        assert checkbox.isChecked() is False
        assert edit.isEnabled() is False

    def test_unknown_annotation_type_raises(self, window: MainWindow) -> None:
        with pytest.raises(ValueError, match="Unknown setting type"):
            window.make_setting("some_field", "x", str)


# =========================
# Output directory enable/disable toggle behaviour
# =========================


class TestOutputDirToggle:
    def test_enabling_checkbox_enables_edit(self, window: MainWindow) -> None:
        checkbox = _cast(QCheckBox, window.widgets[MainWindow.OUTPUT_DIR_ENABLED_FIELD])
        edit = _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD])
        assert edit.isEnabled() is False
        checkbox.setChecked(True)
        assert edit.isEnabled() is True

    def test_disabling_checkbox_clears_edit_text(self, window: MainWindow) -> None:
        checkbox = _cast(QCheckBox, window.widgets[MainWindow.OUTPUT_DIR_ENABLED_FIELD])
        edit = _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD])

        checkbox.setChecked(True)
        edit.setText("/some/output/dir")
        assert edit.text() == "/some/output/dir"

        checkbox.setChecked(False)
        assert edit.text() == ""
        assert edit.isEnabled() is False


# =========================
# collect_settings / collect_output_dir / collect_suffix
# =========================


class TestCollectSettings:
    def test_round_trips_default_settings_unchanged(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        assert window.collect_settings() == settings

    def test_each_field_round_trips_through_its_widget(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        fields = _model_fields_excluding(type(settings))
        for name, field in fields.items():
            widget = window.widgets[name]
            original = getattr(settings, name)
            bumped = _bump_value(field.annotation, original)

            _set_widget_value(widget, field.annotation, bumped)
            collected = window.collect_settings()
            assert getattr(collected, name) == bumped, f"'{name}' did not round-trip"

            # restore, so later fields in this loop aren't affected
            _set_widget_value(widget, field.annotation, original)

    def test_excluded_fields_fall_back_to_current_model_value(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        collected = window.collect_settings()
        for name in MainWindow.EXCLUDED_FIELDS:
            assert getattr(collected, name) == getattr(settings, name)

    def test_invalid_int_list_raises(
        self, window: MainWindow, settings: MythenSettings
    ) -> None:
        name = _first_field_named(type(settings), _is_int_list)
        if name is None:
            pytest.skip("model has no list[int] field")
        _cast(QLineEdit, window.widgets[name]).setText("not, a, list")
        with pytest.raises(ValueError):
            window.collect_settings()

    def test_collect_output_dir_none_when_disabled(self, window: MainWindow) -> None:
        assert window.collect_output_dir() is None

    def test_collect_output_dir_returns_text_when_enabled(
        self, window: MainWindow
    ) -> None:
        _cast(
            QCheckBox, window.widgets[MainWindow.OUTPUT_DIR_ENABLED_FIELD]
        ).setChecked(True)
        _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD]).setText(
            "/tmp/out"
        )
        assert window.collect_output_dir() == "/tmp/out"

    def test_collect_output_dir_none_when_enabled_but_blank(
        self, window: MainWindow
    ) -> None:
        _cast(
            QCheckBox, window.widgets[MainWindow.OUTPUT_DIR_ENABLED_FIELD]
        ).setChecked(True)
        _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD]).setText("   ")
        assert window.collect_output_dir() is None

    def test_collect_suffix_strips_whitespace(self, window: MainWindow) -> None:
        _cast(QLineEdit, window.widgets[MainWindow.SUFFIX_FIELD]).setText("  _foo  ")
        assert window.collect_suffix() == "_foo"

    def test_collect_suffix_default(self, window: MainWindow) -> None:
        assert window.collect_suffix() == "_reprocessed"


# =========================
# process() orchestration
# =========================


class TestProcess:
    def test_process_warns_when_no_files_selected(
        self, window: MainWindow, monkeypatch
    ) -> None:
        warned = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.warning", warned)
        window.process()
        warned.assert_called_once()
        assert window.worker is None

    def test_process_shows_error_on_invalid_settings(
        self, window: MainWindow, monkeypatch, tmp_path, settings
    ) -> None:
        list_field = _first_field_named(type(settings), _is_int_list)
        if list_field is None:
            pytest.skip("model has no list[int] field to corrupt")

        window.selected_files = [tmp_path / "a.nxs"]
        _cast(QLineEdit, window.widgets[list_field]).setText("not valid")

        critical = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.critical", critical)

        window.process()

        critical.assert_called_once()
        assert window.worker is None
        assert window.process_btn.isEnabled() is True

    def test_process_starts_worker_with_collected_settings(
        self, window: MainWindow, monkeypatch, tmp_path
    ) -> None:
        input_file = tmp_path / "a.nxs"
        window.selected_files = [input_file]

        started_worker: dict[str, Any] = {}

        class FakeWorker:
            def __init__(self, files, settings, output_dir=None, suffix=""):
                started_worker.update(
                    files=files, settings=settings, output_dir=output_dir, suffix=suffix
                )
                self.file_started = MagicMock()
                self.file_finished = MagicMock()
                self.file_failed = MagicMock()
                self.finished = MagicMock()

            def start(self) -> None:
                started_worker["started"] = True

        monkeypatch.setattr(f"{MODULE}.ProcessingWorker", FakeWorker)

        _cast(QLineEdit, window.widgets[MainWindow.SUFFIX_FIELD]).setText("_myrun")

        window.process()

        assert window.process_btn.isEnabled() is False
        assert started_worker["files"] == [input_file]
        assert started_worker["suffix"] == "_myrun"
        assert started_worker.get("started") is True


# =========================
# File selection (tree -> selected_files / file_list)
# =========================


class TestOnSelectionChanged:
    def test_only_nxs_files_are_selected(
        self, window: MainWindow, tmp_path, monkeypatch
    ) -> None:
        nxs_path = tmp_path / "scan.nxs"
        txt_path = tmp_path / "notes.txt"
        nxs_path.touch()
        txt_path.touch()

        idx_nxs, idx_txt, idx_other_col = MagicMock(), MagicMock(), MagicMock()
        idx_nxs.column.return_value = 0
        idx_txt.column.return_value = 0
        idx_other_col.column.return_value = 1

        def file_path(index):
            return {id(idx_nxs): str(nxs_path), id(idx_txt): str(txt_path)}[id(index)]

        monkeypatch.setattr(window.fs_model, "filePath", file_path)

        selection_model = MagicMock()
        selection_model.selectedIndexes.return_value = [idx_nxs, idx_txt, idx_other_col]
        monkeypatch.setattr(window.tree, "selectionModel", lambda: selection_model)

        window.on_selection_changed()

        assert window.selected_files == [nxs_path]
        assert window.file_list.count() == 1
        assert window.file_list.item(0).text() == f"⏳ {nxs_path.name}"

    def test_duplicate_paths_added_once(
        self, window: MainWindow, tmp_path, monkeypatch
    ) -> None:
        nxs_path = tmp_path / "scan.nxs"
        nxs_path.touch()

        idx_a, idx_b = MagicMock(), MagicMock()
        idx_a.column.return_value = 0
        idx_b.column.return_value = 0

        monkeypatch.setattr(window.fs_model, "filePath", lambda index: str(nxs_path))

        selection_model = MagicMock()
        selection_model.selectedIndexes.return_value = [idx_a, idx_b]
        monkeypatch.setattr(window.tree, "selectionModel", lambda: selection_model)

        window.on_selection_changed()

        assert window.selected_files == [nxs_path]
        assert window.file_list.count() == 1

    def test_no_selection_model_is_a_noop(
        self, window: MainWindow, monkeypatch
    ) -> None:
        monkeypatch.setattr(window.tree, "selectionModel", lambda: None)
        window.on_selection_changed()
        assert window.selected_files == []


# =========================
# Worker callbacks
# =========================


class TestWorkerCallbacks:
    def _add_item(self, window: MainWindow, path: Path) -> QListWidgetItem:
        item = QListWidgetItem(f"⏳ {path.name}")
        item.setData(int(Qt.ItemDataRole.UserRole), path)
        item.setData(OUTPUT_PATH_ROLE, None)
        window.file_list.addItem(item)
        return item

    def test_find_item_returns_matching_item(self, window: MainWindow) -> None:
        p1, p2 = Path("/tmp/a.nxs"), Path("/tmp/b.nxs")
        self._add_item(window, p1)
        self._add_item(window, p2)

        found = window.find_item(p2)
        assert found is not None
        assert found.data(int(Qt.ItemDataRole.UserRole)) == p2

    def test_find_item_returns_none_when_missing(self, window: MainWindow) -> None:
        assert window.find_item(Path("/nowhere.nxs")) is None

    def test_on_file_started_updates_text(self, window: MainWindow) -> None:
        p = Path("/tmp/a.nxs")
        self._add_item(window, p)
        window.on_file_started(p)
        assert window.file_list.item(0).text() == f"🔄 {p.name}"

    def test_on_file_finished_updates_text_and_output_paths(
        self, window: MainWindow
    ) -> None:
        p, out = Path("/tmp/a.nxs"), Path("/tmp/out/a.xye")
        self._add_item(window, p)

        window.on_file_finished(p, out)

        assert window.output_paths[p] == out
        item = window.file_list.item(0)
        assert item.text() == f"✅ {p.name}  →  {out.name}"
        assert item.data(OUTPUT_PATH_ROLE) == out

    def test_on_file_failed_updates_text(self, window: MainWindow, capsys) -> None:
        p = Path("/tmp/a.nxs")
        self._add_item(window, p)
        window.on_file_failed(p, "boom")
        assert window.file_list.item(0).text() == f"❌ {p.name}"
        assert "boom" in capsys.readouterr().out

    def test_callbacks_are_noop_for_unknown_path(self, window: MainWindow) -> None:
        window.on_file_started(Path("/tmp/unknown.nxs"))
        window.on_file_finished(Path("/tmp/unknown.nxs"), Path("/tmp/unknown.xye"))
        window.on_file_failed(Path("/tmp/unknown.nxs"), "err")

    def test_on_all_done_reenables_button_and_notifies(
        self, window: MainWindow, monkeypatch
    ) -> None:
        window.process_btn.setEnabled(False)
        info = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.information", info)
        window.on_all_done()
        assert window.process_btn.isEnabled() is True
        info.assert_called_once()


# =========================
# File path field: browse / open
# =========================


class TestFilePathFieldHelpers:
    FIELD = next(iter(MainWindow.SPECIAL_FILE_FIELDS))

    def test_open_file_path_warns_when_blank(
        self, window: MainWindow, monkeypatch
    ) -> None:
        _cast(QLineEdit, window.widgets[self.FIELD]).setText("")
        warn = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.warning", warn)
        window.open_file_path(self.FIELD)
        warn.assert_called_once()
        assert "No file" in warn.call_args[0][1]

    def test_open_file_path_warns_when_file_missing(
        self, window: MainWindow, monkeypatch, tmp_path
    ) -> None:
        missing = tmp_path / "does_not_exist.cal"
        _cast(QLineEdit, window.widgets[self.FIELD]).setText(str(missing))
        warn = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.warning", warn)
        window.open_file_path(self.FIELD)
        warn.assert_called_once()
        assert "not found" in warn.call_args[0][1].lower()

    def test_open_file_path_calls_qdesktopservices_when_file_exists(
        self, window: MainWindow, monkeypatch, tmp_path
    ) -> None:
        real_file = tmp_path / "cal.dat"
        real_file.touch()
        _cast(QLineEdit, window.widgets[self.FIELD]).setText(str(real_file))

        opened_urls: list[str] = []

        def fake_open_url(url: QUrl) -> bool:
            opened_urls.append(url.toLocalFile())
            return True

        monkeypatch.setattr(f"{MODULE}.QDesktopServices.openUrl", fake_open_url)
        critical = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.critical", critical)

        window.open_file_path(self.FIELD)

        assert opened_urls == [str(real_file)]
        critical.assert_not_called()

    def test_open_file_path_shows_critical_when_open_fails(
        self, window: MainWindow, monkeypatch, tmp_path
    ) -> None:
        real_file = tmp_path / "cal.dat"
        real_file.touch()
        _cast(QLineEdit, window.widgets[self.FIELD]).setText(str(real_file))

        monkeypatch.setattr(f"{MODULE}.QDesktopServices.openUrl", lambda url: False)
        critical = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.critical", critical)

        window.open_file_path(self.FIELD)
        critical.assert_called_once()

    def test_browse_file_path_sets_text_when_file_chosen(
        self, window: MainWindow, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            f"{MODULE}.QFileDialog.getOpenFileName",
            lambda *a, **k: ("/chosen/path.cal", "All Files (*)"),
        )
        window.browse_file_path(self.FIELD, "title", "All Files (*)")
        assert _cast(QLineEdit, window.widgets[self.FIELD]).text() == "/chosen/path.cal"

    def test_browse_file_path_leaves_text_when_dialog_cancelled(
        self, window: MainWindow, monkeypatch
    ) -> None:
        _cast(QLineEdit, window.widgets[self.FIELD]).setText("/existing/path.cal")
        monkeypatch.setattr(
            f"{MODULE}.QFileDialog.getOpenFileName", lambda *a, **k: ("", "")
        )
        window.browse_file_path(self.FIELD, "title", "All Files (*)")
        assert (
            _cast(QLineEdit, window.widgets[self.FIELD]).text() == "/existing/path.cal"
        )

    def test_browse_output_dir_sets_text_when_directory_chosen(
        self, window: MainWindow, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            f"{MODULE}.QFileDialog.getExistingDirectory",
            lambda *a, **k: "/chosen/output",
        )
        window.browse_output_dir()
        assert (
            _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD]).text()
            == "/chosen/output"
        )

    def test_browse_output_dir_leaves_text_when_dialog_cancelled(
        self, window: MainWindow, monkeypatch
    ) -> None:
        _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD]).setText(
            "/existing"
        )
        monkeypatch.setattr(
            f"{MODULE}.QFileDialog.getExistingDirectory", lambda *a, **k: ""
        )
        window.browse_output_dir()
        assert (
            _cast(QLineEdit, window.widgets[MainWindow.OUTPUT_DIR_FIELD]).text()
            == "/existing"
        )


# =========================
# Double-click to plot
# =========================


class TestOnFileDoubleClicked:
    def test_shows_info_when_not_ready(self, window: MainWindow, monkeypatch) -> None:
        item = QListWidgetItem("⏳ a.nxs")
        item.setData(OUTPUT_PATH_ROLE, None)

        info = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.information", info)

        window.on_file_double_clicked(item)

        info.assert_called_once()
        assert window.plot_windows == []

    def test_shows_critical_when_load_fails(
        self, window: MainWindow, monkeypatch, tmp_path
    ) -> None:
        missing = tmp_path / "missing.xye"
        item = QListWidgetItem("done")
        item.setData(OUTPUT_PATH_ROLE, missing)

        critical = MagicMock()
        monkeypatch.setattr(f"{MODULE}.QMessageBox.critical", critical)

        window.on_file_double_clicked(item)

        critical.assert_called_once()
        assert window.plot_windows == []

    def test_opens_plot_window_on_success(
        self, qtbot, window: MainWindow, tmp_path
    ) -> None:
        xye_file = tmp_path / "result.xye"
        xye_file.write_text("0.1 10 1\n0.2 12 1\n0.3 9 1\n")

        item = QListWidgetItem("done")
        item.setData(OUTPUT_PATH_ROLE, xye_file)

        window.on_file_double_clicked(item)

        assert len(window.plot_windows) == 1
        plot_win = window.plot_windows[0]
        qtbot.addWidget(plot_win)
        assert plot_win.windowTitle() == xye_file.name


# =========================
# MainWindow construction argument validation
# =========================


class TestMainWindowConstruction:
    def test_raises_without_settings_or_path(self) -> None:
        with pytest.raises(ValueError, match="Either settings or settings_path"):
            MainWindow()

    def test_settings_columns_clamped_to_minimum_one(self, qtbot, settings) -> None:
        win = MainWindow(settings=settings, settings_columns=0)
        qtbot.addWidget(win)
        assert win.settings_columns == 1

    def test_loads_settings_from_path(self, qtbot, tmp_path, settings) -> None:
        import json

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings.model_dump(mode="json")))

        win = MainWindow(settings_path=settings_file, settings_columns=2)
        qtbot.addWidget(win)

        assert win.settings_model == settings
