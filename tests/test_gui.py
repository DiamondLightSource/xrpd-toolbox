import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.collections import LineCollection
from PyQt6.QtCore import QItemSelectionModel, QPoint, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)
from pytestqt.qtbot import QtBot

from xrpd_toolbox import BASE_PATH
from xrpd_toolbox.core import XYEData
from xrpd_toolbox.gui import bad_pixel_gui as bpg
from xrpd_toolbox.gui import mythen3_process_gui as pg
from xrpd_toolbox.i11.mythen import PIXELS_PER_MODULE, MythenDataLoader, MythenSettings

TEST_DATA_DIR = BASE_PATH.parent.parent / "tests" / "data" / "i11"
SI_DATA_FILE = TEST_DATA_DIR / "1410696.nxs"
SI_MYTHEN_FILE = TEST_DATA_DIR / "mythen3_1410696.hdf5"
CONFIG_FILE = (
    BASE_PATH.parent.parent / "config" / "i11" / "mythen3_reduction_config.toml"
)


@pytest.fixture
def si_nexus(tmp_path: Path) -> Path:
    # the nexus links to its mythen data file by name, so copy both
    nexus_filepath = tmp_path / SI_DATA_FILE.name
    shutil.copy(SI_DATA_FILE, nexus_filepath)
    shutil.copy(SI_MYTHEN_FILE, tmp_path)
    return nexus_filepath


@pytest.fixture
def loader(si_nexus: Path) -> Iterator[MythenDataLoader]:
    with MythenDataLoader(si_nexus) as data_loader:
        yield data_loader


@pytest.fixture
def message_boxes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Records every message box instead of blocking on it.
    question() always answers Yes"""

    shown: list[tuple[str, str]] = []

    def recorder(kind: str, answer=QMessageBox.StandardButton.Ok):
        def show(*args, **kwargs):
            text = kwargs.get("text", args[2] if len(args) > 2 else "")
            shown.append((kind, str(text)))
            return answer

        return staticmethod(show)

    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, recorder(kind))

    monkeypatch.setattr(
        QMessageBox, "question", recorder("question", QMessageBox.StandardButton.Yes)
    )

    return shown


# =========================
# Bad pixel GUI
# =========================


@pytest.fixture
def bad_pixel_window(
    qtbot: QtBot, loader: MythenDataLoader, tmp_path: Path
) -> bpg.BadModuleMainWindow:
    window = bpg.BadModuleMainWindow(
        data=loader,
        bad_channels={5, 1285},
        bad_channels_save_path=tmp_path / "badchannels.txt",
    )
    qtbot.addWidget(window)
    return window


def make_canvas(
    qtbot: QtBot, loader: MythenDataLoader, bad_channels: set[int], callback=None
) -> bpg.PlotCanvas:
    canvas = bpg.PlotCanvas(loader, bad_channels, callback or (lambda idx: None))
    qtbot.addWidget(canvas)
    return canvas


def click(qtbot: QtBot, button: QPushButton) -> None:
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def right_click_channel(
    qtbot: QtBot, window: bpg.BadModuleMainWindow, channel: float
) -> None:
    """A real right click on the plot, at a channel of the current module"""

    canvas = window.canvas

    # zoom in, so each channel is several screen pixels wide
    canvas.ax.set_xlim(channel - 10, channel + 10)
    canvas.draw()

    # matplotlib display coords start at the bottom left, Qt's at the top left
    x, y = canvas.ax.get_xaxis_transform().transform((channel, 0.5))
    ratio = canvas.device_pixel_ratio
    position = QPoint(round(x / ratio), round((canvas.figure.bbox.height - y) / ratio))

    qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=position)


@pytest.fixture
def shown_bad_pixel_window(
    qtbot: QtBot, bad_pixel_window: bpg.BadModuleMainWindow
) -> bpg.BadModuleMainWindow:
    bad_pixel_window.resize(1500, 900)
    bad_pixel_window.show()
    qtbot.waitExposed(bad_pixel_window)
    return bad_pixel_window


def plotted_traces(canvas: bpg.PlotCanvas) -> list[np.ndarray]:
    return [np.asarray(line.get_ydata()) for line in canvas.ax.get_lines()]


def bad_pixel_markers(canvas: bpg.PlotCanvas) -> np.ndarray:
    """x positions of the red bad pixel lines, in module channels"""

    lines = canvas.bad_pixel_lines
    if lines is None:
        return np.array([])

    assert isinstance(lines, LineCollection)
    return np.sort([segment[0, 0] for segment in lines.get_segments()])


def test_plot_canvas_plots_every_frame_of_the_first_module(
    qtbot: QtBot, loader: MythenDataLoader
):
    canvas = make_canvas(qtbot, loader, set())

    traces = plotted_traces(canvas)

    assert canvas.current_module == 0
    assert canvas.pixels_per_modules == PIXELS_PER_MODULE
    assert len(traces) == loader.n_frames
    for frame, trace in enumerate(traces):
        np.testing.assert_array_equal(trace, loader.data[frame, :PIXELS_PER_MODULE])


@pytest.mark.parametrize("module", [1, 13, 27])
def test_plot_canvas_set_module_plots_that_modules_channels(
    qtbot: QtBot, loader: MythenDataLoader, module: int
):
    canvas = make_canvas(qtbot, loader, set())

    canvas.set_module(module)

    start = module * PIXELS_PER_MODULE
    traces = plotted_traces(canvas)

    assert len(traces) == loader.n_frames
    np.testing.assert_array_equal(
        traces[0], loader.data[0, start : start + PIXELS_PER_MODULE]
    )
    assert canvas.ax.get_title() == f"Mythen Data — Module {module}"


def test_plot_canvas_converts_between_module_and_global_channels(
    qtbot: QtBot, loader: MythenDataLoader
):
    canvas = make_canvas(qtbot, loader, set())
    canvas.set_module(3)

    np.testing.assert_array_equal(
        canvas._module_to_global([0, 10]), [3 * PIXELS_PER_MODULE, 3 * 1280 + 10]
    )
    np.testing.assert_array_equal(canvas._global_to_module([3 * 1280 + 10]), [10])


def test_plot_canvas_rebuilds_the_global_channel_axis_on_module_change(
    qtbot: QtBot, loader: MythenDataLoader
):
    canvas = make_canvas(qtbot, loader, set())
    first_axis = canvas.ax2

    canvas.set_module(1)

    assert canvas.ax2 is not None
    assert canvas.ax2 is not first_axis


class FakeClick:
    def __init__(self, button: int, xdata: float | None):
        self.button = button
        self.xdata = xdata


@pytest.mark.parametrize(
    "click, expected",
    [
        (FakeClick(button=3, xdata=10.4), [2 * 1280 + 10]),
        (FakeClick(button=3, xdata=1278.6), [2 * 1280 + 1279]),
        (FakeClick(button=1, xdata=10), []),  # left click is for zooming
        (FakeClick(button=3, xdata=None), []),  # outside the axes
        (FakeClick(button=3, xdata=-3), []),
        (FakeClick(button=3, xdata=1280), []),  # past the end of the module
    ],
)
def test_plot_canvas_right_click_selects_the_global_channel(
    qtbot: QtBot, loader: MythenDataLoader, click: FakeClick, expected: list
):
    selected: list[int] = []
    canvas = make_canvas(qtbot, loader, set(), selected.append)
    canvas.set_module(2)

    canvas._on_click(click)

    assert selected == expected


def test_plot_canvas_only_marks_bad_pixels_in_the_current_module(
    qtbot: QtBot, loader: MythenDataLoader
):
    canvas = make_canvas(qtbot, loader, {5, 1285, 1300, 2600})

    np.testing.assert_array_equal(bad_pixel_markers(canvas), [5])

    canvas.set_module(1)
    np.testing.assert_array_equal(bad_pixel_markers(canvas), [5, 20])

    canvas.set_module(3)
    assert canvas.bad_pixel_lines is None


def test_plot_canvas_hiding_bad_pixels_removes_the_markers(
    qtbot: QtBot, loader: MythenDataLoader
):
    canvas = make_canvas(qtbot, loader, {5})

    canvas.set_bad_pixels_visible(False)
    assert canvas.bad_pixel_lines is None

    canvas.set_module(0)  # stays hidden after a redraw
    assert canvas.bad_pixel_lines is None

    canvas.set_bad_pixels_visible(True)
    np.testing.assert_array_equal(bad_pixel_markers(canvas), [5])


def test_bad_pixel_window_sets_up_from_the_data(
    bad_pixel_window: bpg.BadModuleMainWindow,
    loader: MythenDataLoader,
    tmp_path: Path,
):
    window = bad_pixel_window

    assert window.module_slider.minimum() == 0
    assert window.module_slider.maximum() == loader.n_modules_in_data - 1
    assert window.n_spin.maximum() == PIXELS_PER_MODULE // 2
    assert window.n_spin.value() == 5

    assert window.file_label.text() == f"File: {loader.filepath}"
    assert window.save_path_label.text() == f"Save path: {tmp_path / 'badchannels.txt'}"

    items = [window.list_widget.item(i) for i in range(window.list_widget.count())]
    assert [item.text() for item in items if item is not None] == ["5", "1285"]
    assert window.module_bad_pixels_box.toPlainText() == "5"
    assert window.module_count_label.text() == "Bad pixels in module: 1"
    assert window.global_count_label.text() == "Total bad pixels (global): 2"


def test_bad_pixel_window_starts_in_zoom_mode(
    qtbot: QtBot, bad_pixel_window: bpg.BadModuleMainWindow
):
    # toolbar.zoom() is deferred until the event loop runs
    qtbot.waitUntil(lambda: bad_pixel_window.toolbar.mode.name == "ZOOM")


def test_bad_pixel_window_menus(bad_pixel_window: bpg.BadModuleMainWindow):
    menu_bar = bad_pixel_window.menuBar()
    assert menu_bar is not None

    actions = {
        menu.title(): [action.text() for action in menu.actions()]
        for top_level in menu_bar.actions()
        if (menu := top_level.menu()) is not None
    }

    assert actions == {
        "File": ["Load Nexus File", "Save", "Save As...", "Load Bad Channels..."],
        "Help": ["Controls"],
    }


def test_bad_pixel_window_slider_changes_module(
    bad_pixel_window: bpg.BadModuleMainWindow, loader: MythenDataLoader
):
    window = bad_pixel_window

    window.module_slider.setValue(1)

    assert window.canvas.current_module == 1
    np.testing.assert_array_equal(
        plotted_traces(window.canvas)[0], loader.data[0, 1280:2560]
    )
    assert window.module_bad_pixels_box.toPlainText() == "1285"

    window.module_slider.setValue(2)
    assert window.module_bad_pixels_box.toPlainText() == "(none)"
    assert window.module_count_label.text() == "Bad pixels in module: 0"


def test_bad_pixel_window_toggle_adds_then_removes_a_channel(
    bad_pixel_window: bpg.BadModuleMainWindow,
):
    window = bad_pixel_window

    window._toggle_index(7)
    assert window.global_bad_channels == {5, 7, 1285}
    assert window.module_bad_pixels_box.toPlainText() == "5, 7"
    assert window.global_count_label.text() == "Total bad pixels (global): 3"
    np.testing.assert_array_equal(bad_pixel_markers(window.canvas), [5, 7])

    window._toggle_index(5)
    assert window.global_bad_channels == {7, 1285}
    np.testing.assert_array_equal(bad_pixel_markers(window.canvas), [7])


def test_bad_pixel_window_right_click_on_the_plot_toggles_a_channel(
    qtbot: QtBot, shown_bad_pixel_window: bpg.BadModuleMainWindow
):
    window = shown_bad_pixel_window
    window.module_slider.setValue(1)

    right_click_channel(qtbot, window, 5)  # 1285 is already bad
    assert window.global_bad_channels == {5}

    right_click_channel(qtbot, window, 300)
    assert window.global_bad_channels == {5, 1280 + 300}
    np.testing.assert_array_equal(bad_pixel_markers(window.canvas), [300])


def test_bad_pixel_window_right_click_works_in_zoom_mode(
    qtbot: QtBot, shown_bad_pixel_window: bpg.BadModuleMainWindow
):
    window = shown_bad_pixel_window
    qtbot.waitUntil(lambda: window.toolbar.mode.name == "ZOOM")

    right_click_channel(qtbot, window, 100)

    assert window.global_bad_channels == {5, 100, 1285}


def test_bad_pixel_window_undo_and_redo_buttons(
    qtbot: QtBot, bad_pixel_window: bpg.BadModuleMainWindow
):
    window = bad_pixel_window

    window._toggle_index(7)
    window._toggle_index(8)

    click(qtbot, window.undo_button)
    assert window.global_bad_channels == {5, 7, 1285}
    click(qtbot, window.undo_button)
    assert window.global_bad_channels == {5, 1285}
    assert window.list_widget.count() == 2

    click(qtbot, window.undo_button)  # nothing left to undo
    assert window.global_bad_channels == {5, 1285}

    click(qtbot, window.redo_button)
    click(qtbot, window.redo_button)
    assert window.global_bad_channels == {5, 7, 8, 1285}

    click(qtbot, window.redo_button)  # nothing left to redo
    assert window.global_bad_channels == {5, 7, 8, 1285}


def test_bad_pixel_window_undo_and_redo_shortcuts(
    qtbot: QtBot, shown_bad_pixel_window: bpg.BadModuleMainWindow
):
    window = shown_bad_pixel_window
    window.activateWindow()
    qtbot.waitActive(window)

    window._toggle_index(7)

    qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.global_bad_channels == {5, 1285}

    qtbot.keyClick(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert window.global_bad_channels == {5, 7, 1285}


def test_bad_pixel_window_new_change_clears_redo(
    bad_pixel_window: bpg.BadModuleMainWindow,
):
    window = bad_pixel_window

    window._toggle_index(7)
    window._undo()
    window._toggle_index(9)
    window._redo()

    assert window.global_bad_channels == {5, 9, 1285}


def test_bad_pixel_window_undo_history_is_limited(
    bad_pixel_window: bpg.BadModuleMainWindow,
):
    window = bad_pixel_window

    for idx in range(100, 130):
        window._toggle_index(idx)

    assert len(window.undo_stack) == window.undo_limit == 20

    for _ in range(50):
        window._undo()

    # only the last 20 changes can be undone
    assert window.global_bad_channels == {5, 1285} | set(range(100, 110))


def test_bad_pixel_window_adds_edge_channels_to_every_module(
    qtbot: QtBot, bad_pixel_window: bpg.BadModuleMainWindow, loader: MythenDataLoader
):
    window = bad_pixel_window
    window.n_spin.setValue(2)

    click(qtbot, window.add_edges_button)

    edges = {
        m * PIXELS_PER_MODULE + i
        for m in range(loader.n_modules_in_data)
        for i in (0, 1, PIXELS_PER_MODULE - 2, PIXELS_PER_MODULE - 1)
    }

    assert window.global_bad_channels == edges | {5, 1285}

    window._undo()  # adding the edges is a single step
    assert window.global_bad_channels == {5, 1285}


def test_bad_pixel_window_toggle_button_hides_the_bad_pixels(
    qtbot: QtBot, bad_pixel_window: bpg.BadModuleMainWindow
):
    window = bad_pixel_window
    button = window.toggle_bad_pixels_button

    assert button.text() == "Hide Bad Pixels"

    click(qtbot, button)
    assert button.text() == "Show Bad Pixels"
    assert window.canvas.show_bad_pixels is False
    assert window.canvas.bad_pixel_lines is None

    click(qtbot, button)
    assert button.text() == "Hide Bad Pixels"
    assert window.canvas.bad_pixel_lines is not None


def test_bad_pixel_window_save_writes_sorted_channels_after_confirming(
    qtbot: QtBot,
    bad_pixel_window: bpg.BadModuleMainWindow,
    message_boxes: list[tuple[str, str]],
    tmp_path: Path,
):
    bad_pixel_window._toggle_index(3)

    click(qtbot, bad_pixel_window.save_button)

    assert [kind for kind, _ in message_boxes] == ["question"]
    assert (tmp_path / "badchannels.txt").read_text() == "3\n5\n1285\n"


def test_bad_pixel_window_save_does_nothing_if_not_confirmed(
    bad_pixel_window: bpg.BadModuleMainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )

    bad_pixel_window._save()

    assert not (tmp_path / "badchannels.txt").exists()


def test_bad_pixel_window_save_without_a_path_asks_for_one(
    qtbot: QtBot,
    loader: MythenDataLoader,
    message_boxes: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    save_path = tmp_path / "chosen.txt"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), "")
    )

    window = bpg.BadModuleMainWindow(data=loader, bad_channels={42})
    qtbot.addWidget(window)
    assert window.save_path_label.text() == (
        "Save path: (not set — will prompt on Save)"
    )

    window._save()

    assert window._current_save_path == save_path
    assert window.save_path_label.text() == f"Save path: {save_path}"
    assert save_path.read_text() == "42\n"


def test_bad_pixel_window_cancelling_save_as_keeps_the_old_path(
    bad_pixel_window: bpg.BadModuleMainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    bad_pixel_window._save_as()

    assert bad_pixel_window._current_save_path == tmp_path / "badchannels.txt"
    assert not (tmp_path / "badchannels.txt").exists()


def test_bad_pixel_window_backs_up_the_default_file_before_overwriting(
    bad_pixel_window: bpg.BadModuleMainWindow,
    message_boxes: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    default_file = tmp_path / "badchannels.txt"
    default_file.write_text("1\n")
    backup_folder = tmp_path / "backups"

    monkeypatch.setattr(bpg, "DEFAULT_BAD_CHANNEL_FILEPATH", str(default_file))
    monkeypatch.setattr(bpg, "BAD_CHANNEL_BACKUP_FOLDER", str(backup_folder))

    bad_pixel_window._save()

    backups = list(backup_folder.glob("badchannels_*.txt"))
    assert len(backups) == 1
    assert backups[0].read_text() == "1\n"
    assert default_file.read_text() == "5\n1285\n"


def test_bad_pixel_window_does_not_back_up_other_files(
    bad_pixel_window: bpg.BadModuleMainWindow,
    message_boxes: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    (tmp_path / "badchannels.txt").write_text("1\n")
    backup_folder = tmp_path / "backups"
    monkeypatch.setattr(bpg, "BAD_CHANNEL_BACKUP_FOLDER", str(backup_folder))

    bad_pixel_window._save()

    assert not backup_folder.exists()


def test_bad_pixel_window_loading_bad_channels_merges_them(
    bad_pixel_window: bpg.BadModuleMainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    bad_channels_file = tmp_path / "more.txt"
    bad_channels_file.write_text("6\n1285\n2000\n")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad_channels_file), "")
    )

    bad_pixel_window.load_bad_channels()

    assert bad_pixel_window.global_bad_channels == {5, 6, 1285, 2000}
    assert bad_pixel_window.list_widget.count() == 4
    np.testing.assert_array_equal(bad_pixel_markers(bad_pixel_window.canvas), [5, 6])


def test_bad_pixel_window_cancelling_load_bad_channels_changes_nothing(
    bad_pixel_window: bpg.BadModuleMainWindow, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))

    bad_pixel_window.load_bad_channels()

    assert bad_pixel_window.global_bad_channels == {5, 1285}


def test_bad_pixel_window_loads_a_new_nexus_file(
    bad_pixel_window: bpg.BadModuleMainWindow, si_nexus: Path, tmp_path: Path
):
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(si_nexus, other)
    shutil.copy(SI_MYTHEN_FILE, other)

    bad_pixel_window.module_slider.setValue(4)
    bad_pixel_window.load_nexus_file(str(other / si_nexus.name))

    assert bad_pixel_window.data.filepath == str(other / si_nexus.name)
    assert bad_pixel_window.canvas.data is bad_pixel_window.data
    assert bad_pixel_window.canvas.current_module == 0
    assert bad_pixel_window.file_label.text() == f"File: {other / si_nexus.name}"
    assert bad_pixel_window.global_bad_channels == {5, 1285}

    bad_pixel_window.data.close()


def test_bad_pixel_window_shows_an_error_for_a_non_nexus_file(
    bad_pixel_window: bpg.BadModuleMainWindow,
    message_boxes: list[tuple[str, str]],
    loader: MythenDataLoader,
    tmp_path: Path,
):
    bad_pixel_window.load_nexus_file(str(tmp_path / "not_nexus.h5"))

    assert len(message_boxes) == 1
    assert message_boxes[0][0] == "information"
    assert bad_pixel_window.data is loader


def test_bad_pixel_window_opens_a_file_dialog_without_data(
    qtbot: QtBot, si_nexus: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        bpg.BadModuleMainWindow, "nexus_file_dialog", lambda self: str(si_nexus)
    )

    window = bpg.BadModuleMainWindow()
    qtbot.addWidget(window)

    assert isinstance(window.data, MythenDataLoader)
    assert window.data.filepath == str(si_nexus)
    window.data.close()


def test_bad_pixel_window_load_from_dialog_does_nothing_on_cancel(
    bad_pixel_window: bpg.BadModuleMainWindow,
    loader: MythenDataLoader,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(bad_pixel_window, "nexus_file_dialog", lambda: "")

    bad_pixel_window.load_nexus_file_from_dialog()

    assert bad_pixel_window.data is loader


def test_bad_pixel_window_shows_the_controls(
    bad_pixel_window: bpg.BadModuleMainWindow, message_boxes: list[tuple[str, str]]
):
    bad_pixel_window._show_controls()

    assert len(message_boxes) == 1
    assert "Ctrl+Z / Ctrl+Y: undo / redo" in message_boxes[0][1]


class FakeApplication:
    """Stands in for QApplication so the entrypoint doesn't
    make a second one or block in the event loop"""

    def __init__(self, argv):
        pass

    def exec(self) -> int:
        return 0


@pytest.fixture
def captured_windows(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setattr(bpg, "QApplication", FakeApplication)
    monkeypatch.setattr(pg, "QApplication", FakeApplication)

    windows = []

    def capturing(window_class):
        class CapturingWindow(window_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                qtbot.addWidget(self)
                windows.append(self)

        return CapturingWindow

    monkeypatch.setattr(bpg, "BadModuleMainWindow", capturing(bpg.BadModuleMainWindow))
    monkeypatch.setattr(pg, "MainWindow", capturing(pg.MainWindow))

    return windows


def test_run_bad_pixel_gui_with_a_bad_channel_file(
    captured_windows: list, si_nexus: Path, tmp_path: Path
):
    bad_channels_file = tmp_path / "badchannels.txt"
    bad_channels_file.write_text("5\n35839\n")

    with pytest.raises(SystemExit) as exit_info:
        bpg.run_bad_pixel_gui(str(si_nexus), str(bad_channels_file))

    assert exit_info.value.code == 0

    (window,) = captured_windows
    assert window.isVisible()
    assert window.data.filepath == str(si_nexus)
    assert window.global_bad_channels == {5, 35839}
    assert window._current_save_path == bad_channels_file

    window.data.close()


def test_run_bad_pixel_gui_falls_back_to_the_default_bad_channels(
    captured_windows: list,
    si_nexus: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    default_file = tmp_path / "default_badchannels.txt"
    default_file.write_text("7\n")
    monkeypatch.setattr(bpg, "DEFAULT_BAD_CHANNEL_FILEPATH", str(default_file))

    with pytest.raises(SystemExit):
        bpg.run_bad_pixel_gui(str(si_nexus))

    (window,) = captured_windows
    assert window.global_bad_channels == {7}
    assert window._current_save_path == default_file

    window.data.close()


def test_run_bad_pixel_gui_starts_empty_if_the_default_is_missing(
    captured_windows: list,
    si_nexus: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        bpg, "DEFAULT_BAD_CHANNEL_FILEPATH", str(tmp_path / "missing.txt")
    )

    with pytest.raises(SystemExit):
        bpg.run_bad_pixel_gui(str(si_nexus))

    (window,) = captured_windows
    assert window.global_bad_channels == set()
    assert window._current_save_path is None

    window.data.close()


# =========================
# Mythen process GUI
# =========================


@pytest.fixture
def settings() -> MythenSettings:
    return MythenSettings(default_counter=1, send_to_ispyb=False)


@pytest.fixture
def process_window(qtbot: QtBot, settings: MythenSettings) -> pg.MainWindow:
    window = pg.MainWindow(settings=settings, settings_columns=2)
    qtbot.addWidget(window)
    return window


W = TypeVar("W", bound=QWidget)


def widget(window: pg.MainWindow, name: str, kind: type[W]) -> W:
    found = window.widgets[name]
    assert isinstance(found, kind)
    return found


def select_files(window: pg.MainWindow, *paths: Path) -> None:
    selection_model = window.tree.selectionModel()
    assert selection_model is not None

    for path in paths:
        selection_model.select(
            window.fs_model.index(str(path)),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows,
        )


def list_texts(window: pg.MainWindow) -> list[str]:
    items = [window.file_list.item(i) for i in range(window.file_list.count())]
    return [item.text() for item in items if item is not None]


def test_processing_worker_reduces_a_step_scan(
    qtbot: QtBot, si_nexus: Path, settings: MythenSettings
):
    worker = pg.ProcessingWorker([si_nexus], settings, suffix="_reprocessed")

    started: list[Path] = []
    finished: list[tuple[Path, Path]] = []
    failed: list[tuple[Path, str]] = []
    worker.file_started.connect(started.append)
    worker.file_finished.connect(lambda path, out: finished.append((path, out)))
    worker.file_failed.connect(lambda path, error: failed.append((path, error)))

    with qtbot.waitSignal(worker.finished, timeout=60_000):
        worker.start()

    expected = si_nexus.parent / "processed" / "1410696_summed_mythen3_reprocessed.xye"

    assert failed == []
    assert started == [si_nexus]
    assert finished == [(si_nexus, expected)]

    xye = XYEData.from_csv(str(expected))
    assert len(xye.x) > 0
    assert np.all(np.diff(xye.x) > 0)


def test_processing_worker_writes_to_the_output_directory(
    si_nexus: Path, settings: MythenSettings, tmp_path: Path
):
    output_dir = tmp_path / "out"
    worker = pg.ProcessingWorker([si_nexus], settings, output_dir=str(output_dir))

    output_path = worker.process_file(si_nexus)

    assert output_path.parent == output_dir
    assert output_path.exists()


def test_processing_worker_reports_failures_and_carries_on(
    qtbot: QtBot, settings: MythenSettings, monkeypatch: pytest.MonkeyPatch
):
    files = [Path("bad.nxs"), Path("good.nxs")]

    def fake_process_file(self, filepath: Path) -> Path:
        if filepath.name == "bad.nxs":
            raise RuntimeError("broken file")
        return filepath.with_suffix(".xye")

    monkeypatch.setattr(pg.ProcessingWorker, "process_file", fake_process_file)

    worker = pg.ProcessingWorker(files, settings)
    finished: list[tuple[Path, Path]] = []
    failed: list[tuple[Path, str]] = []
    worker.file_finished.connect(lambda path, out: finished.append((path, out)))
    worker.file_failed.connect(lambda path, error: failed.append((path, error)))

    with qtbot.waitSignal(worker.finished):
        worker.start()

    assert failed == [(Path("bad.nxs"), "broken file")]
    assert finished == [(Path("good.nxs"), Path("good.xye"))]


def test_processing_worker_only_supports_step_scans(si_nexus: Path):
    settings = MythenSettings(data_reduction_mode="time_resolved")
    worker = pg.ProcessingWorker([si_nexus], settings)

    with pytest.raises(ValueError, match="Not implemented yet"):
        worker.process_file(si_nexus)


def test_plot_window_plots_the_pattern_with_errors(qtbot: QtBot):
    data = XYEData(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([4.0, 5.0, 6.0]),
        e=np.array([0.1, 0.2, 0.3]),
        title="pattern",
        filepath="/some/where/pattern.xye",
    )

    window = pg.PlotWindow(data)
    qtbot.addWidget(window)

    ax = window.figure.axes[0]
    assert window.windowTitle() == "/some/where/pattern.xye"
    assert ax.get_title() == "pattern.xye"
    assert ax.get_xlabel() == "tth (deg)"
    np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), data.y)
    assert len(ax.collections) == 1  # the error bars


def test_plot_window_without_errors(qtbot: QtBot):
    data = XYEData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), title="pattern")

    window = pg.PlotWindow(data, title="My Plot")
    qtbot.addWidget(window)

    ax = window.figure.axes[0]
    assert window.windowTitle() == "My Plot"
    assert len(ax.collections) == 0
    np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), data.x)


def test_main_window_needs_settings(qtbot: QtBot):
    with pytest.raises(ValueError, match="Either settings or settings_path"):
        pg.MainWindow()


def test_main_window_loads_settings_from_a_file(qtbot: QtBot):
    window = pg.MainWindow(settings_path=CONFIG_FILE)
    qtbot.addWidget(window)

    assert window.settings_path == CONFIG_FILE
    assert window.settings_model == MythenSettings.load(CONFIG_FILE)


def test_main_window_has_a_widget_for_every_shown_setting(
    process_window: pg.MainWindow, settings: MythenSettings
):
    widgets = process_window.widgets

    shown = set(MythenSettings.model_fields) - pg.MainWindow.EXCLUDED_FIELDS
    assert shown <= set(widgets)
    assert "send_to_ispyb" not in widgets

    assert isinstance(widgets["active_modules"], QLineEdit)
    assert widgets["active_modules"].text() == str(settings.active_modules)
    assert isinstance(widgets["data_reduction_mode"], QComboBox)
    assert widgets["data_reduction_mode"].currentText() == "step_scan"
    assert isinstance(widgets["rebin_step"], QDoubleSpinBox)
    assert widgets["rebin_step"].value() == pytest.approx(settings.rebin_step)
    assert isinstance(widgets["edge_bad_channels"], QSpinBox)
    assert widgets["edge_bad_channels"].value() == settings.edge_bad_channels
    assert isinstance(widgets["bad_channel_masking"], QCheckBox)
    assert widgets["bad_channel_masking"].isChecked()

    apply_flatfield = widgets["apply_flatfield"]
    assert isinstance(apply_flatfield, QCheckBox)
    assert apply_flatfield.text() == "apply_flatfield"

    for name in pg.MainWindow.SPECIAL_FILE_FIELDS:
        assert widget(process_window, name, QLineEdit).text() == str(
            getattr(settings, name)
        )


def test_main_window_collects_the_settings_it_was_given(
    process_window: pg.MainWindow, settings: MythenSettings
):
    assert process_window.collect_settings() == settings


def test_main_window_collects_edited_settings(process_window: pg.MainWindow):
    widget(process_window, "active_modules", QLineEdit).setText("1, 2 3")
    widget(process_window, "bad_modules", QLineEdit).setText("[]")
    widget(process_window, "data_reduction_mode", QComboBox).setCurrentText(
        "time_resolved"
    )
    widget(process_window, "rebin_step", QDoubleSpinBox).setValue(0.01)
    widget(process_window, "edge_bad_channels", QSpinBox).setValue(3)
    widget(process_window, "normalise", QCheckBox).setChecked(True)
    widget(process_window, "apply_flatfield", QCheckBox).setChecked(True)
    widget(process_window, "flatfield_filepath", QLineEdit).setText(
        "  /path/to/flat.h5  "
    )

    collected = process_window.collect_settings()

    assert collected.active_modules == [1, 2, 3]
    assert collected.bad_modules == []
    assert collected.data_reduction_mode == "time_resolved"
    assert collected.rebin_step == pytest.approx(0.01)
    assert collected.edge_bad_channels == 3
    assert collected.normalise is True
    assert collected.apply_flatfield is True
    assert collected.flatfield_filepath == "/path/to/flat.h5"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[0, 1, 2]", [0, 1, 2]),
        ("0, 1, 2", [0, 1, 2]),
        ("0 1 2", [0, 1, 2]),
        ("  4,5 ,  6 ", [4, 5, 6]),
        ("[]", []),
        ("", []),
    ],
)
def test_parse_int_list(text: str, expected: list[int]):
    assert pg.MainWindow._parse_int_list(text) == expected


def test_parse_int_list_rejects_non_integers():
    with pytest.raises(ValueError, match="Could not parse a list of integers"):
        pg.MainWindow._parse_int_list("1, two, 3")


def test_main_window_output_directory_is_off_by_default(
    process_window: pg.MainWindow,
):
    edit = process_window.widgets[pg.MainWindow.OUTPUT_DIR_FIELD]
    assert isinstance(edit, QLineEdit)

    assert not edit.isEnabled()
    assert process_window.collect_output_dir() is None


def test_main_window_output_directory_once_enabled(
    qtbot: QtBot, process_window: pg.MainWindow
):
    process_window.show()
    qtbot.waitExposed(process_window)

    edit = process_window.widgets[pg.MainWindow.OUTPUT_DIR_FIELD]
    enable = process_window.widgets[pg.MainWindow.OUTPUT_DIR_ENABLED_FIELD]
    assert isinstance(edit, QLineEdit)
    assert isinstance(enable, QCheckBox)

    qtbot.mouseClick(enable, Qt.MouseButton.LeftButton)
    assert edit.isEnabled()
    assert process_window.collect_output_dir() is None  # still blank

    edit.setText("  /my/output  ")
    assert process_window.collect_output_dir() == "/my/output"

    qtbot.mouseClick(enable, Qt.MouseButton.LeftButton)
    assert not edit.isEnabled()
    assert edit.text() == ""
    assert process_window.collect_output_dir() is None


def test_main_window_suffix(process_window: pg.MainWindow):
    assert process_window.collect_suffix() == "_reprocessed"

    edit = process_window.widgets[pg.MainWindow.SUFFIX_FIELD]
    assert isinstance(edit, QLineEdit)
    edit.setText("  _v2 ")

    assert process_window.collect_suffix() == "_v2"


def test_main_window_browse_output_directory(
    process_window: pg.MainWindow, monkeypatch: pytest.MonkeyPatch
):
    edit = process_window.widgets[pg.MainWindow.OUTPUT_DIR_FIELD]
    assert isinstance(edit, QLineEdit)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: "/chosen")
    process_window.browse_output_dir()
    assert edit.text() == "/chosen"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: "")
    process_window.browse_output_dir()
    assert edit.text() == "/chosen"


def test_main_window_browse_file_path(
    process_window: pg.MainWindow, monkeypatch: pytest.MonkeyPatch
):
    edit = process_window.widgets["angcal_filepath"]
    assert isinstance(edit, QLineEdit)
    original = edit.text()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("", ""))
    process_window.browse_file_path("angcal_filepath", "title", "All Files (*)")
    assert edit.text() == original

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("/new.json", ""))
    process_window.browse_file_path("angcal_filepath", "title", "All Files (*)")
    assert edit.text() == "/new.json"


def test_main_window_file_buttons_are_wired_up(
    qtbot: QtBot, process_window: pg.MainWindow, monkeypatch: pytest.MonkeyPatch
):
    browsed: list[str] = []
    opened: list[str] = []
    monkeypatch.setattr(
        process_window, "browse_file_path", lambda name, *a: browsed.append(name)
    )
    monkeypatch.setattr(process_window, "open_file_path", opened.append)

    edit = process_window.widgets["bad_channels_filepath"]
    row = edit.parentWidget()
    assert row is not None

    buttons = {b.text(): b for b in row.findChildren(QPushButton)}
    click(qtbot, buttons["Browse…"])
    click(qtbot, buttons["Open"])

    assert browsed == ["bad_channels_filepath"]
    assert opened == ["bad_channels_filepath"]


@pytest.mark.parametrize(
    "path_text, expected_kind",
    [("", "warning"), ("/does/not/exist.txt", "warning")],
)
def test_main_window_open_file_path_warns_if_there_is_no_file(
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    path_text: str,
    expected_kind: str,
):
    widget(process_window, "angcal_filepath", QLineEdit).setText(path_text)

    process_window.open_file_path("angcal_filepath")

    assert [kind for kind, _ in message_boxes] == [expected_kind]


@pytest.mark.parametrize("opened, expected", [(True, []), (False, ["critical"])])
def test_main_window_open_file_path_opens_existing_files(
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened: bool,
    expected: list[str],
):
    existing = tmp_path / "angcal.json"
    existing.write_text("{}")
    widget(process_window, "angcal_filepath", QLineEdit).setText(str(existing))

    urls = []

    def open_url(url):
        urls.append(url.toLocalFile())
        return opened

    monkeypatch.setattr(pg.QDesktopServices, "openUrl", open_url)

    process_window.open_file_path("angcal_filepath")

    assert urls == [str(existing)]
    assert [kind for kind, _ in message_boxes] == expected


def test_main_window_lists_only_selected_nexus_files(
    process_window: pg.MainWindow, tmp_path: Path
):
    for name in ("a.nxs", "b.nxs", "c.txt"):
        (tmp_path / name).touch()

    select_files(process_window, tmp_path / "a.nxs", tmp_path / "c.txt")
    select_files(process_window, tmp_path / "b.nxs", tmp_path)

    assert process_window.selected_files == [tmp_path / "a.nxs", tmp_path / "b.nxs"]
    assert list_texts(process_window) == ["⏳ a.nxs", "⏳ b.nxs"]

    item = process_window.find_item(tmp_path / "a.nxs")
    assert item is not None
    assert item.data(pg.OUTPUT_PATH_ROLE) is None
    assert process_window.find_item(tmp_path / "c.txt") is None


def test_main_window_process_needs_selected_files(
    qtbot: QtBot, process_window: pg.MainWindow, message_boxes: list[tuple[str, str]]
):
    click(qtbot, process_window.process_btn)

    assert message_boxes == [("warning", "Select at least one .nxs file.")]
    assert process_window.worker is None


def test_main_window_process_rejects_invalid_settings(
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    tmp_path: Path,
):
    (tmp_path / "a.nxs").touch()
    select_files(process_window, tmp_path / "a.nxs")
    widget(process_window, "active_modules", QLineEdit).setText("1, x")

    process_window.process()

    assert [kind for kind, _ in message_boxes] == ["critical"]
    assert process_window.worker is None
    assert process_window.process_btn.isEnabled()


def test_main_window_process_runs_the_worker(
    qtbot: QtBot,
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for name in ("good.nxs", "bad.nxs"):
        (tmp_path / name).touch()

    used: list[tuple[MythenSettings, str | None, str]] = []

    def fake_process_file(self, filepath: Path) -> Path:
        used.append((self.settings, self.output_dir, self.suffix))
        if filepath.name == "bad.nxs":
            raise RuntimeError("broken")
        return filepath.with_suffix(".xye")

    monkeypatch.setattr(pg.ProcessingWorker, "process_file", fake_process_file)

    select_files(process_window, tmp_path / "good.nxs", tmp_path / "bad.nxs")
    click(qtbot, process_window.process_btn)

    assert not process_window.process_btn.isEnabled()

    # the button is enabled again once the worker thread has finished
    qtbot.waitUntil(process_window.process_btn.isEnabled, timeout=10_000)
    assert message_boxes == [("information", "All files processed.")]
    assert process_window.output_paths == {tmp_path / "good.nxs": tmp_path / "good.xye"}
    assert set(list_texts(process_window)) == {"✅ good.nxs  →  good.xye", "❌ bad.nxs"}
    assert used == [(process_window.settings_model, None, "_reprocessed")] * 2


def test_main_window_worker_callbacks_update_the_list(
    process_window: pg.MainWindow, tmp_path: Path
):
    path = tmp_path / "a.nxs"
    path.touch()
    select_files(process_window, path)

    process_window.on_file_started(path)
    assert list_texts(process_window) == ["🔄 a.nxs"]

    process_window.on_file_finished(path, tmp_path / "a.xye")
    assert list_texts(process_window) == ["✅ a.nxs  →  a.xye"]
    item = process_window.find_item(path)
    assert item is not None
    assert item.data(pg.OUTPUT_PATH_ROLE) == tmp_path / "a.xye"

    process_window.on_file_failed(path, "error")
    assert list_texts(process_window) == ["❌ a.nxs"]


def test_main_window_double_click_plots_a_finished_file(
    qtbot: QtBot,
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    tmp_path: Path,
):
    xye = tmp_path / "a.xye"
    xye.write_text("1 2 0.1\n2 3 0.2\n")

    item = QListWidgetItem("a.nxs")
    item.setData(pg.OUTPUT_PATH_ROLE, xye)

    process_window.on_file_double_clicked(item)

    assert message_boxes == []
    (plot_window,) = process_window.plot_windows
    qtbot.addWidget(plot_window)
    assert plot_window.isVisible()
    assert plot_window.windowTitle() == "a.xye"


def test_main_window_double_click_on_an_unfinished_file(
    process_window: pg.MainWindow, message_boxes: list[tuple[str, str]]
):
    item = QListWidgetItem("a.nxs")
    item.setData(pg.OUTPUT_PATH_ROLE, None)

    process_window.on_file_double_clicked(item)

    assert [kind for kind, _ in message_boxes] == ["information"]
    assert process_window.plot_windows == []


def test_main_window_double_click_on_an_unreadable_file(
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    tmp_path: Path,
):
    item = QListWidgetItem("a.nxs")
    item.setData(pg.OUTPUT_PATH_ROLE, tmp_path / "missing.xye")

    process_window.on_file_double_clicked(item)

    assert [kind for kind, _ in message_boxes] == ["critical"]
    assert process_window.plot_windows == []


def test_main_window_double_clicking_the_list_plots(
    qtbot: QtBot,
    process_window: pg.MainWindow,
    message_boxes: list[tuple[str, str]],
    tmp_path: Path,
):
    path = tmp_path / "a.nxs"
    path.touch()
    select_files(process_window, path)

    process_window.show()
    qtbot.waitExposed(process_window)

    item = process_window.find_item(path)
    assert item is not None
    viewport = process_window.file_list.viewport()
    assert viewport is not None
    position = process_window.file_list.visualItemRect(item).center()

    # a real double click starts with a press, which the list needs to see
    qtbot.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=position)

    assert [kind for kind, _ in message_boxes] == ["information"]


def test_run_mythen_process_uses_the_live_config(
    captured_windows: list, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pg, "DEFAULT_LIVE_CONFIG_FILE", str(CONFIG_FILE))

    with pytest.raises(SystemExit) as exit_info:
        pg.run_mythen_process()

    assert exit_info.value.code == 0

    (window,) = captured_windows
    assert window.isVisible()
    expected = MythenSettings.load(CONFIG_FILE)
    expected.send_to_ispyb = False
    assert window.settings_model == expected


def test_run_mythen_process_uses_defaults_without_a_live_config(
    captured_windows: list, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(pg, "DEFAULT_LIVE_CONFIG_FILE", str(tmp_path / "none.toml"))

    with pytest.raises(SystemExit):
        pg.run_mythen_process()

    (window,) = captured_windows
    assert window.settings_model == MythenSettings(send_to_ispyb=False)


def test_run_mythen_process_uses_defaults_if_the_live_config_is_broken(
    captured_windows: list, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    broken = tmp_path / "broken.toml"
    broken.write_text("this is not = valid toml [")
    monkeypatch.setattr(pg, "DEFAULT_LIVE_CONFIG_FILE", str(broken))

    with pytest.raises(SystemExit):
        pg.run_mythen_process()

    (window,) = captured_windows
    assert window.settings_model == MythenSettings(send_to_ispyb=False)
