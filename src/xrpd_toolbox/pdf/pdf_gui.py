"""PyQt6 GUI for the PDF (pair distribution function) pipeline in pdf.py.

Load a raw .xy file, tune every PDFConfig parameter via sliders, spin boxes,
checkboxes and combo boxes, and watch I(Q)/S(Q)/F(Q)/G(r) update live.
Save the config (.json) and every curve (.xye) with one click.

Run with: python pdf_gui.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from xrpd_toolbox.core import ScatteringData
from xrpd_toolbox.pdf.pdf import ExportFormat, PDFConfig, PDFResult, compute_pdf
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula

RECOMPUTE_DEBOUNCE_MS = 300


# --------------------------------------------------------------------------- #
# Reusable parameter widgets
# --------------------------------------------------------------------------- #
class SliderSpin(QWidget):
    """A slider + spin box pair for a bounded float, kept in sync."""

    valueChanged = pyqtSignal(float)  # noqa
    _STEPS = 2000

    def __init__(
        self,
        lo: float,
        hi: float,
        value: float,
        decimals: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(10 ** (-decimals))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 3)
        layout.addWidget(self.spin, 1)

        self.set_value(value, emit=False)
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_edited)

    def _slider_moved(self, raw: int) -> None:
        value = self._lo + (raw / self._STEPS) * (self._hi - self._lo)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _spin_edited(self, value: float) -> None:
        raw = round((value - self._lo) / (self._hi - self._lo) * self._STEPS)
        self.slider.blockSignals(True)
        self.slider.setValue(int(raw))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def set_value(self, value: float, emit: bool = True) -> None:
        value = min(max(value, self._lo), self._hi)
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(value)
        self.slider.setValue(
            round((value - self._lo) / (self._hi - self._lo) * self._STEPS)
        )
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if emit:
            self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()


class OptionalFloat(QWidget):
    """A "Auto" checkbox next to a SliderSpin -- represents Optional[float].

    Checked ("Auto") means the underlying value is None; unchecked exposes
    the slider for a manual value.
    """

    valueChanged = pyqtSignal(object)  # float | None # noqa

    def __init__(
        self,
        lo: float,
        hi: float,
        default: float,
        decimals: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.auto_box = QCheckBox("Auto")
        self.auto_box.setChecked(True)
        self.slider = SliderSpin(lo, hi, default, decimals)
        self.slider.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.auto_box)
        layout.addWidget(self.slider, 1)

        self.auto_box.toggled.connect(self._auto_toggled)
        self.slider.valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))

    def _auto_toggled(self, is_auto: bool) -> None:
        self.slider.setEnabled(not is_auto)
        self.valueChanged.emit(self.value())

    def value(self) -> float | None:
        return None if self.auto_box.isChecked() else self.slider.value()


def _combo(options: list[str], default: str) -> QComboBox:
    box = QComboBox()
    box.addItems(options)
    box.setCurrentText(default)
    return box


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class PDFMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF Calculator")
        self.resize(1400, 900)

        self.xy_path: Path | None = None
        self.result: PDFResult | None = None
        self.config: PDFConfig | None = None
        # Only preserve an axis's view across a recompute if the *user*
        # actually zoomed/panned it -- otherwise every update would stay
        # clipped to whatever the very first (arbitrary) autoscale picked.
        self._axis_view_locked: dict[object, bool] = {}
        self._suppress_zoom_tracking = False

        self._recompute_timer = QTimer()
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(RECOMPUTE_DEBOUNCE_MS)
        self._recompute_timer.timeout.connect(self._recompute)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_param_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_toolbar())
        outer.addWidget(splitter, 1)
        self.status_label = QLabel("Load a .xy file to begin.")
        outer.addWidget(self.status_label)
        self.setCentralWidget(central)

    # -- top toolbar -------------------------------------------------------
    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        load_button = QPushButton("Load .xy File...")
        load_button.clicked.connect(self._open_file_dialog)
        self.save_button = QPushButton("Save Results...")
        self.save_button.clicked.connect(self._save_results)
        self.save_button.setEnabled(False)
        self.file_label = QLabel("No file loaded")
        layout.addWidget(load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.file_label, 1)
        return bar

    # -- plot panel ----------------------------------------------------------
    def _build_plot_panel(self) -> QWidget:
        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        axes = self.figure.subplots(2, 2)
        self.ax_iq, self.ax_sq, self.ax_fq, self.ax_gr = (
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
        )
        for ax in (self.ax_iq, self.ax_sq, self.ax_fq, self.ax_gr):
            self._axis_view_locked[ax] = False
            self._connect_zoom_tracking(ax)

        home_action = self.toolbar._actions.get("home")  # noqa
        if home_action is not None:
            home_action.triggered.connect(self._unlock_all_axis_views)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return container

    def _connect_zoom_tracking(self, ax) -> None:
        """ax.clear() replaces ax.callbacks with a fresh, empty registry --
        call this again after every clear() or the connection made here is
        silently lost and zoom tracking stops working from then on.
        """
        ax.callbacks.connect("xlim_changed", self._on_axis_view_changed)
        ax.callbacks.connect("ylim_changed", self._on_axis_view_changed)

    def _on_axis_view_changed(self, ax) -> None:
        """Marks an axis as user-zoomed -- ignored while we're the ones
        changing limits (initial autoscale, or restoring a locked view).
        """
        if not self._suppress_zoom_tracking:
            self._axis_view_locked[ax] = True

    def _unlock_all_axis_views(self) -> None:
        """Toolbar's Home button means "back to fit-all" -- go back to
        autoscaling on every future update too, not just this one.
        """
        for ax in self._axis_view_locked:
            self._axis_view_locked[ax] = False

    # -- parameter panel -----------------------------------------------------
    def _build_param_panel(self) -> QWidget:
        self.widgets: dict[str, QWidget] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(430)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._sample_group())
        layout.addWidget(self._q_range_group())
        layout.addWidget(self._polarisation_group())
        layout.addWidget(self._background_group())
        layout.addWidget(self._normalisation_group())
        layout.addWidget(self._r_grid_group())
        layout.addWidget(self._termination_group())
        layout.addWidget(self._real_space_constraint_group())
        layout.addWidget(self._export_group())
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _add_row(
        self, form: QFormLayout, label: str, key: str, widget: QWidget
    ) -> None:
        self.widgets[key] = widget
        form.addRow(label, widget)
        signal = self._change_signal(widget)
        if signal is not None:
            signal.connect(self._schedule_recompute)

    @staticmethod
    def _change_signal(widget: QWidget):
        if isinstance(widget, (SliderSpin, OptionalFloat)):
            return widget.valueChanged
        if isinstance(widget, QCheckBox):
            return widget.toggled
        if isinstance(widget, QComboBox):
            return widget.currentTextChanged
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.valueChanged
        if isinstance(widget, QLineEdit):
            return widget.textChanged
        return None

    def _schedule_recompute(self, *_args) -> None:
        self._recompute_timer.start()

    def _sample_group(self) -> QGroupBox:
        box = QGroupBox("Sample")
        form = QFormLayout(box)

        self.formula_widget = QLineEdit("Si")
        self._add_row(form, "Formula (e.g. SiO2)", "formula", self.formula_widget)

        self.sample_name_widget = QLineEdit("pdf")
        self._add_row(form, "Sample name", "sample_name", self.sample_name_widget)

        self.wavelength_widget = SliderSpin(0.05, 3.0, 0.161669, 6)
        self._add_row(form, "Wavelength (Å)", "wavelength", self.wavelength_widget)

        self.number_density_widget = SliderSpin(0.001, 0.2, 0.05, 6)
        self._add_row(
            form,
            "Number density (atoms/Å³)",
            "number_density",
            self.number_density_widget,
        )
        return box

    def _q_range_group(self) -> QGroupBox:
        box = QGroupBox("Q-range")
        form = QFormLayout(box)

        self.q_min_widget = SliderSpin(0.01, 5.0, 0.5, 3)
        self.q_max_widget = SliderSpin(1.0, 50.0, 24.0, 3)
        self.q_step_widget = OptionalFloat(0.001, 0.5, 0.02, 4)

        self._add_row(form, "q_min (Å⁻¹)", "q_min", self.q_min_widget)
        self._add_row(form, "q_max (Å⁻¹)", "q_max", self.q_max_widget)
        self._add_row(form, "q_step (auto if unset)", "q_step", self.q_step_widget)

        self.auto_q_max_widget = QCheckBox()
        self._add_row(form, "auto_q_max", "auto_q_max", self.auto_q_max_widget)

        self.auto_qmax_snr = SliderSpin(0.1, 10.0, 1.5, 2)
        self.auto_qmax_search_min = SliderSpin(0.1, 20.0, 5.0, 2)
        self._add_row(
            form,
            "auto_q_max SNR threshold",
            "auto_q_max_snr_threshold",
            self.auto_qmax_snr,
        )

        self._add_row(
            form,
            "auto_q_max search min",
            "auto_q_max_search_min",
            self.auto_qmax_search_min,
        )
        return box

    def _polarisation_group(self) -> QGroupBox:
        box = QGroupBox("Polarisation")
        form = QFormLayout(box)
        self.polarisation_factor = QCheckBox()
        self.polarisation_factor.setChecked(True)
        self._add_row(
            form, "polarisation_factor", "polarisation_factor", self.polarisation_factor
        )

        self.is_synchrotron = QCheckBox()
        self.is_synchrotron.setChecked(True)

        self._add_row(form, "is_synchrotron", "is_synchrotron", self.is_synchrotron)

        self.polarisation_p = SliderSpin(0.0, 1.0, 0.99, 3)
        self._add_row(form, "polarisation_p", "polarisation_p", self.polarisation_p)
        return box

    def _background_group(self) -> QGroupBox:
        box = QGroupBox("Background / absorption")
        form = QFormLayout(box)
        self.background_file_edit = QLineEdit()
        browse_row = QWidget()
        browse_layout = QHBoxLayout(browse_row)
        browse_layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(
            lambda: self._browse_into(self.background_file_edit)
        )
        browse_layout.addWidget(self.background_file_edit, 1)
        browse_layout.addWidget(browse_button)
        self.widgets["background_file"] = self.background_file_edit
        self.background_file_edit.textChanged.connect(self._schedule_recompute)
        form.addRow("background_file (optional)", browse_row)

        self.background_scale = SliderSpin(0.0, 5.0, 1.0, 3)
        self._add_row(
            form, "background_scale", "background_scale", self.background_scale
        )

        self.absorption_correction = QCheckBox()
        self._add_row(
            form,
            "absorption_correction",
            "absorption_correction",
            self.absorption_correction,
        )

        self.mu_r = OptionalFloat(0.01, 10.0, 1.0, 3)
        self._add_row(
            form,
            "mu_r (required if absorption)",
            "mu_r",
            self.mu_r,
        )
        return box

    def _normalisation_group(self) -> QGroupBox:
        box = QGroupBox("Normalisation")
        form = QFormLayout(box)
        self.norm_poly_degree = QSpinBox()
        self.norm_poly_degree.setRange(0, 15)
        self.norm_poly_degree.setValue(3)
        self._add_row(
            form, "norm_poly_degree", "norm_poly_degree", self.norm_poly_degree
        )

        self.norm_q_min = OptionalFloat(0.0, 50.0, 15.0, 2)
        self._add_row(
            form,
            "norm_q_min (auto if unset)",
            "norm_q_min",
            self.norm_q_min,
        )

        self.background_type = _combo(
            ["chebyshev", "polynomial", "constant", "linear", "bspline", "cosine"],
            "chebyshev",
        )

        self._add_row(
            form,
            "background_type",
            "background_type",
            self.background_type,
        )

        self.normalisation_method = _combo(
            ["krogh_moe", "eggert", "billinge", "warren"], "krogh_moe"
        )
        self._add_row(
            form,
            "normalisation_method",
            "normalisation_method",
            self.normalisation_method,
        )
        # compute_full_covariance = QCheckBox()
        # compute_full_covariance.setChecked(True)
        # self._add_row(
        #     form,
        #     "compute_full_covariance",
        #     "compute_full_covariance",
        #     compute_full_covariance,
        # )
        # covariance_max_points = QSpinBox()
        # covariance_max_points.setRange(50, 20000)
        # covariance_max_points.setValue(800)
        # self._add_row(
        #     form,
        #     "covariance_max_points",
        #     "covariance_max_points",
        #     covariance_max_points,
        # )
        return box

    def _r_grid_group(self) -> QGroupBox:
        box = QGroupBox("Real-space grid")
        form = QFormLayout(box)

        self.r_min = SliderSpin(0.0, 5.0, 0.0, 3)
        self._add_row(form, "r_min (Å)", "r_min", self.r_min)

        self.r_max = SliderSpin(1.0, 100.0, 30.0, 2)
        self._add_row(form, "r_max (Å)", "r_max", self.r_max)

        self.r_step = SliderSpin(0.001, 0.5, 0.01, 4)
        self._add_row(form, "r_step (Å)", "r_step", self.r_step)

        return box

    def _termination_group(self) -> QGroupBox:
        box = QGroupBox("Termination / damping")
        form = QFormLayout(box)

        self.termination_window = _combo(
            ["soper_lorch", "lorch", "cosine", "none"], "lorch"
        )
        self._add_row(
            form,
            "termination_window",
            "termination_window",
            self.termination_window,
        )

        self.soper_lorch_power = SliderSpin(1.0, 8.0, 2.0, 1)
        self._add_row(
            form, "soper_lorch_power", "soper_lorch_power", self.soper_lorch_power
        )

        self.qdamp = SliderSpin(0.0, 0.2, 0.03, 4)
        self._add_row(form, "qdamp (Å⁻¹)", "qdamp", self.qdamp)

        self.qbroad = SliderSpin(0.0, 0.01, 0.0, 5)
        self._add_row(form, "qbroad", "qbroad", self.qbroad)

        return box

    def _real_space_constraint_group(self) -> QGroupBox:
        box = QGroupBox("Real-space (Toby-Egami) constraint")
        form = QFormLayout(box)

        self.use_real_space_constraint = QCheckBox()
        self.use_real_space_constraint.setChecked(True)
        self._add_row(
            form,
            "use_real_space_constraint",
            "use_real_space_constraint",
            self.use_real_space_constraint,
        )

        self.real_space_constraint_iterations = QSpinBox()
        self.real_space_constraint_iterations.setRange(0, 100)
        self.real_space_constraint_iterations.setValue(10)
        self._add_row(
            form,
            "real_space_constraint_iterations",
            "real_space_constraint_iterations",
            self.real_space_constraint_iterations,
        )

        self.r_constraint_max = OptionalFloat(0.1, 10.0, 2.0, 3)
        self._add_row(
            form,
            "r_constraint_max (auto if unset)",
            "r_constraint_max",
            self.r_constraint_max,
        )

        self.r_constraint_search_min = SliderSpin(0.1, 10.0, 1.2, 2)
        self._add_row(
            form,
            "r_constraint_search_min",
            "r_constraint_search_min",
            self.r_constraint_search_min,
        )

        self.r_constraint_search_max = SliderSpin(0.1, 15.0, 3.5, 2)
        self._add_row(
            form,
            "r_constraint_search_max",
            "r_constraint_search_max",
            self.r_constraint_search_max,
        )

        return box

    def _export_group(self) -> QGroupBox:
        box = QGroupBox("Export formats (for Save Results)")
        layout = QHBoxLayout(box)
        self.export_checks: dict[ExportFormat, QCheckBox] = {}
        for fmt in ExportFormat:
            check = QCheckBox(fmt.value)
            check.setChecked(True)
            self.export_checks[fmt] = check
            layout.addWidget(check)
        return box

    def _browse_into(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select background file")
        if path:
            line_edit.setText(path)

    # -- file loading ----------------------------------------------------------
    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open diffraction data",
            "",
            "XY data (*.xy *.xye *.dat);;All files (*)",
        )
        if not path:
            return
        self.xy_path = Path(path)
        self.file_label.setText(str(self.xy_path))
        self._unlock_all_axis_views()
        self._recompute()

    # -- config assembly / recompute --------------------------------------------
    def _build_config(self) -> PDFConfig:
        formula = self.formula_widget.text().strip()
        if not formula:
            raise ValueError("Formula is required")
        composition = ChemicalFormula(formula=formula)

        if self.xy_path is not None and Path(self.xy_path).exists():
            data = ScatteringData.from_xye(
                filepath=self.xy_path,
                x_unit="tth",
                data_type="xray",
                wavelength=self.wavelength_widget.value(),
            )
        else:
            raise FileNotFoundError("File not found:{self.xy_path}")

        background_file_text = self.background_file_edit.text().strip()

        return PDFConfig(
            composition=composition,
            sample_name=self.sample_name_widget.text().strip() or "pdf",
            data=data,
            number_density=self.number_density_widget.value(),
            q_min=self.q_min_widget.value(),
            q_max=self.q_max_widget.value(),
            q_step=self.q_step_widget.value(),
            polarisation_factor=self.polarisation_factor.isChecked(),
            is_synchrotron=self.is_synchrotron.isChecked(),
            polarisation_p=self.polarisation_p.value(),
            background_file=Path(background_file_text)
            if background_file_text
            else None,
            background_scale=self.background_scale.value(),
            absorption_correction=self.absorption_correction.isChecked(),
            mu_r=self.mu_r.value(),
            norm_poly_degree=self.norm_poly_degree.value(),
            norm_q_min=self.norm_q_min.value(),
            background_type=self.background_type.currentText(),  # type: ignore
            normalisation_method=self.normalisation_method.currentText(),  # type: ignore
            # compute_full_covariance=self.compute_full_covariance.isChecked(),
            # covariance_max_points=self.covariance_max_points.value(),
            auto_q_max=self.auto_q_max_widget.isChecked(),
            auto_q_max_snr_threshold=self.auto_qmax_snr.value(),
            auto_q_max_search_min=self.auto_qmax_search_min.value(),
            r_min=self.r_min.value(),
            r_max=self.r_max.value(),
            r_step=self.r_step.value(),
            termination_window=(
                None
                if self.termination_window.currentText() == "none"  # type: ignore
                else self.termination_window.currentText()
            ),
            soper_lorch_power=self.soper_lorch_power.value(),
            qdamp=self.qdamp.value(),
            qbroad=self.qbroad.value(),
            use_real_space_constraint=self.use_real_space_constraint.isChecked(),
            real_space_constraint_iterations=self.real_space_constraint_iterations.value(),
            r_constraint_max=self.r_constraint_max.value(),
            r_constraint_search_min=self.r_constraint_search_min.value(),
            r_constraint_search_max=self.r_constraint_search_max.value(),
        )

    def _recompute(self) -> None:
        if self.xy_path is None:
            return
        try:
            config = self._build_config()
            result = compute_pdf(self.xy_path, config)
        except Exception as exc:  # noqa: BLE001 - surfaced to the status bar, not fatal
            self.status_label.setText(f"Error: {exc}")
            traceback.print_exc()
            self.save_button.setEnabled(False)
            return

        self.config = config
        self.result = result
        self.status_label.setText(
            f"OK -- {config.sample_name}: {len(result.q)} Q points"
        )
        self.save_button.setEnabled(True)
        self._update_plots()

    # -- plotting ------------------------------------------------------------
    def _update_plots(self) -> None:
        assert self.result is not None
        result = self.result
        eline = 0.1
        axes = (self.ax_iq, self.ax_sq, self.ax_fq, self.ax_gr)

        preserved_limits = {
            ax: (ax.get_xlim(), ax.get_ylim())
            for ax in axes
            if self._axis_view_locked.get(ax, False)
        }

        self._suppress_zoom_tracking = True
        try:
            for ax in axes:
                ax.clear()
                self._connect_zoom_tracking(ax)

            self.ax_iq.errorbar(
                result.iq.x, result.iq.y, result.iq.e, elinewidth=eline, label="I(Q)"
            )
            # self.ax_iq.plot(
            #     result.iq.x,
            #     result.background_normalised,
            #     "r--",
            #     lw=0.9,
            #     label="background",
            # )
            PDFResult.style_axis(
                self.ax_iq,
                "Q (Å⁻¹)",
                "I(Q) (e.u.)",
                "Intensity (electron units)",
                legend=True,
            )

            self.ax_sq.errorbar(result.sq.x, result.sq.y, result.sq.e, elinewidth=eline)
            self.ax_sq.axhline(1.0, color="k", lw=0.6, ls="--", label="S(Q) = 1")
            PDFResult.style_axis(
                self.ax_sq, "Q (Å⁻¹)", "S(Q)", "Total structure function", legend=True
            )

            self.ax_fq.errorbar(result.fq.x, result.fq.y, result.fq.e, elinewidth=eline)
            self.ax_fq.axhline(0.0, color="k", lw=0.6, ls="--")
            PDFResult.style_axis(
                self.ax_fq, "Q (Å⁻¹)", "F(Q) (Å⁻¹)", "Reduced structure function"
            )

            self.ax_gr.errorbar(result.gr.x, result.gr.y, result.gr.e, elinewidth=eline)
            plot_r = result.r[result.r < 5]
            self.ax_gr.plot(
                plot_r,
                result.baseline[result.r < 5],
                "k--",
                lw=0.8,
                label=r"$-4\pi r\rho_0$",
            )
            PDFResult.style_axis(
                self.ax_gr,
                "r (Å)",
                "G(r) (Å⁻²)",
                "Pair distribution function",
                legend=True,
            )

            for ax, (xlim, ylim) in preserved_limits.items():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            self.figure.tight_layout()
            # Synchronous draw, not draw_idle(): autoscale-driven xlim/ylim
            # changes are resolved at actual draw time, not when plot()/
            # errorbar() is called -- draw_idle() would defer that past the
            # `finally` below and get it misread as a user zoom.
            self.canvas.draw()
        finally:
            self._suppress_zoom_tracking = False

    # -- saving ----------------------------------------------------------------
    def _save_results(self) -> None:
        if self.result is None or self.config is None:
            QMessageBox.warning(self, "Nothing to save", "Compute a result first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
        output_dir = Path(output_dir)

        selected_formats = [
            fmt for fmt, check in self.export_checks.items() if check.isChecked()
        ]
        try:
            self.result.save_results(
                export_formats=selected_formats, output_dir=output_dir
            )
            config_path = output_dir / f"{self.config.sample_name}_config.json"
            config_path.write_text(self.config.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            traceback.print_exc()
            return

        self.status_label.setText(
            f"Saved {[f.value for f in selected_formats]} + config to {output_dir}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = PDFMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
