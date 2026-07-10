import numpy as np
import pytest

from xrpd_toolbox.pdf.pdf import normalise_intensity


def test_krogh_moe_refines_scale_and_background_independently():
    q_values = np.linspace(1.0, 9.0, 50)
    self_scattering = q_values**2 + 1.0
    true_background = 0.25 + 0.03 * q_values
    true_scale = 2.3
    intensity_q = true_scale * self_scattering + true_background

    scale_factor, background = normalise_intensity(
        q_values,
        intensity_q,
        f_sq_mean=self_scattering,
        compton=np.zeros_like(q_values),
        q_min_fit=1.0,
        q_max_fit=float(q_values.max()),
        poly_degree=1,
        background_type="linear",
        method="krogh_moe",
        f_mean_sq=self_scattering,
        rho=1.0,
        r_min=0.0,
        r_step=0.1,
        termination_window="none",
        qdamp=0.0,
    )

    expected_scale = np.sum(q_values**2 * self_scattering * intensity_q) / np.sum(
        q_values**2 * self_scattering**2
    )
    assert scale_factor == pytest.approx(expected_scale, rel=1e-10)
    assert background.shape == q_values.shape


def test_known_scale_factor_preserves_scale_and_fits_background():
    q_values = np.linspace(1.0, 9.0, 50)
    self_scattering = q_values**2 + 1.0
    true_background = 0.5 + 0.02 * q_values
    true_scale = 1.7
    intensity_q = true_scale * self_scattering + true_background

    scale_factor, background = normalise_intensity(
        q_values,
        intensity_q,
        f_sq_mean=self_scattering,
        compton=np.zeros_like(q_values),
        q_min_fit=1.0,
        q_max_fit=float(q_values.max()),
        poly_degree=1,
        background_type="linear",
        method="krogh_moe",
        f_mean_sq=self_scattering,
        rho=1.0,
        r_min=0.0,
        r_step=0.1,
        termination_window="none",
        qdamp=0.0,
        known_scale_factor=true_scale,
    )

    assert scale_factor == pytest.approx(true_scale, rel=1e-8)
    assert background == pytest.approx(true_background, rel=1e-6)
