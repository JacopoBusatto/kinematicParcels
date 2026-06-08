from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest
import xarray as xr

from kinematicparcels.postprocessing.plotting.exponent_maps import plot_exponent_map


def _build_da(values: np.ndarray, *, name: str = "fsle") -> xr.DataArray:
    nlat, nlon = values.shape
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(-2.0, 2.0, nlat),
            "lon": np.linspace(10.0, 14.0, nlon),
        },
        name=name,
    )


def test_plot_exponent_map_log_scale_positive_values(tmp_path) -> None:
    da = _build_da(np.array([[0.1, 0.3], [0.5, 1.0]], dtype=float))
    outpath = tmp_path / "positive_log.png"

    plot_exponent_map(
        da,
        outpath=outpath,
        projection="PlateCarree",
        log_scale=True,
        add_land=False,
        add_coastlines=False,
        add_gridlines=False,
    )

    assert outpath.exists()


def test_plot_exponent_map_log_scale_negative_values(tmp_path) -> None:
    da = _build_da(np.array([[-0.1, -0.3], [-0.5, -1.0]], dtype=float), name="ftle")
    outpath = tmp_path / "negative_log.png"

    plot_exponent_map(
        da,
        outpath=outpath,
        projection="PlateCarree",
        log_scale=True,
        add_land=False,
        add_coastlines=False,
        add_gridlines=False,
    )

    assert outpath.exists()


def test_plot_exponent_map_log_scale_negative_values_use_negative_only_default_bounds(tmp_path, monkeypatch) -> None:
    da = _build_da(np.array([[-0.1, -0.3], [-0.5, -1.0]], dtype=float), name="ftle")
    outpath = tmp_path / "negative_log_bounds.png"
    seen: dict[str, object] = {}

    def _capture_colorbar(mappable, *args, **kwargs):
        seen["norm"] = mappable.norm
        class _DummyColorbar:
            def set_label(self, *_args, **_kwargs):
                return None
        return _DummyColorbar()

    monkeypatch.setattr("matplotlib.pyplot.colorbar", _capture_colorbar)

    plot_exponent_map(
        da,
        outpath=outpath,
        projection="PlateCarree",
        log_scale=True,
        add_land=False,
        add_coastlines=False,
        add_gridlines=False,
    )

    norm = seen["norm"]
    assert isinstance(norm, mcolors.SymLogNorm)
    assert norm.vmax <= 0


def test_plot_exponent_map_log_scale_mixed_sign_values(tmp_path) -> None:
    da = _build_da(np.array([[-1.0, -0.1], [0.2, 1.5]], dtype=float), name="ftle")
    outpath = tmp_path / "mixed_log.png"

    plot_exponent_map(
        da,
        outpath=outpath,
        projection="PlateCarree",
        log_scale=True,
        add_land=False,
        add_coastlines=False,
        add_gridlines=False,
    )

    assert outpath.exists()


def test_plot_exponent_map_log_scale_all_zero_raises(tmp_path) -> None:
    da = _build_da(np.zeros((2, 2), dtype=float), name="ftle")
    outpath = tmp_path / "zero_log.png"

    with pytest.raises(ValueError, match="non-zero"):
        plot_exponent_map(
            da,
            outpath=outpath,
            projection="PlateCarree",
            log_scale=True,
            add_land=False,
            add_coastlines=False,
            add_gridlines=False,
        )
