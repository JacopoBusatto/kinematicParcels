import numpy as np
import xarray as xr


def get_fieldset_domain(fieldset):
    """
    Restituisce i limiti geografici reali del fieldset.
    """
    lonmin = float(np.min(fieldset.U.grid.lon))
    lonmax = float(np.max(fieldset.U.grid.lon))
    latmin = float(np.min(fieldset.U.grid.lat))
    latmax = float(np.max(fieldset.U.grid.lat))

    return {
        "lonmin": lonmin,
        "lonmax": lonmax,
        "latmin": latmin,
        "latmax": latmax,
    }


def mask_inside_domain(
    lons,
    lats,
    fieldset,
    *,
    inclusive=False,
):
    """
    Restituisce una maschera booleana dei punti interni al dominio del fieldset.

    Parameters
    ----------
    lons, lats : array-like
        Coordinate dei punti iniziali.
    fieldset : parcels.FieldSet
        FieldSet Parcels.
    inclusive : bool, default False
        Se True usa <= e >=.
        Se False usa < e > per escludere i bordi esatti.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    dom = get_fieldset_domain(fieldset)

    if inclusive:
        mask = (
            (lons >= dom["lonmin"]) & (lons <= dom["lonmax"]) &
            (lats >= dom["latmin"]) & (lats <= dom["latmax"])
        )
    else:
        mask = (
            (lons > dom["lonmin"]) & (lons < dom["lonmax"]) &
            (lats > dom["latmin"]) & (lats < dom["latmax"])
        )

    return mask


def filter_inside_domain(
    lons,
    lats,
    fieldset,
    *,
    inclusive=False,
    return_mask=False,
):
    """
    Filtra i punti iniziali tenendo solo quelli interni al dominio del fieldset.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    mask = mask_inside_domain(lons, lats, fieldset, inclusive=inclusive)

    lons_ok = lons[mask]
    lats_ok = lats[mask]

    if return_mask:
        return lons_ok, lats_ok, mask
    return lons_ok, lats_ok


def _surface_field_view(field):
    """Return a 2D surface slice from a Parcels field using its normalized axis order."""
    data = np.ma.asarray(field.data)

    if data.ndim == 4:
        return data[0, 0, :, :]
    if data.ndim == 3:
        return data[0, :, :]
    if data.ndim == 2:
        return data
    return None


def _load_surface_arrays_from_source(fieldset):
    """Load surface U/V arrays from source files using named dimensions, then reorder to (lat, lon)."""
    files = getattr(fieldset, "_kp_source_files", None)
    dimensions = getattr(fieldset, "_kp_dimensions", None)
    variables = getattr(fieldset, "_kp_variables", None)

    if not files or dimensions is None or variables is None:
        return None

    with xr.open_dataset(files[0]) as ds:
        u = ds[variables["U"]]
        v = ds[variables["V"]]

        time_dim = dimensions.get("time")
        depth_dim = dimensions.get("depth")
        lat_dim = dimensions["lat"]
        lon_dim = dimensions["lon"]

        if time_dim and time_dim in u.dims:
            u = u.isel({time_dim: 0})
        if time_dim and time_dim in v.dims:
            v = v.isel({time_dim: 0})
        if depth_dim and depth_dim in u.dims:
            u = u.isel({depth_dim: 0})
        if depth_dim and depth_dim in v.dims:
            v = v.isel({depth_dim: 0})

        u = u.transpose(lat_dim, lon_dim)
        v = v.transpose(lat_dim, lon_dim)

        lon_axis = np.asarray(ds[lon_dim].values, dtype=float).ravel()
        lat_axis = np.asarray(ds[lat_dim].values, dtype=float).ravel()
        u_vals = np.asarray(u.values, dtype=float)
        v_vals = np.asarray(v.values, dtype=float)
        ocean_mask_2d = np.isfinite(u_vals) & np.isfinite(v_vals)

    return lon_axis, lat_axis, ocean_mask_2d


def _build_surface_ocean_mask_cache(fieldset):
    """Build a cached nearest-neighbour ocean mask from source files or, if unavailable, the fieldset data."""
    source_arrays = _load_surface_arrays_from_source(fieldset)
    if source_arrays is not None:
        return source_arrays

    lon_axis = np.asarray(fieldset.U.grid.lon, dtype=float).ravel()
    lat_axis = np.asarray(fieldset.U.grid.lat, dtype=float).ravel()

    u2d = _surface_field_view(fieldset.U)
    v2d = _surface_field_view(fieldset.V)

    if u2d is None or v2d is None:
        raise ValueError("Ocean mask metadata not available on fieldset")

    ocean_mask_2d = (~np.ma.getmaskarray(u2d)) & (~np.ma.getmaskarray(v2d))
    ocean_mask_2d &= np.isfinite(np.ma.filled(u2d, np.nan))
    ocean_mask_2d &= np.isfinite(np.ma.filled(v2d, np.nan))
    return lon_axis, lat_axis, ocean_mask_2d


def _get_surface_ocean_mask_cache(fieldset):
    cache = getattr(fieldset, "_kp_surface_ocean_cache", None)
    if cache is None:
        cache = _build_surface_ocean_mask_cache(fieldset)
        setattr(fieldset, "_kp_surface_ocean_cache", cache)
    return cache


def mask_inside_ocean(
    lons,
    lats,
    fieldset,
):
    """
    Boolean mask for points that fall on valid ocean cells.

    A point is considered ocean if the nearest surface U/V cell is finite and
    not masked in the source data. This complements the geographic domain check
    with the model land-sea mask.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")

    if lons.size == 0:
        return np.zeros(0, dtype=bool)

    dom_mask = mask_inside_domain(lons, lats, fieldset, inclusive=True)
    lon_axis, lat_axis, ocean_mask_2d = _get_surface_ocean_mask_cache(fieldset)

    ocean_mask = np.zeros(lons.size, dtype=bool)

    for idx, (lon, lat, inside) in enumerate(zip(lons, lats, dom_mask)):
        if not inside:
            continue

        j = int(np.argmin(np.abs(lon_axis - lon)))
        i = int(np.argmin(np.abs(lat_axis - lat)))
        ocean_mask[idx] = bool(ocean_mask_2d[i, j])

    return ocean_mask


def filter_inside_ocean(
    lons,
    lats,
    fieldset,
    *,
    return_mask=False,
):
    """Filter initial points keeping only those on valid ocean cells."""
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    mask = mask_inside_ocean(lons, lats, fieldset)
    lons_ok = lons[mask]
    lats_ok = lats[mask]

    if return_mask:
        return lons_ok, lats_ok, mask
    return lons_ok, lats_ok


def summarize_initial_points(lons, lats, *, name="initial points"):
    """
    Stampa un piccolo summary dei punti iniziali.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    n = len(lons)

    if n == 0:
        print(f"[{name}] nessun punto")
        return

    print(f"[{name}] n = {n}")
    print(f"[{name}] lon: {lons.min():.3f} .. {lons.max():.3f}")
    print(f"[{name}] lat: {lats.min():.3f} .. {lats.max():.3f}")


def check_initial_points_in_domain(
    lons,
    lats,
    fieldset,
    *,
    inclusive=False,
    verbose=True,
):
    """
    Controlla quanti punti iniziali sono dentro/fuori dal dominio del fieldset.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    mask = mask_inside_domain(lons, lats, fieldset, inclusive=inclusive)

    n_total = len(lons)
    n_ok = int(mask.sum())
    n_bad = int((~mask).sum())

    result = {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_bad": n_bad,
        "mask": mask,
    }

    if verbose:
        dom = get_fieldset_domain(fieldset)
        print("[domain check]")
        print(
            f"fieldset lon = [{dom['lonmin']:.3f}, {dom['lonmax']:.3f}], "
            f"lat = [{dom['latmin']:.3f}, {dom['latmax']:.3f}]"
        )
        print(f"points total = {n_total}")
        print(f"points inside = {n_ok}")
        print(f"points outside = {n_bad}")

    return result