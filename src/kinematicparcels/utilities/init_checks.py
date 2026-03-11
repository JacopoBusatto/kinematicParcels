import numpy as np


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