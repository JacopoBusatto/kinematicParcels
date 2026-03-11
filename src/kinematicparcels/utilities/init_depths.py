import numpy as np


def get_fieldset_depth_values(fieldset):
    """
    Restituisce l'asse depth del fieldset come array 1D pulito.

    Raises
    ------
    ValueError
        Se il fieldset non ha una dimensione depth.
    """
    if not hasattr(fieldset.U.grid, "depth") or fieldset.U.grid.depth is None:
        raise ValueError("Il fieldset non ha una dimensione depth")

    z = np.asarray(fieldset.U.grid.depth, dtype=float).ravel()
    z = z[np.isfinite(z)]

    if z.size == 0:
        raise ValueError("Asse depth vuoto o non valido")

    return z


def summarize_depth_axis(fieldset, *, name="depth axis"):
    """
    Stampa un piccolo summary dell'asse verticale del fieldset.
    """
    z = get_fieldset_depth_values(fieldset)
    zu = np.unique(z)

    if np.all(np.diff(z) >= 0):
        order = "increasing"
    elif np.all(np.diff(z) <= 0):
        order = "decreasing"
    else:
        order = "non-monotonic"

    if np.all(zu >= 0):
        sign = "non-negative"
    elif np.all(zu <= 0):
        sign = "non-positive"
    else:
        sign = "mixed"

    print(f"[{name}] n={len(z)}, unique={len(zu)}")
    print(f"[{name}] min={zu.min():.3f}, max={zu.max():.3f}")
    print(f"[{name}] order={order}, sign={sign}")
    print(f"[{name}] first values={zu[:min(10, len(zu))]}")


def infer_depth_sign_mode(fieldset):
    """
    Inferisce la convenzione di segno dell'asse depth del fieldset.

    Returns
    -------
    str
        'positive_down', 'negative_down' oppure 'mixed'
    """
    z = np.unique(get_fieldset_depth_values(fieldset))

    if np.all(z >= 0):
        return "positive_down"
    if np.all(z <= 0):
        return "negative_down"
    return "mixed"


def adapt_requested_depths_to_field_convention(
    requested_depths,
    fieldset,
    *,
    request_convention="positive_down",
):
    """
    Adatta il segno delle depth richieste alla convenzione del fieldset.
    Non fa snapping ai livelli del file: cambia solo il segno se serve.

    Parameters
    ----------
    requested_depths : array-like
        Depth richieste dall'utente.
    fieldset : parcels.FieldSet
        FieldSet contenente l'asse depth.
    request_convention : {'positive_down', 'as_is'}
        Convenzione con cui interpretare requested_depths.

    Returns
    -------
    np.ndarray
        Depth adattate alla convenzione del fieldset.
    """
    req = np.asarray(requested_depths, dtype=float).ravel()
    sign_mode = infer_depth_sign_mode(fieldset)

    if request_convention == "as_is":
        return req.copy()

    if request_convention != "positive_down":
        raise ValueError("request_convention deve essere 'positive_down' o 'as_is'")

    if sign_mode == "positive_down":
        return req.copy()
    elif sign_mode == "negative_down":
        return -req
    else:
        # caso ambiguo: non forziamo ulteriormente
        return req.copy()


def snap_depths_to_field(
    depths,
    fieldset,
    *,
    method="nearest",
    atol=1e-8,
    remove_duplicates=False,
):
    """
    Adatta le depth ai livelli del fieldset.

    Parameters
    ----------
    depths : array-like
        Depth già adattate nella convenzione del fieldset.
    fieldset : parcels.FieldSet
        FieldSet contenente l'asse depth.
    method : {'nearest', 'exact'}
        Metodo di snapping.
    atol : float
        Tolleranza per il metodo exact.
    remove_duplicates : bool
        Se True rimuove i duplicati dopo lo snapping.

    Returns
    -------
    np.ndarray
        Depth snapped ai livelli del fieldset.
    """
    zfield = np.unique(get_fieldset_depth_values(fieldset))
    depths = np.asarray(depths, dtype=float).ravel()

    mapped = []

    for d in depths:
        if method == "nearest":
            idx = int(np.argmin(np.abs(zfield - d)))
            mapped.append(float(zfield[idx]))
        elif method == "exact":
            idx = np.where(np.abs(zfield - d) <= atol)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Depth {d} non presente esattamente nel fieldset. "
                    f"Livelli disponibili: {zfield}"
                )
            mapped.append(float(zfield[idx[0]]))
        else:
            raise ValueError("method deve essere 'nearest' o 'exact'")

    mapped = np.asarray(mapped, dtype=float)

    if remove_duplicates:
        mapped = np.unique(mapped)

    return mapped


def summarize_release_depths(
    requested_depths,
    used_depths,
    *,
    name="release depths",
):
    """
    Stampa una diagnostica sulle depth richieste e su quelle effettivamente usate.
    """
    req = np.asarray(requested_depths, dtype=float).ravel()
    used = np.asarray(used_depths, dtype=float).ravel()

    print(f"[{name}] requested ({len(req)}): {req}")
    print(f"[{name}] used ({len(used)}): {used}")

    if len(req) == len(used):
        diff = used - req
        print(f"[{name}] used-requested: {diff}")
    else:
        print(f"[{name}] numero livelli cambiato (possibili duplicati rimossi dopo snapping)")


def build_multilevel_release(
    lons2d,
    lats2d,
    requested_depths,
    fieldset,
    *,
    depth_mode="as_requested",
    request_convention="positive_down",
    snap_method="nearest",
    remove_duplicate_depths=True,
    verbose=False,
):
    """
    Costruisce un rilascio multilivello replicando una griglia 2D su più depth.

    Parameters
    ----------
    lons2d, lats2d : array-like
        Griglia orizzontale iniziale.
    requested_depths : array-like
        Profondità richieste dall'utente.
    fieldset : parcels.FieldSet
        FieldSet con asse depth.
    depth_mode : {'as_requested', 'snap_to_field'}
        - as_requested: usa i livelli richiesti, adattando solo il segno se serve
        - snap_to_field: adatta i livelli ai layer del fieldset
    request_convention : {'positive_down', 'as_is'}
        Convenzione con cui interpretare requested_depths.
    snap_method : {'nearest', 'exact'}
        Metodo di snapping se depth_mode='snap_to_field'
    remove_duplicate_depths : bool
        Se True rimuove depth duplicate dopo snapping.
    verbose : bool
        Se True stampa diagnostica.

    Returns
    -------
    lons, lats, depths : np.ndarray
        Array pronti per ParticleSet.from_list(...)
    """
    lons2d = np.asarray(lons2d, dtype=float).ravel()
    lats2d = np.asarray(lats2d, dtype=float).ravel()

    if lons2d.shape != lats2d.shape:
        raise ValueError("lons2d e lats2d devono avere la stessa shape")

    adapted_depths = adapt_requested_depths_to_field_convention(
        requested_depths,
        fieldset,
        request_convention=request_convention,
    )

    if depth_mode == "snap_to_field":
        final_depths = snap_depths_to_field(
            adapted_depths,
            fieldset,
            method=snap_method,
            remove_duplicates=remove_duplicate_depths,
        )
    elif depth_mode == "as_requested":
        final_depths = adapted_depths
    else:
        raise ValueError("depth_mode deve essere 'as_requested' o 'snap_to_field'")

    lons = np.tile(lons2d, len(final_depths))
    lats = np.tile(lats2d, len(final_depths))
    depths = np.repeat(final_depths, len(lons2d))

    if verbose:
        summarize_release_depths(adapted_depths, final_depths)

    return lons, lats, depths