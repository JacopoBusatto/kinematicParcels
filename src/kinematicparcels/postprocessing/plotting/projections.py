import cartopy.crs as ccrs


_SUPPORTED_PROJECTIONS = {
    "PlateCarree": ccrs.PlateCarree,
    "Mercator": ccrs.Mercator,
    "SouthPolarStereo": ccrs.SouthPolarStereo,
    "NorthPolarStereo": ccrs.NorthPolarStereo,
    "Robinson": ccrs.Robinson,
}


def get_projection(name: str) -> ccrs.CRS:
    """
    Return a Cartopy CRS from a projection name.
    """
    if name not in _SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"Unsupported projection '{name}'. "
            f"Supported: {list(_SUPPORTED_PROJECTIONS.keys())}"
        )

    return _SUPPORTED_PROJECTIONS[name]()