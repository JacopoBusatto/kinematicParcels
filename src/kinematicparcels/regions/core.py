# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:27:19 2024

@author: Jacopo Busatto
"""
import numpy as np

from .definitions import REGIONS_DATA

# -------------------------------------------------------------------------
# Longitude utilities
# -------------------------------------------------------------------------

def lon_to_360(lon):
    """
    Convert longitude(s) to the [0, 360) convention.
    Works with scalars or numpy arrays.
    """
    import numpy as np
    return np.mod(lon, 360.0)


def lon_to_180(lon):
    """
    Convert longitude(s) to the [-180, 180) convention.
    Works with scalars or numpy arrays.
    """
    import numpy as np
    return (np.mod(lon + 180.0, 360.0) - 180.0)


def convert_lon(lon, from_mode="-180_180", to_mode="-180_180"):
    """
    Convert longitude(s) between coordinate conventions.

    Parameters
    ----------
    lon : float or ndarray
        Longitude(s) to convert.
    from_mode : str
        "-180_180" or "0_360"
    to_mode : str
        "-180_180" or "0_360"
    """

    if from_mode == to_mode:
        return lon

    if to_mode == "0_360":
        return lon_to_360(lon)

    if to_mode == "-180_180":
        return lon_to_180(lon)

    raise ValueError(f"Unsupported longitude mode: {to_mode}")



class Region:
    def __init__(
        self,
        name,
        label,
        numericLabel,
        bounds=None,
        polygons=None,
        priority=0,
        lon_mode="-180_180",
    ):
        """
        Inizializza una regione.

        Parameters
        ----------
        name : str
            Complete name of the region.
        label : str
            Short label of the region.
        numericLabel : int
            Numeric label of the region.
        bounds : list[dict] or None
            Old rectangular definition: list of rectangles, each defined by a dictionary with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
            For the sake of compatibility with old framework.
        polygons : list[list[tuple[float, float]]] or None
            New geometric definition: list of polygons.
            Each polygon is a list of vertices (lon, lat).
        priority : int, default 0
            Region priority.
        lon_mode : str, default "-180_180"
            Longitude convention used in the polygons and bounds. Can be "-180_180" or "0_360".
        """
        self.name         = name
        self.label        = label
        self.bounds       = bounds if bounds is not None else []
        self.polygons     = polygons if polygons is not None else []
        self.priority     = priority
        self.NumericLabel = numericLabel
        self.lon_mode     = lon_mode

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _point_on_segment(px, py, x1, y1, x2, y2, eps=1e-12):
        """
        True se il punto P=(px,py) giace sul segmento [(x1,y1),(x2,y2)].
        Utile per includere i bordi come interni.
        """
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) > eps:
            return False

        return (
            min(x1, x2) - eps <= px <= max(x1, x2) + eps
            and min(y1, y2) - eps <= py <= max(y1, y2) + eps
        )
    
    @classmethod
    def _point_in_polygon(cls, x, y, polygon, include_boundary=True):
        """
        Test point-in-polygon con ray casting.
        """
        if polygon is None or len(polygon) < 3:
            return False

        if polygon[0] != polygon[-1]:
            poly = list(polygon) + [polygon[0]]
        else:
            poly = polygon

        inside = False

        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]

            if include_boundary and cls._point_on_segment(x, y, x1, y1, x2, y2):
                return True

            intersects = (y1 > y) != (y2 > y)
            if intersects:
                x_intersect = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) + 1e-300)
                if x < x_intersect:
                    inside = not inside

        return inside

    def _contains_in_bounds(self, x, y):
        """
        Compatibilità con la vecchia logica a rettangoli.
        """
        for i in range(len(self.bounds)):
            lon_min_list = self.bounds[i]["lon_min"]
            lon_max_list = self.bounds[i]["lon_max"]
            lat_min_list = self.bounds[i]["lat_min"]
            lat_max_list = self.bounds[i]["lat_max"]

            for lon_min, lon_max, lat_min, lat_max in zip(
                lon_min_list, lon_max_list, lat_min_list, lat_max_list
            ):
                if lon_min <= x <= lon_max and lat_min <= y <= lat_max:
                    return True

        return False


    def _contains_in_polygons(self, x, y):
        """
        True se il punto cade in almeno un poligono della regione.
        """
        for polygon in self.polygons:
            if self._point_in_polygon(x, y, polygon, include_boundary=True):
                return True
        return False


    def contains(self, x, y, input_lon_mode="-180_180"):
        """
        Verifica se il punto (x, y) è contenuto nella regione.

        La longitudine in input viene convertita nel sistema nativo
        della regione prima del test geometrico.

        Parameters
        ----------
        x : float
            Longitudine del punto.
        y : float
            Latitudine del punto.
        input_lon_mode : str, default "-180_180"
            Convenzione della longitudine in input:
            "-180_180" oppure "0_360".

        Returns
        -------
        bool
        """
        x_native = float(convert_lon(x, from_mode=input_lon_mode, to_mode=self.lon_mode))
        y_native = float(y)

        if self.polygons:
            return self._contains_in_polygons(x_native, y_native)

        return self._contains_in_bounds(x_native, y_native)

    def get_bbox(self):
        """
        Calcola il bounding box della regione.

        Supporta sia definizioni tramite bounds sia tramite polygons.

        Returns
        -------
        lon_min, lon_max, lat_min, lat_max : float
        """

        lon_vals = []
        lat_vals = []

        # --- bounds ---
        for bounds in self.bounds:
            lon_vals.extend(bounds["lon_min"])
            lon_vals.extend(bounds["lon_max"])
            lat_vals.extend(bounds["lat_min"])
            lat_vals.extend(bounds["lat_max"])

        # --- polygons ---
        for poly in self.polygons:
            for lon, lat in poly:
                lon_vals.append(lon)
                lat_vals.append(lat)

        if len(lon_vals) == 0:
            raise ValueError(f"Region '{self.label}' has no geometry defined")

        return (
            float(np.min(lon_vals)),
            float(np.max(lon_vals)),
            float(np.min(lat_vals)),
            float(np.max(lat_vals)),
        )

    def __repr__(self):
        return (
            f"Region(name='{self.name}', label='{self.label}', "
            f"priority={self.priority}, lon_mode='{self.lon_mode}')"
        )



ALL_REGIONS = [Region(**cfg) for cfg in REGIONS_DATA]



class RegionManager:
    def __init__(self, regions_list=None):
        """
        Inizializza un gestore delle regioni.

        Parameters
        ----------
        regions_list : None, list[str], or list[Region]
            - None: usa ALL_REGIONS
            - list[str]: interpreta gli elementi come nomi di regione
            - list[Region]: usa direttamente gli oggetti Region passati
        """
        self.regions = []

        if regions_list is None:
            regions_to_add = ALL_REGIONS

        elif len(regions_list) == 0:
            regions_to_add = []

        elif all(isinstance(r, Region) for r in regions_list):
            regions_to_add = regions_list

        elif all(isinstance(r, str) for r in regions_list):
            regions_to_add = [
                region for region in ALL_REGIONS
                if region.name in regions_list
            ]

        else:
            raise TypeError(
                "regions_list deve essere None, una lista di nomi (str), "
                "oppure una lista di oggetti Region"
            )

        self._add_regions(regions_to_add)

    def _add_regions(self, regions_to_add):
        """Aggiunge le regioni al gestore."""
        for region in regions_to_add:
            self.add_region(region)

    def add_region(self, region):
        """Aggiunge una regione al gestore."""
        self.regions.append(region)

    def get_regions(self):
        """Restituisce la lista delle regioni nel gestore."""
        return self.regions

    def __repr__(self):
        return f"RegionManager(regions={', '.join([region.name for region in self.regions])})"

    def find_regions(
        self,
        x,
        y,
        howMany="first",
        priority_level=None,
        priority_mode="exact",
        input_lon_mode="-180_180",
    ):
        """
        Trova le regioni che contengono il punto (x, y), con opzioni di filtro
        per priorità.

        Parameters
        ----------
        x : float
            Longitudine del punto.
        y : float
            Latitudine del punto.
        howMany : str
            "first", "last", "all", "priority_max", "priority_min".
        priority_level : int, optional
            Se specificato, filtra le regioni in base alla priorità.
        priority_mode : str, default "exact"
            Modalità di confronto della priorità:
            - "exact"   -> priority == priority_level
            - "atleast" -> priority >= priority_level
            - "atmost"  -> priority <= priority_level
        input_lon_mode : str, default "-180_180"
            Convenzione della longitudine in input:
            "-180_180" oppure "0_360".

        Returns
        -------
        dict or list[dict] or None
        """
        matching_regions = [
            region
            for region in self.regions
            if region.contains(x, y, input_lon_mode=input_lon_mode)
        ]

        if priority_level is not None:
            if priority_mode == "exact":
                matching_regions = [
                    region for region in matching_regions
                    if region.priority == priority_level
                ]
            elif priority_mode == "atleast":
                matching_regions = [
                    region for region in matching_regions
                    if region.priority >= priority_level
                ]
            elif priority_mode == "atmost":
                matching_regions = [
                    region for region in matching_regions
                    if region.priority <= priority_level
                ]
            else:
                raise ValueError(
                    'Parametro "priority_mode" deve essere '
                    '"exact", "atleast" o "atmost"'
                )

        # ordinamento deterministico:
        # prima per priorità crescente, poi numericLabel, poi label
        matching_regions = sorted(
            matching_regions,
            key=lambda r: (r.priority, r.NumericLabel, r.label)
        )

        def as_dict(region):
            return {
                "label": region.label,
                "numericLabel": region.NumericLabel,
            }

        if not matching_regions:
            return None if howMany != "all" else []

        if howMany == "first":
            return as_dict(matching_regions[0])

        elif howMany == "last":
            return as_dict(matching_regions[-1])

        elif howMany == "all":
            return [as_dict(region) for region in matching_regions]

        elif howMany == "priority_min":
            min_priority = matching_regions[0].priority
            selected = [
                region for region in matching_regions
                if region.priority == min_priority
            ]
            return [as_dict(region) for region in selected]

        elif howMany == "priority_max":
            max_priority = matching_regions[-1].priority
            selected = [
                region for region in matching_regions
                if region.priority == max_priority
            ]
            return [as_dict(region) for region in selected]

        else:
            raise ValueError(
                'Parametro "howMany" deve essere '
                '"first", "last", "all", "priority_min" o "priority_max"'
            )


# -------------------------------------------------------------------------
# Functions to create maps
# -------------------------------------------------------------------------
def create_region_mask(
    ds,
    region_manager,
    lonName="longitude",
    latName="latitude",
    input_lon_mode="-180_180",
):
    """
    Crea una maschera booleana per selezionare i dati in base alle regioni desiderate.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset contenente coordinate lon/lat.
    region_manager : RegionManager
        Gestore delle regioni.
    lonName : str
        Nome della coordinata longitudinale.
    latName : str
        Nome della coordinata latitudinale.
    input_lon_mode : str, default "-180_180"
        Convenzione della longitudine nel dataset.

    Returns
    -------
    numpy.ndarray
        Maschera booleana di shape (lat, lon).
    """
    lon_vals = np.asarray(ds[lonName].values, dtype=float)
    lat_vals = np.asarray(ds[latName].values, dtype=float)

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    mask = np.zeros(lon_grid.shape, dtype=bool)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            x = float(lon_grid[i, j])
            y = float(lat_grid[i, j])

            for region in region_manager.regions:
                if region.contains(x, y, input_lon_mode=input_lon_mode):
                    mask[i, j] = True
                    break

    return mask


def get_region_by_label(label, regions=None):
    """
    Restituisce la regione con la label richiesta.

    Parameters
    ----------
    label : str
        Etichetta regione, es. 'NPstg'.
    regions : list[Region] | RegionManager | None
        Lista di regioni o RegionManager.
        Se None usa ALL_REGIONS.

    Returns
    -------
    Region
        Oggetto Region corrispondente.

    Raises
    ------
    ValueError
        Se la label non viene trovata.
    """

    if regions is None:
        regions = ALL_REGIONS

    # supporta anche RegionManager
    if isinstance(regions, RegionManager):
        regions = regions.regions

    for region in regions:
        if region.label == label:
            return region

    available = ", ".join(r.label for r in regions)
    raise ValueError(
        f"Regione con label '{label}' non trovata. "
        f"Label disponibili: {available}"
    )


def make_regular_grid_in_region(
    region,
    dlon,
    dlat,
    *,
    include_edges=True,
    deduplicate=True,
    sort_points=True,
    output_lon_mode=None,
):
    """
    Crea una griglia regolare di punti (lon, lat) dentro una regione.

    La griglia viene costruita sul bounding box della regione e poi filtrata
    usando region.contains(...), quindi funziona sia con regioni definite
    tramite bounds sia con regioni definite tramite polygons.

    Parameters
    ----------
    region : Region
        Oggetto regione.
    dlon : float
        Passo longitudinale della griglia.
    dlat : float
        Passo latitudinale della griglia.
    include_edges : bool, default True
        Se True include anche i bordi estremi quando possibile.
    deduplicate : bool, default True
        Se True rimuove eventuali duplicati.
    sort_points : bool, default True
        Se True ordina i punti per latitudine e poi longitudine.
    output_lon_mode : str or None, default None
        Convenzione delle longitudini in output.
        Se None, usa region.lon_mode.

    Returns
    -------
    lons : np.ndarray
        Longitudini dei punti.
    lats : np.ndarray
        Latitudini dei punti.
    """
    if dlon <= 0 or dlat <= 0:
        raise ValueError("dlon e dlat devono essere positivi")

    if output_lon_mode is None:
        output_lon_mode = region.lon_mode

    lon_min, lon_max, lat_min, lat_max = region.get_bbox()

    if include_edges:
        lon_vals = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
        lat_vals = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
    else:
        lon_vals = np.arange(lon_min + dlon, lon_max, dlon)
        lat_vals = np.arange(lat_min + dlat, lat_max, dlat)

    points = []

    for lat in lat_vals:
        for lon in lon_vals:
            if region.contains(float(lon), float(lat), input_lon_mode=region.lon_mode):
                points.append((float(lon), float(lat)))

    if len(points) == 0:
        return np.array([]), np.array([])

    if deduplicate:
        points = list({(round(lon, 10), round(lat, 10)) for lon, lat in points})

    if sort_points:
        points = sorted(points, key=lambda p: (p[1], p[0]))

    lons = np.array([p[0] for p in points], dtype=float)
    lats = np.array([p[1] for p in points], dtype=float)

    if output_lon_mode != region.lon_mode:
        lons = convert_lon(lons, from_mode=region.lon_mode, to_mode=output_lon_mode)

    return lons, lats



def make_regular_grid_from_label(label, dlon, dlat, **kwargs):
    """
    Wrapper: cerca la regione da label e costruisce la griglia regolare.
    """
    region = get_region_by_label(label)
    return make_regular_grid_in_region(region, dlon, dlat, **kwargs)


__all__ = [
    "ALL_REGIONS",
    "Region",
    "RegionManager",
    "convert_lon",
    "create_region_mask",
    "get_region_by_label",
    "lon_to_180",
    "lon_to_360",
    "make_regular_grid_from_label",
    "make_regular_grid_in_region",
]