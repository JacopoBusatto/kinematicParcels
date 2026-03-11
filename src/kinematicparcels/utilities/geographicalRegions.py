# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:27:19 2024

@author: Jacopo Busatto
"""
import pandas as pd
import numpy  as np

class Region:
    def __init__(self, name, label, numericLabel, bounds, priority=0):
        """
        Inizializza una regione.

        Args:
            name (str): Nome completo della regione.
            label (str): Etichetta della regione.
            bounds (list[dict]): Lista di dizionari con i confini delle regioni (min e max di latitudine e longitudine).
            priority (int): Priorità della regione, di default 0 (facoltativo).
        """
        self.name         = name
        self.label        = label
        self.bounds       = bounds
        self.priority     = priority
        self.NumericLabel = numericLabel

    def contains(self, x, y):
        """
        Verifica se il punto (x, y) è contenuto nella regione.

        Args:
            x (float): Longitudine del punto.
            y (float): Latitudine del punto.

        Returns:
            bool: True se il punto è all'interno della regione, False altrimenti.
        """
        for i in range(len(self.bounds)):
            # Estrai le liste di confini per questo rettangolo
            lon_min_list = self.bounds[i]["lon_min"]
            lon_max_list = self.bounds[i]["lon_max"]
            lat_min_list = self.bounds[i]["lat_min"]
            lat_max_list = self.bounds[i]["lat_max"]

            # Itera su ciascun rettangolo definito dalle liste
            for lon_min, lon_max, lat_min, lat_max in zip(lon_min_list, lon_max_list, lat_min_list, lat_max_list):
                # Verifica se il punto è contenuto in questo rettangolo
                if lon_min <= x <= lon_max and lat_min <= y <= lat_max:
                    return True

        return False
    
    def __repr__(self):
        # Ritorna una rappresentazione stringa utile dell'oggetto Region
        return f"Region(name='{self.name}', label='{self.label}', priority={self.priority})"

ALL_REGIONS = [
        Region("Adriatic Sea 1",                      "adr1",   7, [{"lon_min": [  12.00],                                                                            "lon_max": [  18.50],                                                                          "lat_min": [ 42.58],                                                                            "lat_max": [ 46.00]}],                                                                             priority=3),
        Region("Adriatic Sea 2",                      "adr2",   8, [{"lon_min": [18, 16.3, 13],                                                                       "lon_max": [21.88, 18, 16.3],                                                                  "lat_min": [40.1, 40.51, 41.31],                                                                "lat_max": [42.58, 42.58, 42.58]}],                                                                priority=3),
        Region("Adriatic Sea",                        "adr",    8, [{"lon_min": [12, 18, 16.3, 13],                                                                   "lon_max": [20, 21.88, 18, 16.3],                                                              "lat_min": [42.58, 40.1, 40.51, 41.31],                                                         "lat_max": [46, 42.58, 42.58, 42.58]}],                                                            priority=2),
        Region("Aegean Sea 1",                        "aeg",    9, [{"lon_min": [  21.88, 27.78],                                                                     "lon_max": [  27.78, 30.15],                                                                   "lat_min": [ 35.3, 40.15],                                                                      "lat_max": [ 41.5, 41.15]}],                                                                       priority=3),
        Region("Alboral Sea",                         "alb",    1, [{"lon_min": [-  6.00],                                                                            "lon_max": [-  1.00],                                                                          "lat_min": [ 34.00],                                                                            "lat_max": [ 39.00]}],                                                                             priority=3),
        Region("Ionian Sea 1",                        "ion1",  10, [{"lon_min": [   9.20],                                                                            "lon_max": [  15.00],                                                                          "lat_min": [ 32.30],                                                                            "lat_max": [ 36.72]}],                                                                             priority=3),
        Region("Ionian Sea 2",                        "ion2",  11, [{"lon_min": [  15.00],                                                                            "lon_max": [  21.88],                                                                          "lat_min": [ 30.00],                                                                            "lat_max": [ 36.72]}],                                                                             priority=3),
        Region("Ionian Sea 3",                        "ion3",  12, [{"lon_min": [  15.00, 16.14, 16.3, 16.3],                                                         "lon_max": [  21.88, 21.88, 21.88, 18.4],                                                      "lat_min": [ 36.72, 38.1, 38.7, 40.1],                                                          "lat_max": [ 38.1, 38.7, 40.1, 40.51]}],                                                           priority=3),
        Region("Sicily Channel",                      "sic",   17, [{"lon_min": [11.15, 11.43, 11.71, 11.99, 12.27, 12.55, 12.82, 13.1, 13.38, 13.66, 13.94, 14.22],  "lon_max": [11.43, 11.71, 11.99, 12.27, 12.55, 12.82, 13.1, 13.38, 13.66, 13.94, 14.22, 14.5], "lat_min": [36.69, 36.54, 36.38, 36.23, 36.07, 35.91, 35.9, 36.04, 36.19, 36.34, 36.5 , 36.65], "lat_max": [37.01, 37.16, 37.32, 37.47, 37.63, 37.79, 37.8 , 37.65, 37.48, 37.31, 37.14, 36.97]}], priority=4),
        Region("Levantine Sea 1",                     "lev1",  13, [{"lon_min": [  21.88],                                                                            "lon_max": [  26.20],                                                                          "lat_min": [ 30.00],                                                                            "lat_max": [ 35.30]}],                                                                             priority=3),
        Region("Levantine Sea 2",                     "lev2",  14, [{"lon_min": [  26.2, 27.78],                                                                      "lon_max": [  33, 33],                                                                         "lat_min": [ 33.5, 35.3],                                                                       "lat_max": [ 35.3, 38]}],                                                                          priority=3),
        Region("Levantine Sea 3",                     "lev3",  15, [{"lon_min": [  26.20],                                                                            "lon_max": [  33.00],                                                                          "lat_min": [ 30.00],                                                                            "lat_max": [ 33.50]}],                                                                             priority=3),
        Region("Levantine Sea 4",                     "lev4",  16, [{"lon_min": [  33.00],                                                                            "lon_max": [  37.00],                                                                          "lat_min": [ 31.00],                                                                            "lat_max": [ 38.00]}],                                                                             priority=3),
        Region("Levantine Sea",                       "lev",   16, [{"lon_min": [  21.88, 27.78],                                                                     "lon_max": [ 37, 37],                                                                          "lat_min": [ 30, 35.3],                                                                         "lat_max": [ 35.3, 38]}],                                                                          priority=2),
        Region("North West Mediterranean",            "nwm",    4, [{"lon_min": [-  1.00],                                                                            "lon_max": [   9.20],                                                                          "lat_min": [ 39.50],                                                                            "lat_max": [ 45.00]}],                                                                             priority=3),
        Region("South West Mediterranean 1",          "swm1",   2, [{"lon_min": [-  1.00],                                                                            "lon_max": [   3.00],                                                                          "lat_min": [ 35.50],                                                                            "lat_max": [ 39.50]}],                                                                             priority=3),
        Region("South West Mediterranean 2",          "swm2",   3, [{"lon_min": [   3.00],                                                                            "lon_max": [   9.20],                                                                          "lat_min": [ 35.50],                                                                            "lat_max": [ 39.50]}],                                                                             priority=3),
        Region("Tyrrhenian Sea 1",                    "tyr1",   5, [{"lon_min": [9.2, 9.2],                                                                           "lon_max": [13, 10.4],                                                                         "lat_min": [41.31, 43.7],                                                                       "lat_max": [43.7, 44.4]}],                                                                         priority=3),
        Region("Tyrrhenian Sea 2",                    "tyr2",   6, [{"lon_min": [9.2, 9.2, 9.2],                                                                      "lon_max": [15., 16.14, 16.3],                                                                 "lat_min": [ 36.72, 38.1, 38,7],                                                                "lat_max": [38.1, 38.7, 41.31]}],                                                                  priority=3),
        Region("Mediterranean Sea",                   "med",    3, [{"lon_min": [-6, 27, 2],                                                                          "lon_max": [27, 39, 20],                                                                       "lat_min": [29, 29, 43],                                                                        "lat_max": [43, 41.1, 46]}],                                                                       priority=1),
        # BLACK SEA
        Region("Black Sea",                           "bs",    17, [{"lon_min": [  27.30],                                                                            "lon_max": [  42.50],                                                                          "lat_min": [ 41.10],                                                                            "lat_max": [ 47.50]}],                                                                             priority=1),
        # ATLANTIC OCEAN
        Region("Atlantic Ocean",                      "AO",     1, [{"lon_min": [-70, -70, -70, 8-9, -98, -90, -84, -78.3, -82.5],                                    "lon_max": [25, -6, 2, 25, -70, -70, -70, -75.5, -79.7],                                       "lat_min": [-80, 29, 43, 48.5, 18, 14, 9.5, 8.2, 8.8],                                          "lat_max": [29, 43, 48.5, 90, 48.5, 18, 14, 9.5, 9.5]}],                                           priority=1),
        Region("North Atlantic sub-tropical gyre",    "NAstg",  2, [{"lon_min": [-70, -70, -70,      -98,         ],                                                  "lon_max": [25, -6, 2,     -70,         ],                                                     "lat_min": [ 20, 29, 43,       20,        ],                                                    "lat_max": [29, 43, 48.5,     48.5        ]}],                                                     priority=2),
        Region("Equatorial Atlantic current system",  "AEcs",   3, [{"lon_min": [-70,                -98, -90, -84, -78.3, -82.5],                                    "lon_max": [25,            -70, -70, -70, -75.5, -79.7],                                       "lat_min": [-20,               18, 14, 9.5, 8.2, 8.8],                                          "lat_max": [20,                 20, 18, 14, 9.5, 9.5]}],                                           priority=2),
        Region("Southern Atlantic sub-tropical gyre", "SAstg",  4, [{"lon_min": [-70.00],                                                                             "lon_max": [25.00],                                                                            "lat_min": [-47.00],                                                                            "lat_max": [-20.00]}],                                                                             priority=2),
        Region("Amazon River basin",                  "ARb",    5, [{"lon_min": [-70.00],                                                                             "lon_max": [-40.00],                                                                           "lat_min": [- 1.00],                                                                            "lat_max": [ 20.00]}],                                                                             priority=3),
        Region("Amazon River estuary",                "ARest",  5, [{"lon_min": [-50.00],                                                                             "lon_max": [-45.00],                                                                           "lat_min": [- 2.00],                                                                            "lat_max": [  6.00]}],                                                                             priority=3),

        # NORDIC SEA
        Region("Nordic Sea",                          "NS",     1, [{"lon_min": [-98, -180, 25, 134],                                                                 "lon_max": [25, -98, 134, 180],                                                                "lat_min": [48.5, 65.9, 55, 65.9],                                                              "lat_max": [90, 90, 90, 90]}],                                                                     priority=2),
        # INDIAN OCEAN                
        Region("Indian Ocean",                        "IO",     2, [{"lon_min": [25, 25, 25, 25, 25, 25],                                                             "lon_max": [147, 142, 103, 100.5, 99.8, 99],                                                   "lat_min": [-80, -20, - 3, 5.3, 6.8, 9.1],                                                      "lat_max": [-20, -3, 5.3, 6.8, 9.1, 30]}],                                                         priority=1),
        Region("Southern Indian sub-tropical gyre",   "SIstg",  3, [{"lon_min": [25                    ],                                                             "lon_max": [147                           ],                                                   "lat_min": [-50                         ],                                                      "lat_max": [-20                       ]}],                                                         priority=2),
        Region("Equatorial Indian current system",    "EIcs",   4, [{"lon_min": [    25, 25, 25, 25, 25],                                                             "lon_max": [     142, 103, 100.5, 99.8, 99],                                                   "lat_min": [     -20, - 3, 5.3, 6.8, 9.1],                                                      "lat_max": [     -3, 5.3, 6.8, 9.1, 30]}],                                                         priority=2),
        # PACIFIC OCEAN
        # Region("Pacific Ocean",                      "PO",      3, [{"lon_min": [147, 142, 103, 100.5, 99, -180], "lon_max": [180, 180, 180, 103, 100.5,-70], "lat_min": [-80, -20, -3, 5.3, 7.5, -80], "lat_max": [65.9, 65.9, 65.9, 15, 15, 8]}], priority=1),
        Region("Pacific Ocean",                       "PO",     3, [{"lon_min": [147, 142, 103, 100.5, 99, -180, -180, -80, -180, -180, -180, -180],                  "lon_max": [180, 147, 142, 103, 100.5, -70, -77.5, -78, -83, -84, -90, -98],                   "lat_min": [-80, -20, -3, 5.3, 7.5, -80, 8, 8.7, 8.7, 10, 14, 18],                              "lat_max": [65.9, 65.9, 65.9, 15, 15,  8, 8.7, 9.2, 10, 14, 18, 65.9]}],                           priority=1),
        Region("Southern Pacific sub-tropical gyre",  "SPstg",  4, [{"lon_min": [147,                      -180                                   ],                  "lon_max": [180,                       -70                                ],                   "lat_min": [-55,                    -55                         ],                              "lat_max": [ -20,                    -20                            ]}],                           priority=2),
        Region("Equatorial Pacific current system",   "EPcs",   5, [{"lon_min": [147, 142, 103, 100.5, 99, -180, -180, -80, -180, -180, -180, -180],                  "lon_max": [180, 147, 142, 103, 100.5, -70, -77.5, -78, -83, -84, -90, -98],                   "lat_min": [-20, -20, -3, 5.3, 7.5, -20, 8, 8.7, 8.7, 10, 14, 18],                              "lat_max": [20  , 20  , 20  , 15, 15,  8, 8.7, 9.2, 10, 14, 18, 20  ]}],                           priority=2),
        Region("North Pacific sub-tropical gyre",     "NPstg",  6, [{"lon_min": [147, 142, 103,                                               -180],                  "lon_max": [180, 147, 142,                                             -98],                   "lat_min": [ 20,  20, 20,                                     20],                              "lat_max": [45  , 45  , 45  ,                                   45  ]}],                           priority=2),
        Region("North Pacific sub-polar gyre",        "NPspg",  7, [{"lon_min": [147, 142, 103,                                               -180],                  "lon_max": [180, 147, 142,                                             -98],                   "lat_min": [ 45,  45, 45,                                     45],                              "lat_max": [65.9, 65.9, 65.9,                                   65.9]}],                           priority=2),
        # Region("Pacific Ocean 2",                    "PO2",     3, [{"lon_min": [-180, -180, -80, -180, -180, -180, -180], "lon_max": [-70, -77.5, -78, -83, -84, -90, -98], "lat_min": [-80, 8, 8.7, 8.7, 10, 14, 18], "lat_max": [8, 8.7, 9.2, 10, 14, 18, 65.9]}], priority=1),
        # SOUTHERN OCEAN
        Region("Southern Ocean",                      "SO",     5, [{"lon_min": [-180.00],                                                                            "lon_max": [180.00],                                                                           "lat_min": [-80.00],                                                                            "lat_max": [-47.00]}],                                                                             priority=2)
]

class RegionManager:
    def __init__(self, regions_list=None):
        """
        Inizializza un gestore delle regioni.

        Args:
            regions_list (list): Lista di regioni da includere. Se None, vengono caricate tutte le regioni da ALL_REGIONS.
        """
        self.regions = []  # Inizializza una lista vuota di regioni
        # Se non viene passata una lista, usa la lista di tutte le regioni
        if regions_list is None:
            regions_to_add = ALL_REGIONS
        else:
            # Altrimenti, prendi solo le regioni i cui nomi sono nella lista
            regions_to_add = [region for region in ALL_REGIONS if region.name in regions_list]
        
        # Aggiungi le regioni selezionate al gestore
        self._add_regions(regions_to_add)

    def _add_regions(self, regions_to_add):
        """Aggiunge le regioni al gestore."""
        for region in regions_to_add:
            self.add_region(region)  # Chiama il metodo add_region per aggiungere ogni regione

    def add_region(self, region):
        """Aggiunge una regione al gestore."""
        # Aggiungi la regione alla lista delle regioni
        self.regions.append(region)
        
    def get_regions(self):
        """Restituisce la lista delle regioni nel gestore."""
        return self.regions

    def __repr__(self):
        # Mostra i nomi delle regioni nel manager
        return f"RegionManager(regions={', '.join([region.name for region in self.regions])})"
    
    def find_regions(self, x, y, howMany="first", priority_level = None): # VA AGGIUNTO PRIORITY MAX O MIN forse con first e priority se la cavamo
        """
        Trova le regioni che contengono il punto (x, y) con un'opzione di filtro per livello di priorità.

        Args:
            x (float): Longitudine del punto.
            y (float): Latitudine del punto.
            howMany (str): Se "first", restituisce solo la prima regione trovata, 
                        se "all", restituisce tutte le regioni trovate, 
                        se "priority", restituisce la regione con la priorità più alta.
            priority_level (int, opzionale): Se specificato, considera solo le regioni con questa priorità.

        Returns:
            list[dict] o dict: Dizionario contenente `label` e `numericLabel` o lista di questi, a seconda del parametro `howMany`.
        """
        matching_regions = [
            region for region in self.regions if region.contains(x, y)
        ]
        if priority_level is not None:
            matching_regions = [
                region for region in matching_regions if region.priority == priority_level
        ]

        if howMany == "first":
            if matching_regions:
                return {"label": matching_regions[0].label, "numericLabel": matching_regions[0].NumericLabel}
            return None  # Se nessuna regione è trovata
        
        elif howMany == "all":
            return [
                {"label": region.label, "numericLabel": region.NumericLabel} for region in matching_regions
            ]
        
        elif howMany == "priority":
            if matching_regions:
                # Restituisce la regione con la priorità più alta
                highest_priority_region = max(matching_regions, key=lambda r: r.priority)
                return {"label": highest_priority_region.label, "numericLabel": highest_priority_region.NumericLabel}
            return None  # Se nessuna regione è trovata
        
        else:
            raise ValueError('Parametro "howMany" deve essere "first", "all" o "priority"')

    

def classify_trajectories(df, region_manager, id_col='id', x_col='X', y_col='Y', howMany="first", priority_level=None):
    """
    Classifica le regioni di partenza e arrivo per ogni traiettoria in un DataFrame,
    con opzione di selezione per livello di priorità.

    Args:
        df (pd.DataFrame): DataFrame con colonne per identificativo (id), longitudine (X) e latitudine (Y).
        region_manager (RegionManager): Gestore delle regioni.
        id_col (str): Nome della colonna degli identificativi delle traiettorie.
        x_col (str): Nome della colonna delle longitudini.
        y_col (str): Nome della colonna delle latitudini.
        howMany (str): "first" per ottenere solo la prima regione, 
                       "all" per ottenere tutte le regioni,
                       "priority" per ottenere la regione con la priorità più alta.
        priority_level (int, opzionale): Se specificato, considera solo le regioni con questa priorità.

    Returns:
        pd.DataFrame: DataFrame con colonne `id`, `start_region`, `start_numericLabel`, `end_region`, `end_numericLabel`.
    """
    results = []

    # Raggruppa per id
    grouped = df.groupby(id_col)

    for traj_id, group in grouped:
        # Ordina per il percorso (opzionale, se i dati non sono già ordinati)
        group = group.sort_index()

        # Estrai punti di partenza e arrivo (scalari)
        start_x, start_y = group.iloc[0][x_col], group.iloc[0][y_col]
        end_x, end_y = group.iloc[-1][x_col], group.iloc[-1][y_col]

        # Classifica le regioni con il parametro `howMany`
        start_region = region_manager.find_regions(float(start_x), float(start_y), howMany, priority_level)
        end_region   = region_manager.find_regions(float(end_x),   float(end_y),   howMany, priority_level)

        # Se `start_region` o `end_region` sono None, gestiscilo correttamente
        start_label   = start_region["label"]        if start_region else None
        start_numeric = start_region["numericLabel"] if start_region else None
        end_label     = end_region["label"]          if end_region   else None
        end_numeric   = end_region["numericLabel"]   if end_region   else None

        # Aggiungi il risultato
        results.append({
            id_col: traj_id,
            'start_region': start_label,
            'start_numericLabel': start_numeric,
            'end_region': end_label,
            'end_numericLabel': end_numeric
        })

    return pd.DataFrame(results)



def classify_full_trajectory(df, region_manager, id_col='id', x_col='X', y_col='Y', howMany="first", priority_level=None):
    """
    Classifica la regione di ciascun punto in una traiettoria, assegnando la regione corrispondente a ogni punto.

    Args:
        df (pd.DataFrame): DataFrame con colonne per identificativo (id), longitudine (X) e latitudine (Y).
        region_manager (RegionManager): Gestore delle regioni.
        id_col (str): Nome della colonna degli identificativi delle traiettorie.
        x_col (str): Nome della colonna delle longitudini.
        y_col (str): Nome della colonna delle latitudini.
        howMany (str): "first" per ottenere solo la prima regione, 
                       "all" per ottenere tutte le regioni,
                       "priority" per ottenere la regione con la priorità più alta.
        priority_level (int, opzionale): Se specificato, considera solo le regioni con questa priorità.

    Returns:
        pd.DataFrame: DataFrame con colonne `id`, `point_id`, `region_label`, `numericLabel`.
    """
    results = []

    # Raggruppa per id
    grouped = df.groupby(id_col)

    for traj_id, group in grouped:
        # Ordina per il percorso (opzionale, se i dati non sono già ordinati)
        group = group.sort_index()

        # Aggiungi un punto alla volta
        for idx, row in group.iterrows():
            point_x, point_y = row[x_col], row[y_col]

            # Classifica la regione con il parametro `howMany` e `priority_level`
            region = region_manager.find_regions(float(point_x), float(point_y), howMany, priority_level)

            # Se la regione è trovata, estrai il label e numericLabel
            if region:
                region_label  = region["label"]
                numeric_label = region["numericLabel"]
            else:
                region_label  = None
                numeric_label = None

            # Aggiungi il risultato per ogni punto
            results.append({
                id_col        : traj_id,
                'point_id'    : idx,  # Aggiunge l'indice del punto
                'region_label': region_label,
                'numericLabel': numeric_label,
                'age'         : row["age"]
            })

    return pd.DataFrame(results)


def create_region_mask(ds, region_manager, lonName = "longitude", latName = "latitude"):
    """
    Crea una maschera booleana per selezionare i dati in base alle regioni desiderate.
    
    Args:
        ds (xarray.Dataset): Il dataset contenente i dati.
        region_manager (RegionManager): Gestore delle regioni che contiene solo le regioni desiderate.
        
    Returns:
        numpy.ndarray: Maschera booleana che seleziona i dati nelle regioni specificate.
    """
    mask = np.zeros((len(ds[latName]), len(ds[lonName])), dtype=bool)

    for region in region_manager.regions:
        for bounds in region.bounds:
            lon_min_list = bounds["lon_min"]
            lon_max_list = bounds["lon_max"]
            lat_min_list = bounds["lat_min"]
            lat_max_list = bounds["lat_max"]
            
            for lon_min, lon_max, lat_min, lat_max in zip(lon_min_list, lon_max_list, lat_min_list, lat_max_list):
                # Usa meshgrid per ottenere una griglia di latitudine e longitudine
                lon_grid, lat_grid = np.meshgrid(ds[lonName], ds[latName])
                
                # Confronta le coordinate in base ai limiti di regione
                mask |= (lon_grid >= lon_min) & (lon_grid <= lon_max) & (lat_grid >= lat_min) & (lat_grid <= lat_max)
    
    return mask

def get_region_by_label(label, regions=None):
    """
    Restituisce la regione con la label richiesta.

    Parameters
    ----------
    label : str
        Etichetta regione, es. 'NPstg'.
    regions : list[Region] or None
        Lista di regioni da cercare. Se None, usa ALL_REGIONS.

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

    for region in regions:
        if region.label == label:
            return region

    available = ", ".join(r.label for r in regions)
    raise ValueError(f"Regione con label '{label}' non trovata. Label disponibili: {available}")


def make_regular_grid_in_region(
    region,
    dlon,
    dlat,
    *,
    include_edges=True,
    deduplicate=True,
    sort_points=True,
):
    """
    Crea una griglia regolare di punti (lon, lat) dentro una regione.

    La regione è definita come unione di uno o più rettangoli nei bounds.

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
        Se True rimuove eventuali duplicati dovuti a rettangoli sovrapposti.
    sort_points : bool, default True
        Se True ordina i punti per latitudine e poi longitudine.

    Returns
    -------
    lons : np.ndarray
        Longitudini dei punti.
    lats : np.ndarray
        Latitudini dei punti.
    """
    if dlon <= 0 or dlat <= 0:
        raise ValueError("dlon e dlat devono essere positivi")

    points = []

    for bounds in region.bounds:
        lon_min_list = bounds["lon_min"]
        lon_max_list = bounds["lon_max"]
        lat_min_list = bounds["lat_min"]
        lat_max_list = bounds["lat_max"]

        for lon_min, lon_max, lat_min, lat_max in zip(
            lon_min_list, lon_max_list, lat_min_list, lat_max_list
        ):
            if include_edges:
                lon_vals = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
                lat_vals = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
            else:
                lon_vals = np.arange(lon_min + dlon, lon_max, dlon)
                lat_vals = np.arange(lat_min + dlat, lat_max, dlat)

            for lat in lat_vals:
                for lon in lon_vals:
                    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                        points.append((float(lon), float(lat)))

    if len(points) == 0:
        return np.array([]), np.array([])

    if deduplicate:
        # arrotondamento per evitare duplicati numerici dovuti a floating point
        points = list({(round(lon, 10), round(lat, 10)) for lon, lat in points})

    if sort_points:
        points = sorted(points, key=lambda p: (p[1], p[0]))

    lons = np.array([p[0] for p in points], dtype=float)
    lats = np.array([p[1] for p in points], dtype=float)

    return lons, lats


def make_regular_grid_from_label(label, dlon, dlat, **kwargs):
    """
    Wrapper: cerca la regione da label e costruisce la griglia regolare.
    """
    region = get_region_by_label(label)
    return make_regular_grid_in_region(region, dlon, dlat, **kwargs)