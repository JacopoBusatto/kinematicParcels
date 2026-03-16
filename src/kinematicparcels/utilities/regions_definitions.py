"""
Regional polygonal definitions. 
complex region:
dict(
    name="Sicily Channel",
    label="sic",
    numericLabel=17,
    polygons=[
        [
            (11.15, 36.69),
            (11.43, 37.01),
            (14.50, 36.97),
            (13.38, 36.19),
        ],
        [
            (12, 42),
            (12, 43),
            (11, 41)
        ]
    ],
    priority=4,
),
"""




REGIONS_DATA = [
    ## MEDITERRANEAN SEA
    dict(
        name="Adriatic Sea",
        label="adr",
        numericLabel=8,
        polygons=[
            [
                (12.10, 45.50),
                (13.45, 46.00),
                (19.80, 42.20),
                (19.80, 40.35),
                (18.30, 40.35),
                (13.50, 42.50),
                (12.10, 44.00)
            ]
        ],
        priority=2,
    ),
    dict(
        name="Adriatic Sea 1",
        label="adr1",
        numericLabel=7,
        polygons=[
            [
                (12.10, 45.50),
                (13.60, 46.00),
                (18.50, 42.60),
                (14.00, 42.60),
                (12.10, 44.00)
            ]
        ],
        priority=3,
    ),
    dict(
        name="Adriatic Sea 2",
        label="adr2",
        numericLabel=8,
        polygons=[
            [
                (14.00, 42.60),
                (18.50, 42.60),
                (19.80, 41.75),
                (19.80, 40.10),
                (18.40, 40.10),
                (15.50, 41.50)
            ]
        ],
        priority=3,
    ),


    dict(
        name="Sicily Channel",
        label="sic",
        numericLabel=17,
        polygons=[
            [
                (11.15, 36.85),
                (12.85, 37.90),
                (14.70, 36.80),
                (12.80, 35.90),
            ]
        ],
        priority=4,
    ),

    dict(
        name="Aegean Sea 1",
        label="aeg",
        numericLabel=9,
        polygons=[
            [
                (21.90, 41.10),
                (30.00, 41.10),
                (30.00, 40.00),
                (27.80, 40.00),
                (27.80, 35.30),
                (21.90, 35.30),
            ]
        ],
        priority=3,
    ),

    dict(
        name="Alboral Sea",
        label="alb",
        numericLabel=1,
        polygons=[
            [
                (-5.50, 36.50),
                (-1.00, 38.00),
                (-1.00, 35.00),
                (-5.50, 35.00)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Ionian Sea 1",
        label="ion1",
        numericLabel=10,
        polygons=[
            [
                (10.00, 32.30),
                (15.00, 32.30),
                (15.00, 36.72),
                (10.00, 36.72)
            ]
        ],
    priority=3,
    ),

    dict(
        name="Ionian Sea 2",
        label="ion2",
        numericLabel=11,
        polygons=[
            [
                (15.00, 30.00),
                (21.88, 30.00),
                (21.88, 36.72),
                (15.00, 36.72)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Ionian Sea 3",
        label="ion3",
        numericLabel=12,
        polygons=[
            [
                (15.00, 38.10),
                (16.00, 38.10),
                (16.30, 38.75),
                (16.30, 40.55),
                (17.50, 40.55),
                (18.40, 40.10),
                (20.00, 40.10),
                (21.90, 38.75),
                (21.90, 36.70),
                (15.00, 36.70)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 1",
        label="lev1",
        numericLabel=13,
        polygons=[
            [
                (21.90, 35.30),
                (25.00, 35.30),
                (26.20, 35.00),
                (26.20, 31.50),
                (25.00, 31.50),
                (21.90, 32.50)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 2",
        label="lev2",
        numericLabel=14,
        polygons=[
            [
                (26.20, 35.30),
                (27.80, 35.30),
                (27.80, 37.20),
                (31.00, 37.20),
                (33.00, 36.50),
                (33.00, 35.30),
                (33.00, 33.50),
                (26.20, 33.50)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 3",
        label="lev3",
        numericLabel=15,
        polygons=[
            [
                (26.20, 30.75),
                (33.00, 30.75),
                (33.00, 33.50),
                (26.20, 33.50)
            ]
            ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 4",
        label="lev4",
        numericLabel=16,
        polygons=[
            [
                (33.00, 31.00),
                (35.00, 31.00),
                (37.00, 37.00),
                (33.00, 37.00)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea",
        label="lev",
        numericLabel=16,
        polygons=[
            [
                (21.90, 30.75),
                (35.00, 30.75),
                (37.00, 35.30),
                (37.00, 38.00),
                (27.80, 38.00),
                (27.80, 35.30),
                (21.90, 35.30)
            ]
        ],
        priority=2,
    ),

    dict(
        name="North West Mediterranean",
        label="nwm",
        numericLabel=4,
        polygons=[
            [
                (- 1.00, 39.50),
                (  9.20, 39.50),
                (  9.20, 45.00),
                (- 1.00, 43.00)
            ]
        ],
        priority=3,
    ),

    dict(
        name="South West Mediterranean 1",
        label="swm1",
        numericLabel=2,
        polygons=[
            [
                (- 1.00, 35.50),
                (  3.00, 35.50),
                (  3.00, 39.50),
                (- 1.00, 39.50)
            ]
        ],
        priority=3,
    ),

    dict(
        name="South West Mediterranean 2",
        label="swm2",
        numericLabel=3,
        polygons=[
            [
                (  3.00, 36.50),
                (  9.20, 36.50),
                (  9.20, 39.50),
                (  3.00, 39.50)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Tyrrhenian Sea 1",
        label="tyr1",
        numericLabel=5,
        polygons=[
            [
                ( 9.20, 44.70),
                (10.50, 44.00),
                (13.10, 41.30),
                ( 9.20, 41.30)
            ]
            ],
        priority=3,
    ),

    dict(
        name="Tyrrhenian Sea 2",
        label="tyr2",
        numericLabel=6,
        polygons=[
            [
                ( 9.20, 41.30),
                (14.00, 41.30),
                (16.30, 40.00),
                (16.30, 38.40),
                (15.00, 37.90),
                (15.00, 36.70),
                ( 9.20, 36.70)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Mediterranean Sea",
        label="med",
        numericLabel=3,
        polygons=[
            [
                (- 6.00, 37.50),
                (  2.50, 44.00),
                ( 13.00, 46.00),
                ( 15.00, 46.00),
                ( 20.00, 42.00),
                ( 30.00, 41.10),
                ( 37.00, 38.00),
                ( 35.00, 30.00),
                ( 10.00, 30.00),
                (- 6.00, 35.00)
            ]
        ],
        priority=1,
    ),


    ## BLACK SEA
    dict(
        name="Black Sea",
        label="bs",
        numericLabel=17,
        polygons=[
            [
                (27.30, 43.00),
                (29.00, 46.00),
                (32.00, 47.50),
                (40.00, 47.50),
                (42.10, 41.00),
                (38.00, 40.80),
                (30.00, 41.00),
                (28.00, 41.30),
                (27.30, 42.00)
            ]
        ],
        priority=1,
    ),


    ## ATLANTIC OCEAN
    dict(
        name="Atlantic Ocean",
        label="AO",
        numericLabel=1,
        polygons=[
            [
                (-98.00, 90.00),
                ( 25.00, 90.00),
                ( 25.00, 53.00),
                (- 1.00, 43.00),
                (- 6.00, 38.00),
                (- 6.00, 29.00),
                ( 10.00, 15.00),
                ( 25.00,-30.00),
                ( 25.00, -80.00),
                (-70.00, -80.00),
                (-70.00,   0.00),
                (-79.00,   9.40),
                (-81.50,   8.30),
                (-91.00,  18.00),
                (-98.00,  18.00)
            ]
        ],
        priority=1,
    ),

    dict(
        name="North Atlantic sub-tropical gyre",
        label="NAstg",
        numericLabel=2,
        polygons=[
            [
                (-98.00, 30.00),
                (-72.50, 48.50),
                (  2.00, 48.50),
                (- 6.00, 37.00),
                (- 6.00, 29.00),
                ( 10.00, 20.00),
                (-75.50, 20.00),
                (-78.30, 22.00),
                (-82.50, 23.00),
                (-84.30, 22.00),
                (-90.00, 20.00),
                (-92.00, 16.00),
                (-98.00, 18.00)
            ]
        ],
        priority=2,
    ),

    dict(
        name="Equatorial Atlantic current system",
        label="EAcs",
        numericLabel=3,
        polygons=[
            [
                (-90.00, 20.00),
                (-84.30, 22.00),
                (-82.50, 23.00),
                (-78.30, 22.00),
                (-75.50, 20.00),
                (-70.00, 20.00),
                (-10.00, 20.00),
                ( 20.00, 10.00),
                ( 20.00, -20.00),
                (-42.00, -20.00),
                (-70.00,   0.00),
                (-79.00,   9.40),
                (-81.50,   8.30),
                (-91.00,  18.00)
            ]
        ],
        priority=2,
    ),

    dict(
        name="Southern Atlantic sub-tropical gyre",
        label="SAstg",
        numericLabel=4,
        polygons=[
            [
                (-70.00, -47.00),
                ( 25.00, -47.00),
                ( 25.00, -20.00),
                (-50.00, -20.00)
            ]
        ],
        priority=2,
    ),

    dict(
        name="Amazon River basin",
        label="ARb",
        numericLabel=5,
        polygons=[    
            [
                (-70.00, 20.00),
                (-30.00, 20.00),
                (-30.00,-10.00),
                (-40.00,-10.00),
                (-70.00,  0.00)
            ]
        ],
        priority=3,
    ),

    dict(
        name="Amazon River estuary",
        label="ARest",
        numericLabel=5,
        polygons=[
            [
                (-53.00,  6.00),
                (-45.00,  6.00),
                (-45.00, -4.00),
                (-53.00, -4.00)
            ]
        ],
        priority=4,
    ),

    dict(
        name="Nordic Sea",
        label="NS",
        numericLabel=1,
        polygons=[
            [
                (-180.00, 90.00),
                ( 180.00, 90.00),
                ( 180.00, 66.00),
                ( 134.00, 66.00),
                (   0.00, 48.50),
                (- 80.00, 48.50),
                (-120.00, 65.00),
                (-180.00, 65.00)
            ]
        ],
        priority=2,
    ),


    ## INDIAN OCEAN
    dict(
        name="Indian Ocean",
        label="IO",
        numericLabel=2,
        polygons=[
            [
                (25.00, 30.00),
                (49.50, 31.00),
                (119.00, 26.60),
                (121.20, 24.40),
                (121.20, 15.00),
                (125.00,  7.50),
                (127.80,  1.50),
                (134.00,- 1.90),
                (134.30,- 3.40),
                (139.00,- 5.00),
                (142.50,- 9.00),
                (142.50,-15.00),
                (147.00,-37.50),
                (147.00,-75.00),
                ( 25.00,-75.00)
            ]
        ],
        priority=1,
    ),

    dict(
        name="Southern Indian sub-tropical gyre",
        label="SIstg",
        numericLabel=3,
        polygons=[
            [
                ( 25.00,-20.00),
                (147.00,-20.00),
                (147.00,-50.00),
                ( 25.00,-50.00)
            ]
        ],
        priority=2,
    ),

    dict(
        name="Equatorial Indian current system",
        label="EIcs",
        numericLabel=4,
        polygons=[
            [
                (32.00, 30.00),
                (49.50, 31.00),
                (119.00, 26.60),
                (121.20, 24.40),
                (121.20, 15.00),
                (125.00,  7.50),
                (127.80,  1.50),
                (134.00,- 1.90),
                (134.30,- 3.40),
                (139.00,- 5.00),
                (142.50,- 9.00),
                (142.50,-20.00),
                ( 32.00,-20.00)
            ]
        ],
        priority=2,
    ),

    ## PACIFIC OCEAN
    dict(
        name="Pacific Ocean",
        label="PO",
        numericLabel=3,
        polygons=[
            [
                (141.00, 65.90),
                (110.00, 40.00),
                (119.00, 26.60),
                (121.20, 24.40),
                (121.20, 15.00),
                (125.00,  7.50),
                (127.80,  1.50),
                (134.00,- 1.90),
                (134.30,- 3.40),
                (139.00,- 5.00),
                (142.50,- 9.00),
                (142.50,-15.00),
                (147.00,-37.50),
                (147.00,-80.00),
                (290.00,-80.00),
                (290.00,  0.00),
                (281.00,  9.40),
                (278.50,  8.30),
                (269.00, 18.00),
                (262.00, 18.00),
                (230.00, 65.90)
            ]
        ],
        priority=1,
        lon_mode="0_360",
    ),

    dict(
        name="Southern Pacific sub-tropical gyre",
        label="SPstg",
        numericLabel=4,
        polygons=[
            [
                (147.00,-20.00),
                (290.00,-20.00),
                (290.00,-55.00),
                (147.00,-55.00)
            ]
        ],
        priority=2,
        lon_mode="0_360",

    ),

    dict(
        name="Equatorial Pacific current system",
        label="EPcs",
        numericLabel=5,
        polygons=[
            [
                (121.20, 20.00),
                (121.20, 15.00),
                (125.00,  7.50),
                (127.80,  1.50),
                (134.00,- 1.90),
                (134.30,- 3.40),
                (139.00,- 5.00),
                (142.50,- 9.00),
                (142.50,-15.00),
                (147.00,-20.00),
                (290.00,-20.00),
                (290.00,  0.00),
                (281.00,  9.40),
                (278.50,  8.30),
                (269.00, 18.00),
                (262.00, 18.00),
                (262.00, 18.00),
                (255.00, 20.00),
            ]
        ],
        priority=2,
        lon_mode="0_360",
    ),

    dict(
        name="North Pacific sub-tropical gyre",
        label="NPstg",
        numericLabel=6,
        polygons=[
            [
                (130.00, 45.00),
                (110.00, 40.00),
                (119.00, 26.60),
                (121.20, 24.40),
                (121.20, 20.00),
                (260.00, 20.00),
                (240.00, 45.00)
            ]
        ],
        priority=2,
        lon_mode="0_360",
    ),

    dict(
        name="North Pacific sub-polar gyre",
        label="NPspg",
        numericLabel=7,
        polygons=[
            [
                (141.00, 65.90),
                (125.00, 45.00),
                (250.00, 45.00),
                (220.00, 65.90)

            ]
        ],
        priority=2,
        lon_mode="0_360",
    ),

    ### Patagonian Fjords
    dict(
        name="Puerto Natales North Fjord",
        label="PNnf",
        numericLabel=1,
        polygons=[
            [
                (-73.45,-51.60),
                (-73.30,-51.45),
                (-73.00,-51.42),
                (-72.35,-51.70),
                (-72.40,-52.50),
                (-73.10,-52.45),
                (-73.50,-52.30),
                (-73.65,-52.05),
                (-73.65,-52.05),
                (-73.60,-51.85),
                (-73.50,-52.00),
                (-73.35,-52.10),

            ]
        ],
        priority=4,
    ),

    ## SOUTHER OCEAN
    dict(
        name="Southern Ocean",
        label="SO",
        numericLabel=5,
        polygons=[
            [
                (-180.00, -80.00), 
                (180.00, -80.00), 
                (180.00, -47.00), 
                (-180.00, -47.00)
            ]
        ],
        priority=2,
    ),
]
























# -----------------------------------
# OLD RECTANGULAR DEFINITIONS
# -----------------------------------
REGIONS_DATA_RECTANGLES = [
    ## MEDITERRANEAN SEA
    dict(
        name="Adriatic Sea 1",
        label="adr1",
        numericLabel=7,
        bounds=[
            dict(
                lon_min=[12.00], 
                lon_max=[18.50], 
                lat_min=[42.58], 
                lat_max=[46.00]
            )
        ],
        priority=3,
    ),
    dict(
        name="Adriatic Sea 2",
        label="adr2",
        numericLabel=8,
        bounds=[
            dict(
                lon_min=[18,16.3,13],
                lon_max=[21.88,18,16.3], 
                lat_min=[40.1,40.51,41.31], 
                lat_max=[42.58,42.58,42.58]
            )
        ],
        priority=3,
    ),
    dict(
        name="Adriatic Sea",
        label="adr",
        numericLabel=8,
        bounds=[
            dict(
                lon_min=[12,18,16.3,13], 
                lon_max=[20,21.88,18,16.3], 
                lat_min=[42.58,40.1,40.51,41.31], 
                lat_max=[46,42.58,42.58,42.58]
            )
        ],
        priority=2,
    ),

    dict(
        name="Aegean Sea 1",
        label="aeg",
        numericLabel=9,
        bounds=[
            dict(
                lon_min=[21.88,27.78], 
                lon_max=[27.78,30.15], 
                lat_min=[35.3,40.15], 
                lat_max=[41.5,41.15]
            )
        ],
        priority=3,
    ),

    dict(
        name="Alboral Sea",
        label="alb",
        numericLabel=1,
        bounds=[
            dict(
                lon_min=[-6.00], 
                lon_max=[-1.00], 
                lat_min=[34.00], 
                lat_max=[39.00]
            )
        ],
        priority=3,
    ),

    dict(
        name="Ionian Sea 1",
        label="ion1",
        numericLabel=10,
        bounds=[
            dict(
                lon_min=[9.20], 
                lon_max=[15.00], 
                lat_min=[32.30], 
                lat_max=[36.72]
            )
        ],
    priority=3,
    ),

    dict(
        name="Ionian Sea 2",
        label="ion2",
        numericLabel=11,
        bounds=[
            dict(
                lon_min=[15.00], 
                lon_max=[21.88], 
                lat_min=[30.00], 
                lat_max=[36.72]
            )
        ],
        priority=3,
    ),

    dict(
        name="Ionian Sea 3",
        label="ion3",
        numericLabel=12,
        bounds=[
            dict(
                lon_min=[15.00,16.14,16.3,16.3], 
                lon_max=[21.88,21.88,21.88,18.4], 
                lat_min=[36.72,38.1,38.7,40.1], 
                lat_max=[38.1,38.7,40.1,40.51]
            )
        ],
        priority=3,
    ),

    dict(
        name="Sicily Channel",
        label="sic",
        numericLabel=17,
        bounds=[
            dict(
                lon_min=[11.15,11.43,11.71,11.99,12.27,12.55,12.82,13.1,13.38,13.66,13.94,14.22],
                lon_max=[11.43,11.71,11.99,12.27,12.55,12.82,13.1,13.38,13.66,13.94,14.22,14.5],
                lat_min=[36.69,36.54,36.38,36.23,36.07,35.91,35.9,36.04,36.19,36.34,36.5,36.65],
                lat_max=[37.01,37.16,37.32,37.47,37.63,37.79,37.8,37.65,37.48,37.31,37.14,36.97]
            )
        ],
        priority=4,
    ),

    dict(
        name="Levantine Sea 1",
        label="lev1",
        numericLabel=13,
        bounds=[
            dict(
                lon_min=[21.88], 
                lon_max=[26.20], 
                lat_min=[30.00], 
                lat_max=[35.30]
            )
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 2",
        label="lev2",
        numericLabel=14,
        bounds=[
            dict(
                lon_min=[26.2,27.78], 
                lon_max=[33,33], 
                lat_min=[33.5,35.3], 
                lat_max=[35.3,38]
            )
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 3",
        label="lev3",
        numericLabel=15,
        bounds=[
            dict(
                lon_min=[26.20], 
                lon_max=[33.00], 
                lat_min=[30.00], 
                lat_max=[33.50]
                )
            ],
        priority=3,
    ),

    dict(
        name="Levantine Sea 4",
        label="lev4",
        numericLabel=16,
        bounds=[
            dict(
                lon_min=[33.00], 
                lon_max=[37.00], 
                lat_min=[31.00], 
                lat_max=[38.00]
            )
        ],
        priority=3,
    ),

    dict(
        name="Levantine Sea",
        label="lev",
        numericLabel=16,
        bounds=[
            dict(
                lon_min=[21.88,27.78], 
                lon_max=[37.00,37.00], 
                lat_min=[30.00,35.30], 
                lat_max=[35.30,38.00]
            )
        ],
        priority=2,
    ),

    dict(
        name="North West Mediterranean",
        label="nwm",
        numericLabel=4,
        bounds=[
            dict(
                lon_min=[-1.00], 
                lon_max=[9.20], 
                lat_min=[39.50], 
                lat_max=[45.00]
            )
        ],
        priority=3,
    ),

    dict(
        name="South West Mediterranean 1",
        label="swm1",
        numericLabel=2,
        bounds=[
            dict(
                lon_min=[-1.00], 
                lon_max=[3.00], 
                lat_min=[35.50], 
                lat_max=[39.50]
            )
        ],
        priority=3,
    ),

    dict(
        name="South West Mediterranean 2",
        label="swm2",
        numericLabel=3,
        bounds=[
            dict(
                lon_min=[3.00], 
                lon_max=[9.20], 
                lat_min=[35.50], 
                lat_max=[39.50]
            )
        ],
        priority=3,
    ),

    dict(
        name="Tyrrhenian Sea 1",
        label="tyr1",
        numericLabel=5,
        bounds=[
            dict(
                lon_min=[9.2,9.2], 
                lon_max=[13,10.4], 
                lat_min=[41.31,43.7], 
                lat_max=[43.7,44.4]
                )
            ],
        priority=3,
    ),

    dict(
        name="Tyrrhenian Sea 2",
        label="tyr2",
        numericLabel=6,
        bounds=[
            dict(
                lon_min=[9.2,9.2,9.2], 
                lon_max=[15.,16.14,16.3], 
                lat_min=[36.72,38.1,38.7], 
                lat_max=[38.1,38.7,41.31]
            )
        ],
        priority=3,
    ),

    dict(
        name="Mediterranean Sea",
        label="med",
        numericLabel=3,
        bounds=[
            dict(
                lon_min=[-6,27,2], 
                lon_max=[27,39,20], 
                lat_min=[29,29,43], 
                lat_max=[43,41.1,46]
            )
        ],
        priority=1,
    ),


    ## BLACK SEA
    dict(
        name="Black Sea",
        label="bs",
        numericLabel=17,
        bounds=[
            dict(
                lon_min=[27.30], 
                lon_max=[42.50], 
                lat_min=[41.10], 
                lat_max=[47.50]
            )
        ],
        priority=1,
    ),


    ## ATLANTIC OCEAN
    dict(
        name="Atlantic Ocean",
        label="AO",
        numericLabel=1,
        bounds=[
            dict(
                lon_min=[-70,-70,-70,-1,-98,-90,-84,-78.3,-82.5],
                lon_max=[25,-6,2,25,-70,-70,-70,-75.5,-79.7],
                lat_min=[-80,29,43,48.5,18,14,9.5,8.2,8.8],
                lat_max=[29,43,48.5,90,48.5,18,14,9.5,9.5]
            )
        ],
        priority=1,
    ),

    dict(
        name="North Atlantic sub-tropical gyre",
        label="NAstg",
        numericLabel=2,
        bounds=[
            dict(
                lon_min=[-70,-70,-70,-98],
                lon_max=[25,-6,2,-70],
                lat_min=[20,29,43,20],
                lat_max=[29,43,48.5,48.5]
            )
        ],
        priority=2,
    ),

    dict(
        name="Equatorial Atlantic current system",
        label="EAcs",
        numericLabel=3,
        bounds=[
            dict(
                lon_min=[-70,-98,-90,-84,-78.3,-82.5],
                lon_max=[25,-70,-70,-70,-75.5,-79.7],
                lat_min=[-20,18,14,9.5,8.2,8.8],
                lat_max=[20,20,18,14,9.5,9.5]
            )
        ],
        priority=2,
    ),

    dict(
        name="Southern Atlantic sub-tropical gyre",
        label="SAstg",
        numericLabel=4,
        bounds=[
            dict(
                lon_min=[-70.00], 
                lon_max=[25.00], 
                lat_min=[-47.00], 
                lat_max=[-20.00]
            )
        ],
        priority=2,
    ),

    dict(
        name="Amazon River basin",
        label="ARb",
        numericLabel=5,
        bounds=[
            dict(
                lon_min=[-70.00], 
                lon_max=[-40.00], 
                lat_min=[-1.00], 
                lat_max=[20.00]
            )
        ],
        priority=3,
    ),

    dict(
        name="Amazon River estuary",
        label="ARest",
        numericLabel=5,
        bounds=[
            dict(
                lon_min=[-50.00], 
                lon_max=[-45.00], 
                lat_min=[-2.00], 
                lat_max=[6.00]
            )
        ],
        priority=3,
    ),

    dict(
        name="Nordic Sea",
        label="NS",
        numericLabel=1,
        bounds=[
            dict(
                lon_min=[-98,-180,25,134],
                lon_max=[25,-98,134,180],
                lat_min=[48.5,65.9,55,65.9],
                lat_max=[90,90,90,90]
            )
        ],
        priority=2,
    ),


    ## INDIAN OCEAN
    dict(
        name="Indian Ocean",
        label="IO",
        numericLabel=2,
        bounds= [
            dict(
                lon_min=[25,25,25,25,25,25],
                lon_max=[147,142,103,100.5,99.8,99],
                lat_min=[-80,-20,-3,5.3,6.8,9.1],
                lat_max=[-20,-3,5.3,6.8,9.1,30]
            )
        ],
        priority=1,
    ),

    dict(
        name="Southern Indian sub-tropical gyre",
        label="SIstg",
        numericLabel=3,
        bounds=[
            dict(
                lon_min=[25], 
                lon_max=[147], 
                lat_min=[-50], 
                lat_max=[-20]
            )
        ],
        priority=2,
    ),

    dict(
        name="Equatorial Indian current system",
        label="EIcs",
        numericLabel=4,
        bounds=[
            dict(
                lon_min=[25,25,25,25,25],
                lon_max=[142,103,100.5,99.8,99],
                lat_min=[-20,-3,5.3,6.8,9.1],
                lat_max=[-3,5.3,6.8,9.1,30]
            )
        ],
        priority=2,
    ),

    ## PACIFIC OCEAN
    dict(
        name="Pacific Ocean",
        label="PO",
        numericLabel=3,
        bounds=[
            dict(
                lon_min=[147,142,103,100.5,99,-180,-180,-80,-180,-180,-180,-180],
                lon_max=[180,147,142,103,100.5,-70,-77.5,-78,-83,-84,-90,-98],
                lat_min=[-80,-20,-3,5.3,7.5,-80,8,8.7,8.7,10,14,18],
                lat_max=[65.9,65.9,65.9,15,15,8,8.7,9.2,10,14,18,65.9]
            )
        ],
        priority=1,
    ),

    dict(
        name="Southern Pacific sub-tropical gyre",
        label="SPstg",
        numericLabel=4,
        bounds=[
            dict(
                lon_min=[147,-180], 
                lon_max=[180,-70], 
                lat_min=[-55,-55], 
                lat_max=[-20,-20]
            )
        ],
        priority=2,
    ),

    dict(
        name="Equatorial Pacific current system",
        label="EPcs",
        numericLabel=5,
        bounds=[
            dict(
                lon_min=[147,142,103,100.5,99,-180,-180,-80,-180,-180,-180,-180],
                lon_max=[180,147,142,103,100.5,-70,-77.5,-78,-83,-84,-90,-98],
                lat_min=[-20,-20,-3,5.3,7.5,-20,8,8.7,8.7,10,14,18],
                lat_max=[20,20,20,15,15,8,8.7,9.2,10,14,18,20]
            )
        ],
        priority=2,
    ),

    dict(
        name="North Pacific sub-tropical gyre",
        label="NPstg",
        numericLabel=6,
        bounds=[
            dict(
                lon_min=[147,142,103,-180], 
                lon_max=[180,147,142,-98], 
                lat_min=[20,20,20,20], 
                lat_max=[45,45,45,45]
            )
        ],
        priority=2,
    ),

    dict(
        name="North Pacific sub-polar gyre",
        label="NPspg",
        numericLabel=7,
        bounds=[
            dict(
                lon_min=[147,142,103,-180], 
                lon_max=[180,147,142,-98], 
                lat_min=[45,45,45,45], 
                lat_max=[65.9,65.9,65.9,65.9]
            )
        ],
        priority=2,
    ),

    ## SOUTHER OCEAN
    dict(
        name="Southern Ocean",
        label="SO",
        numericLabel=5,
        bounds=[
            dict(
                lon_min=[-180.00], 
                lon_max=[180.00], 
                lat_min=[-80.00], 
                lat_max=[-47.00]
            )
        ],
        priority=2,
    ),
]
