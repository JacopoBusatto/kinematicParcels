## MERIDIONAL EXCURSION
I want to build another postprocessing module.
this one has to take each trajectory, and its initial position `lat_0` and `lon_0`
then it takes the northest position and the southest position along the trajectory. We then store
the minimum latitude, the corrisponding longitude, age and time; the maximum latitude and the corresponding age, time and longitude.
The calculated variables could be

- `lat_min`
- `lon_where_lat_min (we can come up with a better name)`
- `age_where_lat_min`
- `time_where_lat_min`
- `lat_max`
- `lon_where_lat_max`
- `age_where_lat_max`
- `time_where_lat_max`
- `lat_0`
- `lon_0`
- `time_0`
- `lat_0_grid`
- `lon_0_grid`
- `lat_min_grid`
- `lon_where_lat_min_grid`
- `lat_max_grid`
- `southward_excursion` that is `lat_0 - lat_min` so it's positive
- `northward_excursion` that isc`lat_max - lat_0`

The gridded variables are obtained from the "grid" section of the postprocessing yaml parameters: we build the grid and assign the value to the coordinates that falls inside.

Optionally we could have a filter on the minimum time length of the trajectory, to avoid including in the analysis short trajectories.

We could call this module `meridional_excursion`
the output can be a netcdf with the trajectory value, or a parquet file. what do you suggest? 
From there it's easy to plot
- a map from the gridded values, with the multiple values on the same pixel 
- a scatter map from the exact coordinates

In the yaml we could have a structure like
```yaml
meridional_excursion:
  min_segment_length_days: null
  plotting:
    merge: mean | min | max # how to deal with overlapping point in a pixel, works only if type has gridded option
    type: # which type of plot to draw, can be both
      - scatter
      - gridded (or another label that clearly describe the pixel-based plot)
    NAME_OF_THE_VARIABLE:
      over: 
        - initial_position
        - southmost_point
        - northmost_point
      vmin: null
      vmax: null
```

What do you think? Ask any question before implementing.
let's first build a robust planning.