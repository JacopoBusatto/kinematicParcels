## **ARGO conversion** speak with GLF per capire cosa sono i dati
Now we have to build some tools.
we have to create in src/kinematicparcels/tool/ a new utility toolkit.
The first tool is the one that converts ARGO data (csv format) in a zarr format identical to the one produced by parcels as output, as if it was a single trajectory simulation. In this way, that all the postprocessing scripts can be used for the argo trajectory data.

For example here: C:/Users/Jacopo/OneDrive - CNR/ARGO/ACC there are many csv files.
this is the header and the first rows of a file:
PLATFORM_CODE,DATE (YYYY-MM-DDTHH:MI:SSZ),DATE_QC,LATITUDE (degree_north),LONGITUDE (degree_east),POSITION_QC,PRES (decibar),PRES_QC,PSAL (psu),PSAL_QC,TEMP (degree_Celsius),TEMP_QC,PRES_ADJUSTED (decibar),PRES_ADJUSTED_QC,TEMP_ADJUSTED (degree_Celsius),TEMP_ADJUSTED_QC,PSAL_ADJUSTED (psu),PSAL_ADJUSTED_QC
1900042,2002-11-30T04:23:36Z,1,-45.966,51.947,1,5.5,1,33.743,1,5.955,1,2.2,1,5.955,1,33.745,1
1900042,2002-11-30T04:23:36Z,1,-45.966,51.947,1,9.4,1,33.746,1,5.931,1,6.1,1,5.931,1,33.748,1
1900042,2002-11-30T04:23:36Z,1,-45.966,51.947,1,19.0,1,33.746,1,5.92,1,15.7,1,5.92,1,33.748,1

we need to open each of the files and retrieve
minimum: time, latitude, longitude, platform_code, parking_depth
optional: other variables passed by the user

As I look through the data it seems that latitude, longitude, date are considered constant from the surface and the next submerged phases, hence it retrieves the date and its position at the surface and then it maintains these values when it descends.

the code has to
1) Obtain the parking depth. It looks that doing an istogram  of the pressure is not a solution since during ascending, parking, descending to 2000 and then profiling upward the time resolution varies. Maybe it is safe to assume 1000 m as a general parking depth.

2) detect the surface point in each transmittin phase and create a trajectory from those points. This can be done for example simply grouping by lat, lon and date and consider the first points (cheching that the grouping is possible)

2) calculate the segment of the same trajectory: if the time resolution of the trajectory becomes larger than 10 days (the maximum time resolution allowed for argo floats), it has to be considered athat it escaped the selected region and the trajectory has to be split in different segments. 
The user can choose to: 
a. select the longest segment
b. ignore the splitting and consider an irregular trajectory
c. consider different segments as different floats: the platform_code in this case becomes a float number and each segment has its own decimal number (the integer part is the platform_code) with a minimal time length 

3) if the user specifies one, or a list of geographical regions, it selects the trajectories that pass by that region at least once. The user can also decide to cut the trajectory from its first point in one of the specified regions onward or to keep the whole trajectory.

4) resample and interpolate (grade of interpolation chosen by user) each segment in a certain time frequency, chosen by the user. the user can also choose to skip this step.

5) move to the next file of the total list of files

6) create a single zarr file similar to the outputs of parcels simulations (example: C:/Users/Jacopo/Documents/GitHub/kinematicParcels/outputs/sicily/ex08_circle_uniform_2d_sicily.zarr but we won't need any grouping strategy here since they are considered single trajectories at the moment)

We need to decide if all the parameters needed can be passed by parse arguments or its better to create a yml parameters file

Before writing the code, we need to address all your doubts, check the workflow; then create tests that confirm the right development of our methodology. Try to use what already exists, if anything is compatible when possible.

-------
Sicuramente il problema è nell'interpolazione: quando scavallano la linea di cambiamento di data si creano salti non fisici: ecco quelli che abbiamo visto prima.
modifichiamo il codice di interpolazione, che capisca quando stiamo interpolando a cavallo della linea di cambiamento di data e a qual punto gestisca la cosa, per esempio rispostandosi in 0-360, interpolando e poi tornare a -180 180
Solo dopo questo andiamo a togliere i salti non fisici.

## **Drifter conversion**
Next tool on the way: we have to create in src/kinematicparcels/tool/ a new utility toolkit.
This tool converts Drifter data data (csv format) in a zarr format identical to the one produced by parcels as output, with the same idea behind the argo conversion tool, as if it was a single trajectory simulation. In this way, that all the postprocessing scripts can be used for the drifters trajectory data.

For example here: `C:\Users\Jacopo\OneDrive - CNR\DRIFTERS\csv\SO_2000-2025.csv` there is the csv file of all the trajectory in the southern ocean from 20000101 to 22506XX.
this is the header and the first rows:
ID,time,latitude,longitude,drogue_lost_date,DrogueLength
,UTC,degrees_north,degrees_east,UTC,
103798,2012-04-23T18:00:00Z,44.66,-9.975,2011-10-26T00:00:00Z,5.2 m
103798,2012-04-24T00:00:00Z,44.626,-9.939,2011-10-26T00:00:00Z,5.2 m

The code has to
1) clip the trajectories after the drogue lost days, if any
2) select the desiderd minimum drogue length (an optional parameter)
3) calculate the segment of the same trajectory: if the time resolution of the trajectory becomes larger than 6 hours (time resolution), or however it becomes irregular, it has to be considered as it escaped the queried (from the source download) region and the trajectory has to be split in different segments. 
The user can choose to: 
a. select the longest segment
b. ignore the splitting and consider an irregular trajectory
c. consider different segments as different platforms. the trajectory sequential identification are different, but they keep the source ID (we can call it platform_code to maintain consistence with the ARGO conversion)
4) if the user specifies one (or a list of) geographical regions, it selects the trajectories that pass by that region at least once. The user can also decide to cut the trajectory from its first point in one of the specified regions onward or to keep the whole trajectory.
5) time resample and interpolation strategy is the same as argo conversion.
6) create a single zarr file similar to the outputs of parcels simulations (same as argo conversion)

We can have a similar yml input file

Before writing the code, we need to address all your doubts, check the workflow; then create tests that confirm the right development of our methodology. Try to use what already exists, if anything is compatible when possible.



## **Couple building**
I want to create the next tool, based on the legacy coupleDrifter.py script.
This new version starts from a zarr file, like the one generated from the argo data. The user must know that it is required that the input zarr must have the time variable of the trajectories in a shared sequencing, as we do in the aro conversion script, such that we can look for syncronus trajectories.

The general idea is
- we loop over the trajectories
- we consider all the other trajectories that have a shared time window.
- pair by pair, we take the shared window, if any
- we calculate the spatial distance between the two trajectory (here is necessary the shared time grid)
- if the minimum distance is smaller than a threshold, we take the two trajectory from the time of that distance onward
- that pair becomes a new group entity with two members.
- we calculate the needed variables to fill the output format of a zarr output of a grouped particle output
- we store that pair
- we move to the next pair candidate
- we proceed this with all the possible pairs in the input file (without counting the pairs twice: i.e. if the trajectory n was paired with the trajectory m, when it is m turn we do not need to check what hapens with the trajectory n since it was already checked)

the legacy script should do the same thing, with different i/o formats.
Let's first check the logic of my idea, then the logic of the legacy code and let's see if they match.
Once every uncertainties are solved we can proceed with the implementation, thinking if we can make the new script more efficient.


## **Transition probability matrix**
I want to build a new post processing module. It works like this
We take N geographical regions; I will refere to a specific region with an index, i.
We need to calculate the transition probability from a region i to the region j.
At each time we will determine in which region the trajectories are. We already have the localiztion functions that utilizes the `src\kinematicparcels\utilities\geographicalRegions.py` module. It is important that the regions in consideration share the same priority level, to avoid spacial overlap. (RAISE WARNING)
Then we will look where each trajectory started.
We can count at each time step how many trajectories are in each region and calculate the matrix:
$P_{i,j}^t = \frac{n_{i,j}^t }{n_i^0} = \frac{n_{i,j}^t }{n_{i,i}^0}$
with $n_{i,j}^t$ being the number of particles that started in the region $i$ that are in the region $j$ at the time $t$ and $n_i^0$ being the number of particles that started in the region $i$, that corresponds exactly to $n_{i,i}^0$.

The module takes as parameters:
- the N regions to be considered
- the time frequency (integer of time-step of the input source), default 1

It returns
- a csv file with NxN columns and a number of lines equals to the total number of time step divided by the time frequency

We can also add a filter on the transition, removing isolated symbolic points:
- if a symbolic value is isolated AND the previous and the next point are equal, we substitute this isolated point with the neighbour value.

Let's first discuss the workflow and the coding strategy, assessing all the uncertainties you might have.
Only then we can produce to the coding. Try to minimize the writing, reusing the existing modules and following the postprocessing structure flow. Check the documentation and the current state. At the end of the coding phase, we also need to create 
- the example yml that describes this module's options
- the documentation `POSTPROCESSING.md` updated with the new module
- the test scripts that need to run smoothly
------------
We need to slightly modify the logic of this module: 
- we don't use the absolute time axis, but the "age" of the particle, namely `time - time[0]`, so that unsyncronized starts are counted correctly. I would give the number in days unit.
- We need a minimum life length for the trajectories: the particles that live less thant that treshold are ignored.
- We could need a maximum time length as well. since we are counting a probability, it is necessary to have a constant number of trajectory. So we need a maximum "age", above which the trajectories are trimmed.

If my reasoning is correct, these case should be accounted:
- putting 0 on the min time would mean that every trajectory is considered, whatever its duration is; then maybe trimmed with max
- putting min and max the same, we would have segments of trajectories exactly of that value of length

Let's clarify all your doubts before writing any code

## **TRANSIZIONI**
max: null, basta normalizzare su quanti ne ho

## **FRONTI**
minima latitudine ad ogni longitudine cosi localizzo barriere
max escursione su posizione iniziale

## **FSLE and FTLE map**
We need to create the next postprocessing tool. FSLE (finite-scale lyapunov exponent) and FTLE (finite time lyapunov exponent) map. The legacy scripts are separate but maybe we can merge them into a single one.
The idea is to have the output mapped on a grid originated by the gridded regional release, working within a group entity environment.
Initializing initially spatially close groups it is possible to assess the geographical properties of relative dispersion
In foreward simulations they map areas of strong and weak horizontal divergence.
In backward simulations, vice versa, they measure convergence areas. 
Foreward simulation:
For every starting location we calculate the FSLE by calculating how much time the members of the same group take to separate of a certain distance (or a list of distances, imposed in yml) from the central one. If the group has more than 2 member we take the minimum time. If the group fails to separate at least of that distance, then the value on the central point will be 0 or NaN, depending on a yml parameter (for example mask_0_fsle or a suitable name). Then we calculate the lyapunov exponent and we assign it to the initial position of the group.
The FTLE is similarely calculated, but we give a fixed time (or a list of times, imposed in yml) from the release date, then we calculate the distance (or the maximum distance reached in that time window? what would you do? what is done in the legacy script? maybe is more correct to take the poits precisely after that window, or the closest previous one) from the central trajectory of each group member and we take the maximum of those. Then we calculate the lyapunov exponent and we assign it to the initial position of the group.
If it is a simulation obtained with a continuous release, we will have a lat-lon map for every time of release,performing the calculations on each release.

Similarely, in backward simulations, the same rules apply but the starting time is actually the time-wise end (so it should be the first obs and maximum time, i.e. the time when the particles are initialized).

The script needs then to assess if the simulation is a backward or a foreward one.

The distances can be both geodedical (haversine) or meridional at the moment
I would have a yaml structure like
```yaml
exponents_maps:
  distance: geodedical # geodedical - meridional
  fsle:
    enable: true
    scale: [] #float or list of floats, here in km
    mask_zeros: false #if zeros of exponent i.e. groups do not separate of that scale in their life have a 0 value (if false) or nan (if true)
    plot:
      enable: true
      average_on_time: true # performs the average along the release time dimension before plotting, otherwise plots one map per release time
      vmin: null
      vmax: null
      min_mask_value: null #set a min threshold of exponent value to mask value smaller than that
      log_scale: false # the colorbar is logscale?
      cmap: viridis # the colormap label
  ftle:
    enable: true
    scale: [] #float or list of floats, here in days
    plot:
      enable: true
      average_on_time: true # performs the average along the release time dimension before plotting, otherwise plots one map per release time
      vmin: null
      vmax: null
      min_mask_value: null #set a min threshold of exponent value to mask value smaller than that
      log_scale: false # the colorbar is logscale?
      cmap: viridis # the colormap label
```
Any modification or add-up is welcome.

The output will be
- netcdf for fsle with the distance type distinction and the scales included, one for the fsle and one for the ftle
- plots, if any

Let's first plan the workflow, checking the agreement with this plan and the legacy scripts.
Ask me any uncertainties you might have and any ambiguity you might rise.
keep the code modular and in line with the already existing postprocessing models. Don't write code yet. 



## **drf drifters converter**
This is the header of one of the files:
```
*2021/12/06 11:39:36.28
*IOS HEADER VERSION 2.0      2016/04/28 2016/06/13 MATLAB

*FILE
    START TIME          : UTC 2015/07/27 18:21:36.000
    END TIME            : UTC 2015/08/15 08:34:48.000
    TIME INCREMENT      : 0 0 5 0 0  ! (day hr min sec ms)
    TIME UNITS          : Minutes
    NUMBER OF RECORDS   : 2214
    DATA DESCRIPTION    : Drifting Buoy
    NUMBER OF CHANNELS  : 6

    $TABLE: CHANNELS
    ! No Name            Units        Minimum      Maximum   
    !--- --------------- ------------ ------------ ----------
       1 Record_Number   n/a          1            2214      
       2 Date            YYYY/MM/DD   n/a          n/a       
       3 Time            HH:MM:SS     n/a          n/a       
       4 Latitude        degrees      53.27857     53.58016  
       5 Longitude       degrees      -129.13622   -128.88232
       6 Flag:At_Sea     n/a          1            3         
    $END

    $TABLE: CHANNEL DETAIL
    ! No  Pad        Start  Width  Format      Type  Decimal_Places
    !---  ---------  -----  -----  ----------  ----  --------------
       1  ' '        ' '        8  F           R4      1           
       2  ' '        ' '    ' '    YYYY/MM/DD  D     ' '           
       3  ' '        ' '    ' '    HH:MM:SS    T     ' '           
       4  999.00000  ' '       10  F           R4      5           
       5  999.00000  ' '       11  F           R4      5           
       6  9          ' '        2  D           I       0           
    $END
    $REMARKS
        At_Sea flags have the following significance:
        -------------------------------------------------------------------------
        0 = Not classified.
        1 = Good - at sea, freely floating (valid).
        2 = Bad - at sea but trapped in rocky intertidal (floating but not free).
        3 = Bad - on land (grounded, test data, etc.).
        4 = Bad - at sea (large GPS error, on ship, etc.).
        5 = Bad - land travel.
        -------------------------------------------------------------------------
    $END

*ADMINISTRATION
    MISSION             : 2015-046
    AGENCY              : IOS, Ocean Sciences Division, Sidney, B.C.
    COUNTRY             : Canada
    PROJECT             : WCVI - WCDC Moorings
    SCIENTIST           : Johannessen S.
    PLATFORM            : John P. Tully
    $REMARKS
        For information about the IOS drifter program see https://www.waterproperties.ca/drifters
    $END

*LOCATION
    GEOGRAPHIC AREA     : B.C. Coast and Inlets
    LATITUDE            :  53  16.71420 N  ! (deg min)
    LONGITUDE           : 129   8.17320 W  ! (deg min)
    LATITUDE 2          :  53  34.80960 N  ! (deg min)
    LONGITUDE 2         : 128  52.93920 W  ! (deg min)

*RECOVERY
    TIME FOUND          : UTC 0000/01/00
    LATITUDE FOUND      :   0   0.00000 N  ! (deg min)
    LONGITUDE FOUND     :   0   0.00000 E  ! (deg min)

*INSTRUMENT
    TYPE                : Oceanetic Measurement
    MODEL               : Surface Circulation Tracker
    ID                  : 277
    DEVICE ID           : 2544364
    $REMARKS
        http://www.oceanetic.com.
        SCT buoy hull:
          Height 50 cm.
          Beam 25 cm.
          Draft 33 cm.
          Dry Weight 1.1 kg.
          Saturated Weight ~3.0 kg.
          Freeboard ~0.3 cm.
          Ratio of Cross-section Area ~1 : 15 (above water : below water).
          Materials cellulose sponge, cork, aluminum, zinc, steel.
          Deployment by hand from ship.
        SPOT Trace transmitter:
          https://www.findmespot.com.
          Waterproofing IPX7, 1 m for up to 30 minutes, if USB port is
            sealed with manufacturer supplied cover.
          Temperature range -30 C to +60 C.
          Dimensions L= 8.7 cm x W= 5.1 cm x H= 2.1 cm.
          Weight 0.1 kg (negatively buoyant).
          Batteries 4 x standard L92 batteries.
          Tracking interval 5, 10, 30, 60 minutes (option for 2.5 minutes).
          Operating lifetime 10-11 days typical with 5 minute rate,
            under ideal conditions.
          Position accuracy 7.8 m with 95% confidence level.
        See also:
          Hourston, R.A.S., Martens, P.S., Juhasz, T., Page, S.J. and Blanken, H. 2021.
            Surface ocean circulation tracking drifter data from the Northeastern Pacific and
            Western Arctic Oceans, 2014-2020. Can. Data Rep. Hydrogr. Ocean Sci. 215: vi + 36 p.
            https://waves-vagues.dfo-mpo.gc.ca/Library/40986500.pdf
          Page, S.J., Hannah, C., Juhasz, T., Spear, D., and Blanken, H. 2019.
            Surface circulation tracking drifter data for the Kitimat Fjord system
            in northern British Columbia and adjacent continental shelf for April,
            2014 to July, 2016. Can. Data. Report. Hydrog. Ocean.Sci. 328: vi + 33 p.
            https://waves-vagues.dfo-mpo.gc.ca/Library/40789676.pdf
    $END

*HISTORY

    $TABLE: PROGRAMS
    !   Name       Vers  Date       Time     Recs In   Recs Out
    !   ---------- ----  ---------- -------- --------- ---------
        DRIFTS2IOS 1.0   2021/12/06 11:39:36      2214      2214
    $END
    $REMARKS
        -DRIFTS2IOS processing: 2021/12/06 11:39:36
         All atSea=4,5 locations were removed.
    $END

*COMMENTS
    Data processed using drifteval software provided by R.Pawlowicz, see:
      Pawlowicz, R., Hannah, C. and Rosenberger, A., 2019. Lagrangian
      observations of estuarine residence times, dispersion, and
      trapping in the Salish Sea. Estuarine, Coastal and Shelf Science,
      225, p.106246.
      http://www.sciencedirect.com/science/article/pii/S0272771419302719

*END OF HEADER
     1.0 2015/07/27 18:21:36  53.53889 -129.01738 1
     2.0 2015/07/27 18:25:56  53.53818 -129.01799 1
     3.0 2015/07/27 18:30:58  53.53726 -129.01865 1
     4.0 2015/07/27 18:35:56  53.53650 -129.01944 1
     5.0 2015/07/27 18:40:57  53.53611 -129.02071 1
     6.0 2015/07/27 18:45:56  53.53586 -129.02255 1
     7.0 2015/07/27 18:50:57  53.53545 -129.02449 1
```

Following what happens in the drifter_to_zarr.py script, we have to convert all of the trajectories in a certain folder into a zarr file
the files are stored here:
ls "C:\Users\Jacopo\OneDrive - CNR\BC_DATA\DRIFTERS\gribbell_island_drifters_IOS\drf"

```
  Directory: C:\Users\Jacopo\OneDrive - CNR\BC_DATA\DRIFTERS\gribbell_island_drifters_IOS\drf
```

```bash
Mode                 LastWriteTime         Length Name                                                                                                                     
----                 -------------         ------ ----                                                                                                                     
-a---l        05/06/2026     13:25         477860 codedavis534060166740_20200803_20201002.drf                                                                              
-a---l        05/06/2026     13:25           6808 sct0008_20140415_20140425.drf                                                                                            
-a---l        05/06/2026     13:25          62447 sct0175_20150310_20150323.drf                                                                                            
-a---l        05/06/2026     13:25          78877 sct0176_20150310_20150405.drf                                                                                            
-a---l        05/06/2026     13:25          48401 sct0178_20150310_20150316.drf                                                                                            
-a---l        05/06/2026     13:25          63242 sct0179_20150310_20150323.drf                                                                                            
-a---l        05/06/2026     13:25          64514 sct0184_20150311_20150325.drf                                                                                            
-a---l        05/06/2026     13:25         123113 sct0277_20150727_20150815.drf                                                                                            
-a---l        05/06/2026     13:25         117018 sct0278_20150727_20150902.drf                                                                                            
-a---l        05/06/2026     13:25          61103 sct0279_20150727_20150830.drf                                                                                            
-a---l        05/06/2026     13:25          38895 sct0280_20150727_20150807.drf                                                                                            
-a---l        05/06/2026     13:25         156379 sct0327_20151020_20151104.drf                                                                                            
-a---l        05/06/2026     13:25          74123 sct0328_20151020_20151129.drf                                                                                            
-a---l        05/06/2026     13:25          66597 sct0330_20151020_20151116.drf                                                                                            
-a---l        05/06/2026     13:25          53982 sct0331_20151020_20151024.drf                                                                                            
-a---l        05/06/2026     13:25          31613 sct0511_20160707_20160713.drf                                                                                            
-a---l        05/06/2026     13:25          22625 sct1124_20200803_20200820.drf                                                                                            
-a---l        05/06/2026     13:25          85007 sct1128_20200803_20200820.drf                                                                                            
-a---l        05/06/2026     13:25          56227 sct1136_20200803_20200917.drf                                                                                            
-a---l        05/06/2026     13:25          15470 sct1143_20200803_20200804.drf                                                                                            
-a---l        05/06/2026     13:25          33871 sct1202_20201014_20201023.drf                                                                                            
-a---l        05/06/2026     13:25          19773 sct1212_20201015_20201117.drf                                                                                            
-a---l        05/06/2026     13:25           6804 sct1695_202507271626_202507280151.drf                                                                                    
-a---l        05/06/2026     13:25           8283 sct1695_202508100850_202508110710.drf                                                                                    
-a---l        05/06/2026     13:25           6135 sct1695_202508271853_202508280046.drf                                                                                    
-a---l        05/06/2026     13:25           7075 sct1700_202507271543_202507280310.drf                                                                                    
-a---l        05/06/2026     13:25           6401 sct1700_202507280803_202507281203.drf                                                                                    
-a---l        05/06/2026     13:25           7279 sct1700_202507302200_202507311355.drf     
```

we can maintain the same structure of the yml input parameter file drifter_to_zarr.yml

What do you think it's necessary?
I see that the time resolution here is 5 minutes, so we can have a very high resolution dataset. when converting can we check the different time resolutions?
Let's plan the new workflow and the output structure.




## CLUSTER STRENGTH
I want to add a new postprocessing analysis module to this repository:

Please read the existing postprocessing architecture before editing. In particular inspect:
- src/kinematicparcels/postprocessing/config/models.py
- src/kinematicparcels/postprocessing/config/loader.py
- src/kinematicparcels/postprocessing/core/gridding.py
- src/kinematicparcels/postprocessing/runner/run_postprocessing.py
- src/kinematicparcels/postprocessing/workflows/run_density.py
- src/kinematicparcels/postprocessing/analyses/density.py
- src/kinematicparcels/postprocessing/animations/density.py
- POSTPROCESSING.md

Add a new analysis called `cluster_strength`.

Scientific definition
---------------------
Use the cluster strength definition from Huntley et al. 2015, “Clusters, deformation, and dilation: Diagnostics for material accumulation regions”, stored in the repo reference folder (`references\JGR Oceans - 2015 - Huntley - Clusters  deformation  and dilation  Diagnostics for material accumulation regions.pdf`).

For each target grid point x* and each time t, compute:
```LaTex
$ C(x*, t) = sum_n exp(- (d(x*, x_n(t)) / L)^2 ) $
```
where:
- $x_n(t)$ is the position of particle n at time t
- $d(x)$ is the selected distance metric
- $L$ is the tunable length scale, provided by config as `scale_km`
- the output variable should be named `cluster_strength`

Configuration
-------------
Add a new optional YAML section:

```yaml
cluster_strength:
  scale_km: float              # required, positive
  distance: haversine          # default haversine;
  mask: true                   # if true, compute/output only grid cells explored at least once
  animate: false               # if true, create an animation
  plot_snaps: false            # if true, plot selected snapshots
  timestep_snaps: null         # int or list[int]; required if plot_snaps is true
  vmin: 0
  vmax: null
  cmap: viridis
```
Please use the corrected spelling `cluster_strength` everywhere. Do not introduce the misspelled `cluster_strenght`.

Integration points
------------------
Follow the existing postprocessing pattern.

Add:
- `src/kinematicparcels/postprocessing/analyses/cluster_strength.py`
- `src/kinematicparcels/postprocessing/workflows/run_cluster_strength.py`
- `experiments/configs/examples/postprocessing/10_cluster_strength.yml`, an example postprocessing YAML


Update:
- `src/kinematicparcels/postprocessing/analyses/__init__.py`
- `src/kinematicparcels/postprocessing/config/models.py`
- `src/kinematicparcels/postprocessing/config/loader.py`
- `src/kinematicparcels/postprocessing/runner/run_postprocessing.py`
- `POSTPROCESSING.md`

The workflow should:
1. get the trajectory table using the existing base-products helpers;
2. build/reuse the regular grid using the existing grid section and `build_grid_from_config`;
3. compute cluster strength for every saved timestep;
4. save:
   - `cluster_strength.nc`
   - optionally a table if consistent with existing products
   - optional snapshot PNGs
   - optional GIF animation

Grid and mask behavior
----------------------
Use the regular grid already implemented in `postprocessing/core/gridding.py` that is defined in the `grid` section of the .

If `mask: true`, define valid target grid cells as cells that are visited by at least one particle at least once during the whole trajectory dataset. Compute/store cluster strength only on those cells; all other grid cells should be NaN.

If mask: true, apply the mask only to the target grid cells where cluster strength is evaluated/output. Do not use the mask to pre-filter the particle table. For each valid target grid cell, all particles present at that timestep may contribute if they are within the distance cutoff.

Distance design
---------------
Implement only the default distance now.
`distance: haversine` should use great-circle/haversine distance in km.

However, design the code so future distance backends can be added cleanly. In the future I want to add a fjord/skeleton distance:
- user passes `distance: skeleton_kml`
- user passes `kml_path`
- the KML contains connected segments defining the fjord skeleton
- distance between A and B is:
  distance from A to closest point on skeleton
  + shortest path length along the skeleton
  + distance from closest skeleton point to B

Do not implement this future skeleton/KML metric now, but avoid hard-coding the cluster-strength computation to one distance function.

Efficiency requirements
-----------------------
The naive computation is O(n_time * n_grid * n_particles), which can become too slow.

Please implement the default distance efficiently:
- avoid building a full dense distance matrix for large datasets;
- process one timestep at a time;
- use a finite Gaussian cutoff, default internally to 4 * scale_km unless there is a better local convention;
- preferably use scipy.spatial.cKDTree on local projected coordinates to find candidate particles near each target grid point, then compute the Gaussian only for candidates;
- handle empty timesteps gracefully;
- keep memory bounded by chunking grid points if needed.

The exact haversine distance may be used for final candidate distances after the KDTree neighbor query. If you use a local projection for both neighbor query and distance approximation, document the approximation clearly and keep it appropriate for regional domains.

Outputs
-------
The NetCDF should have dimensions:

time, lat, lon

and variable:

cluster_strength(time, lat, lon)

Include useful attributes:
- scale_km
- distance metric
- formula
- cutoff_factor or cutoff_km if used
- grid metadata consistent with existing gridded outputs

Plotting
--------
For snapshots:
- `plot_snaps: true` requires `timestep_snaps`.
- `timestep_snaps` can be one integer or a list of integers.
- Support negative indices like Python indexing: -1 means the last timestep.
- Save files with clear names such as `cluster_strength_timestep_DATE:TIME.png`.

For animation:
- reuse the style of the existing density animation where practical.
- respect `vmin`, `vmax`, and `cmap`.

Validation and tests
--------------------
Add focused tests if the test structure supports it.

At minimum test:
1. a tiny synthetic dataset where one particle exactly on one grid point gives contribution 1 at that point;
2. two particles give the sum of two Gaussian contributions;
3. `mask: true` leaves never-explored cells as NaN;
4. invalid config: missing or non-positive `scale_km`;
5. snapshot index parsing including negative indices.

Please run the relevant tests or, if the environment cannot run them, report exactly what could not be run and why.

Before implementing, ask me all the questions you might have.


## Rtraj 2 zarr
I want to add a new dedicated conversion tool, not modify the existing CSV converter too much.

Relevant existing files:
- src/kinematicparcels/tools/argo_to_zarr.py
- src/kinematicparcels/tools/zarr_writer.py
- experiments/configs/southern_ocean/ARGO_to_zarr.yml

Please inspect these files first to understand the existing output format and the current resampling / region filtering logic.

Goal:
Create a new script:

    src/kinematicparcels/tools/rtraj_to_zarr.py

This script should convert original Argo trajectory NetCDF files, named like:

    1902267_Rtraj.nc
    4903848_Rtraj.nc

into the same final Zarr format produced by the current ARGO CSV converter and compatible with my Lagrangian code outputs.

Important:
Do not try to preserve CSV compatibility inside this new script. The old CSV converter can stay as it is. This new script is only for original Argo Rtraj NetCDF files.

Input:
The new YAML config should support:
```yaml
    input:
      rtraj_dir: F:/PLATFORMS/ARGO/netcdf/Rtraj/core
      pattern: "*_Rtraj.nc"
```

and optionally:
```yaml
    input:
      rtraj_files:
        - F:/PLATFORMS/ARGO/netcdf/Rtraj/core/1902267_Rtraj.nc
        - F:/PLATFORMS/ARGO/netcdf/Rtraj/core/4903848_Rtraj.nc
```
Output:
Use the existing build_dataset_from_trajectories(...) and build_zarr_encoding(...) functions so that the final dataset is compatible with the current Lagrangian trajectory Zarr outputs.

Required per-observation dataframe columns:
- platform_code
- time
- lat
- lon
- z

Also preserve a trajectory index and obs index in the same way as the existing converter.

Data extraction from Rtraj:
- platform_code: read `PLATFORM_NUMBER` and convert it to integer `WMO`.
- time: prefer `JULD_ADJUSTED` when valid, otherwise use `JULD`.
- lat/lon: use `LATITUDE` and `LONGITUDE`.
- parking pressure: use `REPRESENTATIVE_PARK_PRESSURE` from the `N_CYCLE` group/axis.
- cycle number: prefer `CYCLE_NUMBER_ADJUSTED` when valid, otherwise `CYCLE_NUMBER`. Cycle numbers should only be used internally to map `REPRESENTATIVE_PARK_PRESSURE` from the cycle axis to the measurement rows. Do not include `cycle_number` in the final output dataset unless it is explicitly useful for debugging. Since the trajectories are later resampled onto a different time grid, `cycle_number` should not be interpolated or treated as a physical trajectory variable.
Parking pressure to z:
- `REPRESENTATIVE_PARK_PRESSURE` is pressure in dbar, not exact geometric depth.
- For now use the approximation:

      `z = REPRESENTATIVE_PARK_PRESSURE`

  with positive values, consistent with the existing converter where `z = 1000.0`.
- Add dataset attributes explaining that z is approximated from Argo `REPRESENTATIVE_PARK_PRESSURE` in dbar.
- If later there is a clean local pattern for optional gsw support, you may add it, but do not make gsw a hard dependency.

Mapping parking pressure to observations:
- In Argo trajectory files, `REPRESENTATIVE_PARK_PRESSURE` is per cycle.
- Map it to measurement rows using cycle numbers:
  - Prefer `CYCLE_NUMBER_ADJUSTED` matched against `CYCLE_NUMBER_INDEX_ADJUSTED`.
  - Fall back to `CYCLE_NUMBER` matched against `CYCLE_NUMBER_INDEX`.
- If a row cannot be mapped to a valid representative park pressure:
  - use `processing.parking_depth.fallback_value` if provided;
  - otherwise keep z as NaN.
- Print a summary of missing z values.

Filtering:
Do not implement the old segmentation logic in this new script.
The trajectory is already complete per WMO, so we do not want `split_as_new` / `longest` / `max_gap_days` / `max_speed_km_per_day` logic here.

Do not use the old cut_from_first_entry boolean in this new script. Instead implement an explicit region selection policy.

Keep region filtering, but simplify it:
- Reuse the existing `RegionManager` / `ALL_REGIONS` logic from argo_to_zarr.py if possible.
- Support config:
```yaml
    processing:
      regions:
        names_or_labels:
          - DP
        selection_mode: from_first_entry
        input_lon_mode: "-180_180"
```
Behavior:
- If no regions are provided, keep all trajectories unchanged.
- If regions are provided, apply the selected selection_mode.
- Drop trajectories that do not satisfy the selected region policy.
- Do not cut a trajectory again if it later leaves the region.

Allowed selection_mode values:

1. from_first_entry
   Keep trajectories that enter one of the selected regions at least once.
   For each kept trajectory, cut it from the first point inside the selected region onward.
   Do not cut it again if it later leaves the region.

2. full_if_enters
   Keep trajectories that enter one of the selected regions at least once.
   Keep the full original trajectory.

3. initial_inside
   Keep only trajectories whose first valid point is inside one of the selected regions.
   Keep the full trajectory for those selected floats.
   This corresponds to floats effectively starting/released inside the region of interest.

Resampling:
Keep the same resampling strategy as the existing argo_to_zarr.py.
During resampling, do not linearly interpolate `platform_code` or z. Treat z as a non-interpolated variable and fill it with nearest/forward-backward fill, because parking pressure is a cycle-level property rather than a continuously sampled depth.
If the existing shared resampling code currently interpolates all numeric columns except `platform_code`, update the new script so that z is also excluded from numeric interpolation.

It should support:
- enabled
- frequency
- interpolate
- reference_time
- shared_time
- shift_start_to_reference

Reuse the existing helper functions if reasonable, or move shared code into a small internal helper module if that is cleaner. Do not do a large refactor unless necessary.

Example config:
Create a new config file, for example:

    experiments/configs/southern_ocean/RTRAJ_to_zarr.yml

with:
```yaml
    input:
      rtraj_dir: F:/PLATFORMS/ARGO/netcdf/Rtraj/core
      pattern: "*_Rtraj.nc"

    output:
      path: F:/PLATFORMS/ARGO/zarr/DP_rtraj.zarr

    processing:
      parking_depth:
        mode: representative_park_pressure
        fallback_value: 1000.0

    regions:
      names_or_labels:
        - DP
      selection_mode: from_first_entry
      input_lon_mode: "-180_180"

    resample:
      enabled: true
      frequency: 10d
      interpolate: time
      reference_time: 2000-01-01T00:00:00Z
      shared_time: true
      shift_start_to_reference: false
```

CLI:
The script should be runnable like:

    `python -m kinematicparcels.tools.rtraj_to_zarr experiments/configs/southern_ocean/RTRAJ_to_zarr.yml`

Testing / validation:
- Add focused tests if the repo has a test framework.
- At minimum, make helper functions small and testable.
- Run a minimal import/CLI help check.
- If possible, create a tiny synthetic xarray Dataset mimicking an Rtraj file and test:
  - PLATFORM_NUMBER extraction
  - JULD_ADJUSTED fallback to JULD
  - cycle-to-REPRESENTATIVE_PARK_PRESSURE mapping
  - resampling does not break the final dataframe shape
  - region selection_mode behavior
Implementation style:
- Keep the new script close in style to argo_to_zarr.py.
- Avoid broad refactors.
- Do not modify the old CSV converter unless extracting shared helper functions is clearly useful and low risk.
- Keep error messages explicit.