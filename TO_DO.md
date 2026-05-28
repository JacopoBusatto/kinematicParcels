distanza da costa ci sono degli NC in giro
cinematico che si spegne ad una certa distanza


**ARGO conversion** speak with GLF per capire cosa sono i dati
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

**Drifter conversion**
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



**Couple building**
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


**Transition probability matrix**
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
