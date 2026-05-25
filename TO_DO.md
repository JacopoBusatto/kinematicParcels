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

**Couple building**

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


