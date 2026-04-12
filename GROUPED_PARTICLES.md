# Grouped-Particle Simulations

## Overview

The **grouped-particle mode** extends kinematicParcels to release and simulate multiple particles initialized from the same release center. This is useful for:

- **Pair dispersion studies**: Track relative separation between two particles (Lagrangian diffusivity)
- **Multi-particle ensembles**: Study uncertainty/spread from initial condition perturbations  
- **Kinematic interactions**: Future support for inter-particle forces or model coupling

---

## Conceptual Model

### Single Mode (Default)
Each release center generates **one particle**:
```
Release center (lon, lat)
         │
         └─→ Particle 1
```

### Grouped Mode
Each release center generates **n particles**:
```
Release center (lon, lat)
         │
         ├─→ Particle 1 (at center)
         ├─→ Particle 2 (on circle, random angle)
         ├─→ Particle 3 (on circle, equally spaced)
         └─→ ...
```

**Geometry:**
- **Member 1**: Always at the release center `(lon_center, lat_center)`
- **Members 2..n**: Placed on circle of radius `r` around center
  - **Angular spacing**: $360° / (n-1)$ degrees
  - **Placement**: `random` (random starting angle) or `equal_angles` (deterministic)

**Examples:**
- `group_size=2`: 1 center + 1 on circle (360° separation = single random angle)
- `group_size=3`: 1 center + 2 on circle (180° apart, diametrically opposite)
- `group_size=4`: 1 center + 3 on circle (120° apart)

---

## Using Grouped Particles in Simulation

### YAML Configuration

Add the `group` section under `release` in your simulation config:

```yaml
release:
  mode: region_grid                # or point_list (existing modes)
  region_label: NPstg              # for region_grid
  dlon: 5.0                        # for region_grid
  dlat: 5.0                        # for region_grid
  
  # Or for point_list:
  points:
    - {lon: 12.0, lat: 36.0}
    - {lon: 14.0, lat: 37.0}
  
  filter_domain: true              # existing behavior
  
  # NEW: Grouped-particle expansion (optional)
  group:
    size: 2                        # Number of particles per group
    radius_km: 0.1                 # Radius of circle for non-center members (km)
    placement: random              # or equal_angles (deterministic)
  
  # Existing: depth handling
  depth:
    enabled: false
    values: [0]
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group.size` | int | 1 | Particles per group. 1 = single mode (no expansion) |
| `group.radius_km` | float | 0.1 | Circle radius in kilometers |
| `group.placement` | str | random | `random` = random starting angle per group; `equal_angles` = deterministic |

### Backward Compatibility

- If `group` section is **not present**, defaults to `group.size: 1` (single mode)
- Existing configs work unchanged
- No modifications to command-line interface

### Domain Filtering

**Important:** If **any particle** in a group falls outside the FieldSet domain, the **entire group is discarded**.

This ensures physical consistency: paired particles stay together.

---

## Simulation Output & Metadata

### ParticleSet Variables

Each particle receives three custom metadata variables:

| Variable | Type | Description |
|----------|------|-------------|
| `group_id` | int32 | Base release center index (0, 1, 2, ...) |
| `group_member` | int32 | Position within group (1=center, 2=first circle, ..., n=last circle) |
| `group_size` | int32 | Total particles in group (constant for all members) |

### Zarr Output

[**Note:** Custom metadata fields are passed to Parcels but availability in Zarr output should be verified with your Parcels version. See _Verification_ below.]

---

## Postprocessing: Visualizing Grouped Trajectories

### Trajectory Plotting with Group Filtering

Add `max_group_member` to the `trajectories` section:

```yaml
trajectories:
  plot: true
  title: Grouped Trajectories
  show_start: true
  show_end: true
  
  # NEW: Filter by group member
  max_group_member: null           # null = plot all members
                                   # 1 = only centers
                                   # 2 = centers + first circle members
                                   # n = members 1 through n

  animate: false
  animation_fps: 6
  animation_color_by: lat0
```

### Behavior

| `max_group_member` | Result |
|--------------------|--------|
| `null` (default) | Plot all group members |
| `1` | Plot only central trajectories (group_member==1) |
| `2` | Plot centers and first circle members |
| `n` | Plot members 1 through n (or all if dataset has fewer members) |

### Coloring

**When group_member column is present:**
- Each unique group member gets a **distinct color**
- Same color across all groups (e.g., all "member 2" particles are blue)
- Uses `tab10` colormap (≤10 members) or `hsv` (>10 members)

**When no group_member column:**
- Falls back to monochrome plotting
- Fully backward compatible

### Example Config

```yaml
analysis:
  types:
    - trajectories

trajectories:
  plot: true
  animate: true
  title: "Pair Dispersion Trajectories"
  show_start: true
  show_end: true
  max_group_member: 2              # Show pairs (center + circle)
  
  animation_fps: 6
  animation_color_by: lat0
  animation_label: "initial latitude"
  show_time_bar: true
  trail: true
  trail_steps: 20
```

---

## Architecture & Design Decisions

### Core Components

**1. Group Expansion (`src/kinematicparcels/utilities/group_expansion.py`)**

- `expand_groups()`: Transforms base release centers into grouped particles
  - Generates circle points using Cartesian approximation (suitable for radii < 100 km)
  - Applies group-level domain filtering
  - Returns flattened arrays + metadata
- `generate_circle_points()`: Geometric helper for circle generation
- `km_to_degrees()`: Distance unit conversion (1° ≈ 111.32 km)

**2. Release Pipeline Extension (`src/kinematicparcels/runner/run_experiment.py`)**

Hook point: **After domain filtering, before depth expansion**

Pipeline:
```
Base release generation (region_grid / point_list)
         ↓
Domain filtering (existing)
         ↓
[NEW] Group expansion (if group.size > 1)
         ↓
Depth expansion (existing, applied uniformly to all group members)
         ↓
ParticleSet creation (with group_id, group_member, group_size Variables)
```

**3. Custom Particle Classes**

Two new subclasses in `run_experiment.py`:
- `ScipyParticleGrouped(ScipyParticle)`: With group metadata Variables
- `JITParticleGrouped(JITParticle)`: With group metadata Variables

Parcels requires Variables to be defined at class definition time. Both custom classes are returned by `get_particle_class()`.

**4. Trajectory Visualization (`src/kinematicparcels/postprocessing/plotting/trajectories.py`, `animations/trajectories.py`)**

- `plot_trajectories_map()`: Static trajectory plot with group filtering & coloring
- `animate_trajectories()`: Animated trajectories with group filtering
- Both support `max_group_member` parameter for flexible visualization

### Key Design Decisions

✅ **Separation of concerns**: Group expansion is orthogonal to base release modes
- `release.mode` (region_grid, point_list) controls base point generation
- `group.size` controls particle multiplication
- Can mix any combination

✅ **Reuse existing infrastructure**: 
- Same fieldset builder, domain filtering, depth handling, integration kernel
- Metadata is optional and backward compatible
- No changes to simulation physics

✅ **Spherical geometry approximation**:
- Uses Cartesian approximation on sphere (suitable for < 100 km)
- Accounts for latitude effect on longitude scaling
- Future: Can upgrade to geodetic formulas for larger radii

✅ **Group-level domain filtering**:
- Ensures paired particles stay together spatially
- Discards entire group if any member is outside domain
- Reported with statistics (N groups filtered)

---

## Examples

### Example 1: Pair Dispersion (2-Particle Groups)

```yaml
# simulation config
release:
  mode: region_grid
  region_label: NPstg
  dlon: 5.0
  dlat: 5.0
  filter_domain: true
  group:
    size: 2
    radius_km: 2.0              # 2 km separation
    placement: random            # Different angle per group
  depth:
    enabled: false

simulation:
  particle_type: scipy
  runtime_days: 30
  dt_hours: 1
  outputdt_hours: 6

output:
  zarr_name: pair_dispersion.zarr
```

```yaml
# postprocessing config
analysis:
  types:
    - trajectories

trajectories:
  plot: true
  animate: true
  max_group_member: 2           # Show both members
  animation_color_by: lat0
```

### Example 2: Ensemble (4-Particle Groups)

```yaml
# simulation config
release:
  mode: point_list
  points:
    - {lon: 10.0, lat: 40.0}
    - {lon: 12.0, lat: 38.0}
  filter_domain: true
  group:
    size: 4                      # 1 center + 3 on circle
    radius_km: 0.5               # 500 meters
    placement: equal_angles      # Deterministic spacing
  depth:
    enabled: true
    values: [0, 50]

simulation:
  particle_type: scipy
  runtime_days: 10
  dt_hours: 0.5
  outputdt_hours: 3

output:
  zarr_name: ensemble_test.zarr
```

### Example 3: Single Mode with Center Only (Postprocessing)

```yaml
# Same simulation as above, but visualize only centers:
trajectories:
  plot: true
  max_group_member: 1            # Show only central particles
  title: "Ensemble Centers"
```

---

## Verification & Known Limitations

### Zarr Metadata Output

[**TO BE VERIFIED**]: Custom particle fields (`group_id`, `group_member`, `group_size`) are passed to `ParticleSet.from_list()` but their preservation in Zarr output depends on:
- Parcels version
- Zarr/NetCDF conversion settings
- Custom output handlers

**Current status**: Metadata flows into Parcels but Zarr inclusion requires verification.

### Spherical Geometry

- Cartesian approximation assumes small radii (< 100 km recommended)
- Accuracy degrades at higher latitudes or larger radii
- Future: Upgrade to Vincenty or Haversine formulas for better accuracy

### Future Extensions

1. **Variable group sizes**: Different groups could have different numbers of members
2. **Kinematic interactions**: Add inter-particle forces (e.g., spring-damper model)  
3. **Adaptive placement**: Generate members based on local gradient or flow
4. **Statistical ensembles**: Systematic perturbations from ensemble data assimilation

---

## References & Related Literature

**Pair dispersion (Lagrangian diffusivity):**
- Richardson (1926), Batchelor (1952), Salazar & Collins (2009)
- Key diagnostic: relative velocity variance, separation rate

**Ensemble particle methods:**
- van Leeuwen (2009), Slivinski & Spiller (2016)
- Applications: uncertainty quantification, data assimilation

---

## Troubleshooting

**Q: All groups filtered out - why?**
A: Check if initial points are just outside domain boundary. Use `filter_domain: false` temporarily to diagnose. Domain filtering is conservative; expand your domain if needed.

**Q: Coloring not working in trajectories plot**  
A: Verify `group_member` column exists in trajectory table. Check if postprocessing is reading the same Zarr file. Consider regenerating if Zarr format changed.

**Q: Wrong number of particles in output**  
A: Confirm `group.size` in config. If `size=2` and `points=2`, expect 4 particles (2 groups × 2 members). Domain filtering may reduce this if members are outside bounds.

**Q: Geodetic accuracy needed for large radii**  
A: Current Cartesian approximation is valid < 100 km. For larger radii, upgrade to `generate_circle_points_geodetic()` using Vincenty or Haversine formulas.
