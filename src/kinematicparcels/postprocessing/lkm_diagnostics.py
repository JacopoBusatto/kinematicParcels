"""
LKM diagnostics and analysis tools.

Provides functions to analyze LKM contributions, separation statistics,
and validation of the implementation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional


def load_trajectory_data(zarr_path: str | Path) -> xr.Dataset:
    """
    Load trajectory data from Zarr output.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the Zarr output file

    Returns
    -------
    xr.Dataset
        Trajectory dataset
    """
    return xr.open_zarr(str(zarr_path))


def compute_group_separations(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute pairwise separations within groups.

    Parameters
    ----------
    ds : xr.Dataset
        Trajectory dataset with group_id, lon, lat

    Returns
    -------
    xr.DataArray
        Separations in meters, dimensions (time, group_id, pair)
    """
    # Group by group_id and time
    grouped = ds.groupby('group_id')

    separations = []
    for group_id, group_data in grouped:
        # Get all particles in this group
        lons = group_data.lon.values
        lats = group_data.lat.values
        times = group_data.time.values

        # Compute pairwise distances (simplified, assumes small separations)
        # In practice, would need proper geodesic distance
        n_particles = len(lons)
        if n_particles < 2:
            continue

        # Simple Euclidean distance approximation
        R = 6371000  # Earth radius in meters
        lon_rad = np.radians(lons)
        lat_rad = np.radians(lats)

        # Convert to Cartesian for distance calculation
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)

        # Compute all pairwise distances
        distances = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                distances.append(dist)

        separations.append(np.array(distances))

    return xr.DataArray(
        np.array(separations),
        dims=['group', 'pair'],
        coords={'group': list(grouped.groups.keys())}
    )


def analyze_lkm_contribution(ds: xr.Dataset) -> dict:
    """
    Analyze LKM velocity contributions.

    Parameters
    ----------
    ds : xr.Dataset
        Trajectory dataset with u_lkm, v_lkm variables

    Returns
    -------
    dict
        Statistics about LKM contributions
    """
    if 'u_lkm' not in ds or 'v_lkm' not in ds:
        return {'error': 'No LKM variables found in dataset'}

    u_lkm = ds.u_lkm.values
    v_lkm = ds.v_lkm.values

    # Compute LKM speed
    lkm_speed = np.sqrt(u_lkm**2 + v_lkm**2)

    stats = {
        'lkm_speed_mean': float(np.mean(lkm_speed)),
        'lkm_speed_std': float(np.std(lkm_speed)),
        'lkm_speed_max': float(np.max(lkm_speed)),
        'lkm_u_mean': float(np.mean(u_lkm)),
        'lkm_v_mean': float(np.mean(v_lkm)),
        'lkm_u_std': float(np.std(u_lkm)),
        'lkm_v_std': float(np.std(v_lkm)),
    }

    return stats


def compare_lkm_vs_no_lkm(ds_lkm: xr.Dataset, ds_no_lkm: xr.Dataset) -> dict:
    """
    Compare trajectories with and without LKM.

    Parameters
    ----------
    ds_lkm, ds_no_lkm : xr.Dataset
        Trajectory datasets with and without LKM

    Returns
    -------
    dict
        Comparison statistics
    """
    # This would require careful alignment of initial conditions
    # For now, just return basic stats
    stats_lkm = analyze_lkm_contribution(ds_lkm)
    stats_no_lkm = analyze_lkm_contribution(ds_no_lkm)

    return {
        'lkm_stats': stats_lkm,
        'no_lkm_stats': stats_no_lkm,
        'difference': {
            'speed_mean_diff': stats_lkm.get('lkm_speed_mean', 0) - stats_no_lkm.get('lkm_speed_mean', 0)
        }
    }


def plot_lkm_diagnostics(ds: xr.Dataset, output_dir: Path) -> None:
    """
    Generate diagnostic plots for LKM analysis.

    Parameters
    ----------
    ds : xr.Dataset
        Trajectory dataset
    output_dir : Path
        Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt

        # LKM speed distribution
        if 'u_lkm' in ds and 'v_lkm' in ds:
            lkm_speed = np.sqrt(ds.u_lkm.values**2 + ds.v_lkm.values**2)

            plt.figure(figsize=(10, 6))
            plt.hist(lkm_speed.flatten(), bins=50, alpha=0.7)
            plt.xlabel('LKM Speed (m/s)')
            plt.ylabel('Frequency')
            plt.title('Distribution of LKM Velocity Magnitudes')
            plt.savefig(output_dir / 'lkm_speed_distribution.png')
            plt.close()

        # Group center stability
        if 'center_lon' in ds and 'center_lat' in ds:
            # Plot center trajectories for a few groups
            unique_groups = np.unique(ds.group_id.values)
            n_groups_to_plot = min(5, len(unique_groups))

            plt.figure(figsize=(12, 8))
            for i, group_id in enumerate(unique_groups[:n_groups_to_plot]):
                mask = ds.group_id == group_id
                centers_lon = ds.center_lon.values[mask]
                centers_lat = ds.center_lat.values[mask]

                plt.subplot(2, 3, i+1)
                plt.plot(centers_lon, centers_lat, 'o-', markersize=2)
                plt.title(f'Group {group_id} Center')
                plt.xlabel('Longitude (°)')
                plt.ylabel('Latitude (°)')

            plt.tight_layout()
            plt.savefig(output_dir / 'group_centers.png')
            plt.close()

        print(f"Diagnostic plots saved to {output_dir}")

    except ImportError:
        print("Matplotlib not available, skipping plots")


def run_lkm_analysis(zarr_path: str | Path, output_dir: Optional[str | Path] = None) -> dict:
    """
    Run complete LKM analysis on a trajectory dataset.

    Parameters
    ----------
    zarr_path : str or Path
        Path to Zarr trajectory file
    output_dir : str or Path, optional
        Directory to save diagnostic outputs

    Returns
    -------
    dict
        Analysis results
    """
    ds = load_trajectory_data(zarr_path)

    results = {
        'dataset_info': {
            'n_trajectories': len(ds.trajectory),
            'n_times': len(ds.time),
            'variables': list(ds.data_vars),
        }
    }

    # Analyze LKM contributions
    if 'u_lkm' in ds and 'v_lkm' in ds:
        results['lkm_stats'] = analyze_lkm_contribution(ds)
    else:
        results['lkm_stats'] = {'error': 'No LKM variables in dataset'}

    # Generate plots if output directory provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plot_lkm_diagnostics(ds, output_path)

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Analyze LKM trajectory data")
    parser.add_argument("zarr_file", help="Path to Zarr trajectory file")
    parser.add_argument("--output-dir", help="Directory to save diagnostic plots")

    args = parser.parse_args()

    results = run_lkm_analysis(args.zarr_file, args.output_dir)
    print("Analysis results:")
    for key, value in results.items():
        print(f"  {key}: {value}")