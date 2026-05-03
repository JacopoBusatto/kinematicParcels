import zarr, numpy as np, pandas as pd

z = zarr.open(r'C:/Users/Jacopo/Documents/DATI/PATAGONIA/simulation/output_PFall.zarr', mode='r')
t = z['time'][:]
t0 = pd.Timestamp('2026-01-01 00:00')
outputdt_s = 7200  # 2h

off_grid_traj = []
for i in range(t.shape[0]):
    row = t[i, :]
    valid = ~np.isnan(row)
    if not valid.any():
        continue
    t_vals = row[valid]
    remnants = t_vals % outputdt_s
    off_mask = remnants != 0
    if off_mask.any():
        first_dt = t0 + pd.to_timedelta(t_vals[0], 's')
        last_dt = t0 + pd.to_timedelta(t_vals[-1], 's')
        off_obs_idx = np.where(off_mask)[0]   # index within valid obs
        off_times = [str(t0 + pd.to_timedelta(t_vals[k], 's')) for k in off_obs_idx]
        is_first = bool(off_obs_idx[0] == 0)
        is_last = bool(off_obs_idx[-1] == len(t_vals) - 1)
        off_grid_traj.append({
            'traj': i,
            'n_obs': int(valid.sum()),
            'first': str(first_dt),
            'last': str(last_dt),
            'off_at_start': is_first,
            'off_at_end': is_last,
            'off_times': off_times,
        })

print(f'Trajectories with off-grid timestamps: {len(off_grid_traj)}')
for r in off_grid_traj[:20]:
    tag = []
    if r['off_at_start']:
        tag.append('START')
    if r['off_at_end']:
        tag.append('END')
    print(f"  traj={r['traj']:5d}  n_obs={r['n_obs']:3d}  first={r['first']}  last={r['last']}  "
          f"off@{'+'.join(tag) or 'MID'}  times={r['off_times']}")
