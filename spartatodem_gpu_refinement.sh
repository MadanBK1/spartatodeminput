#!/usr/bin/env python3
"""
SPARTA DSMC → 3D Point Cloud (GPU, User-Biased, Pressure + |V|)
---------------------------------------------------------------
- Builds a fresh user-defined biased permanent 3D point cloud at each run
- Includes pressure, density, Mach, viscosity, and Knudsen number
- Preserves consistent coordinates across all timesteps within a run
- Fully GPU-accelerated using CuPy with multi-GPU parallelism
- Generates Rocky 2025R2-compliant .txt files (with [unit] tags)
"""

import os
import glob
import math
import time
import numpy as np
from collections import deque
from multiprocessing import get_context

# ===============================================================
# ENVIRONMENT (limit threading for HPC)
# ===============================================================
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "CUPY_ASYNC_ALLOCATOR": "cuda",
    "CUPY_ACCELERATORS": "cub",
    "CUPY_FAST_MATH": "1",
    "CUPY_DEBUG": "0",
})

# ===============================================================
# USER CONFIGURATION
# ===============================================================
input_dir  = "/mnt/home/k0006390/research/Impingement20m/grid"
output_dir = "/mnt/research/WMUCFDLAB/2Daxisto3D/20m"
os.makedirs(output_dir, exist_ok=True)

start_step = 320000
end_step   = 320000
step_size  = 10000

# Point cloud geometry (newly built every run)
num_points    = 20_000_000
rng_seed_base = int(time.time())  # unique seed each run
x_range       = (-30.0, 20.0)
y_range       = (-50.0, 50.0)
z_range       = (-50.0, 50.0)
max_radius    = 50.0

dense_regions = [
    {"x": (0, 15), "r": (0, 15), "fraction": 0.25},   # nozzle
    {"x": (15, 20), "r": (0, 50), "fraction": 0.45},  # impingement
]
# Remaining 30% = background

# Physics constants
GAMMA, R_SPEC = 1.4, 287.0
H2O_MU_REF, H2O_T_REF, H2O_OMEGA = 9.0e-6, 300.0, 0.75
BIN_NX = 512
BIN_NR = 512

# ===============================================================
# GPU DETECTION
# ===============================================================
try:
    import cupy as cp
    N_GPU = cp.cuda.runtime.getDeviceCount()
    print(f"🟢 CuPy {cp.__version__} detected ({N_GPU} GPU(s)).")
except Exception as e:
    cp = None
    N_GPU = 0
    print(f"⚠️ CuPy unavailable, using CPU fallback. Reason: {e}")

# ===============================================================
# PHYSICS HELPERS
# ===============================================================
def mu_h2o_vhs(T):
    T = np.asarray(T, float)
    invalid = np.sum(T <= 0)
    if invalid > 0:
        print(f"⚠️ {invalid} invalid T values detected; clamped to 1e−6 K")
    T = np.maximum(T, 1e-6)
    return H2O_MU_REF * (T / H2O_T_REF)**H2O_OMEGA

# ===============================================================
# LOAD SPARTA GRID
# ===============================================================
def load_sparta(file_path):
    x, r, vx, vr, rho, p, M, T, Kn = [], [], [], [], [], [], [], [], []
    reading = False
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ITEM: CELLS"):
                reading = True
                continue
            if not reading or line.startswith("ITEM"):
                continue
            t = line.strip().split()
            if len(t) < 13:
                continue
            try:
                xx, rr = float(t[1]), float(t[2])
                vx_, vr_ = float(t[5]), float(t[6])
                p_, T_, rho_, Kn_ = float(t[7]), float(t[8]), float(t[9]), float(t[12])
                a = math.sqrt(GAMMA * R_SPEC * max(T_, 1e-12))
                M_ = math.sqrt(vx_**2 + vr_**2) / a
                x.append(xx); r.append(rr)
                vx.append(vx_); vr.append(vr_)
                rho.append(rho_); p.append(p_); M.append(M_)
                T.append(T_); Kn.append(Kn_)
            except:
                continue
    return map(np.array, (x, r, vx, vr, rho, p, M, T, Kn))

# ===============================================================
# BUILD USER-BIASED CLOUD (rebuild each run)
# ===============================================================
def build_fixed_random_cloud():
    print(f"🌐 Building NEW user-biased 3D cloud ({num_points:,} points)...")
    rng = np.random.default_rng(rng_seed_base)
    total_alloc = 0
    Xs, Ys, Zs = [], [], []

    # ---------- Dense user-defined regions ----------
    for reg in dense_regions:
        x_rng, r_rng, frac = reg["x"], reg["r"], reg["fraction"]
        npts = int(num_points * frac)
        total_alloc += npts
        print(f"   Dense region x={x_rng}, r={r_rng}, points={npts:,}")

        X = rng.uniform(x_rng[0], x_rng[1], npts)
        U = rng.random(npts)
        R = r_rng[1] * np.sqrt(U)
        theta = rng.uniform(-math.pi, math.pi, npts)
        Y = R * np.cos(theta)
        Z = R * np.sin(theta)
        Xs.append(X); Ys.append(Y); Zs.append(Z)

    # ---------- Background region ----------
    remaining = num_points - total_alloc
    if remaining > 0:
        X = rng.uniform(x_range[0], x_range[1], remaining)
        U = rng.random(remaining)
        R = max_radius * np.sqrt(U)
        theta = rng.uniform(-math.pi, math.pi, remaining)
        Y = R * np.cos(theta)
        Z = R * np.sin(theta)
        Xs.append(X); Ys.append(Y); Zs.append(Z)
        print(f"   Background domain: {remaining:,} points")

    # ---------- Combine all points ----------
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    Z = np.concatenate(Zs)

    # ---------- Apply domain bounds ----------
    mask = ((X >= x_range[0]) & (X <= x_range[1]) &
            (Y >= y_range[0]) & (Y <= y_range[1]) &
            (Z >= z_range[0]) & (Z <= z_range[1]))
    X, Y, Z = X[mask], Y[mask], Z[mask]
    pts = np.column_stack((X, Y, Z)).astype(np.float32)

    print(f"✅ Built cloud: {pts.shape[0]:,} pts")
    print(f"📏 X-range: {pts[:,0].min():.2f} → {pts[:,0].max():.2f} m")
    return pts

# ===============================================================
# BUILD BIN MAP (x,r)
# ===============================================================
def build_bin_map(x_cells, r_cells, nx=BIN_NX, nr=BIN_NR):
    xmin, xmax = x_cells.min(), x_cells.max()
    rmin, rmax = r_cells.min(), r_cells.max()
    if xmax <= xmin: xmax = xmin + 1e-9
    if rmax <= rmin: rmax = rmin + 1e-9
    bin_to_cell = np.full((nx, nr), -1, dtype=np.int32)
    sx = (x_cells - xmin)/(xmax - xmin)
    sr = (r_cells - rmin)/(rmax - rmin)
    ix = np.clip((sx*nx).astype(int), 0, nx-1)
    ir = np.clip((sr*nr).astype(int), 0, nr-1)
    for i in range(x_cells.size):
        bin_to_cell[ix[i], ir[i]] = i
    q = deque()
    for i in range(nx):
        for j in range(nr):
            if bin_to_cell[i,j] != -1: q.append((i,j))
    while q:
        i,j = q.popleft()
        cid = bin_to_cell[i,j]
        for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < nr and bin_to_cell[ni,nj] == -1:
                bin_to_cell[ni,nj] = cid
                q.append((ni,nj))
    return bin_to_cell, xmin, xmax, rmin, rmax

# ===============================================================
# GPU WORKER
# ===============================================================
def gpu_worker(args):
    (gid, pts_chunk, bin_to_cell, xmin, xmax, rmin, rmax, nx, nr,
     vx, vr, rho, p, M, mu, Kn, r_cells) = args
    import cupy as cp
    cp.cuda.Device(gid).use()
    ptsg = cp.asarray(pts_chunk, dtype=cp.float32)
    Xg, Yg, Zg = ptsg[:,0], ptsg[:,1], ptsg[:,2]
    Rg = cp.sqrt(Yg**2 + Zg**2)
    sx = (Xg - xmin)/(xmax - xmin)
    sr = (Rg - rmin)/(rmax - rmin)
    ix = cp.clip((sx*nx).astype(cp.int32),0,nx-1)
    ir = cp.clip((sr*nr).astype(cp.int32),0,nr-1)
    bin_map_g = cp.asarray(bin_to_cell, dtype=cp.int32)
    cid = bin_map_g[ix, ir]
    vxg = cp.asarray(vx)[cid]
    vrg = cp.asarray(vr)[cid]
    rhog = cp.asarray(rho)[cid]
    pg   = cp.asarray(p)[cid]
    Mg   = cp.asarray(M)[cid]
    mug  = cp.asarray(mu)[cid]
    Kng  = cp.asarray(Kn)[cid]
    eps = cp.float32(1e-12)
    invr = 1.0/cp.maximum(Rg, eps)
    cos_th = cp.where(Rg>eps, Yg*invr, cp.float32(1.0))
    sin_th = cp.where(Rg>eps, Zg*invr, cp.float32(0.0))
    vyg = vrg*cos_th
    vzg = vrg*sin_th
    velmag = cp.sqrt(vxg**2 + vyg**2 + vzg**2)
    data = cp.column_stack((Xg,Yg,Zg,vxg,vyg,vzg,velmag,rhog,pg,Mg,mug,Kng))
    return cp.asnumpy(data)

# ===============================================================
# PROCESS TIMESTEP
# ===============================================================
def process_timestep(fp, out_index, pts, pool=None):
    t0 = time.time()
    step = int(os.path.basename(fp).split(".")[1])
    out_path = os.path.join(output_dir, f"20m_cloud{out_index}.txt")
    print(f"\n📂 Processing {fp} (step {step}) → {out_path}")
    x, r, vx, vr, rho, p, M, T, Kn = load_sparta(fp)
    mu = mu_h2o_vhs(T)
    bin_to_cell, xmin, xmax, rmin, rmax = build_bin_map(x, r)
    N = pts.shape[0]
    if N_GPU > 0 and pool is not None:
        base, rem = divmod(N, N_GPU)
        args_list, start = [], 0
        for g in range(N_GPU):
            end = start + base + (1 if g < rem else 0)
            args_list.append((g, pts[start:end], bin_to_cell,
                              xmin, xmax, rmin, rmax, BIN_NX, BIN_NR,
                              vx, vr, rho, p, M, mu, Kn, r))
            start = end
        results = pool.map(gpu_worker, args_list)
        all_data = np.vstack(results)
    else:
        all_data = np.empty((0,12))

    # ---------- Rocky 2025R2-compliant header with units ----------
    header = ("x\ty\tz\tx-velocity\ty-velocity\tz-velocity\tVelocity-Magnitude\t"
              "density\tpressure\tMach\tviscosity\tKn\n")
    with open(out_path, "w") as fout:
        fout.write(header)
        np.savetxt(fout, all_data, fmt="%.2g", delimiter="\t")
    dt = time.time() - t0
    print(f"✅ Saved {out_path} ({all_data.shape[0]:,} pts) in {dt:.1f}s")
    return step, out_path, all_data.shape[0]

# ===============================================================
# MAIN
# ===============================================================
def main():
    pts = build_fixed_random_cloud()  # always build new each run
    def stepnum(fp):
        try: return int(os.path.basename(fp).split(".")[1])
        except: return -1
    files = sorted(glob.glob(os.path.join(input_dir,"velocity.*.grid")), key=stepnum)
    sel = [fp for fp in files if start_step <= stepnum(fp) <= end_step
           and (stepnum(fp)-start_step)%step_size==0]
    print("📌 Selected timesteps:")
    for i,fp in enumerate(sel): print(f"   {fp} → 20m_cloud{i}.txt")
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=N_GPU) if N_GPU>0 else None
    results=[]
    try:
        for idx,fp in enumerate(sel):
            results.append(process_timestep(fp, idx, pts, pool))
    finally:
        if pool: pool.close(); pool.join()
    print("\n📖 Summary:")
    for s,o,n in results:
        print(f"   Step {s} → {os.path.basename(o)} ({n:,} pts)")

if __name__ == "__main__":
    main()
