spartatodeminput.py is a high-performance multi-GPU data transformation script designed to convert 2D axisymmetric SPARTA DSMC simulation outputs (velocity.*.grid files) into 3D volumetric point clouds for visualization and analysis.

The code reconstructs the 3D flow field by rotating the 2D axisymmetric data around the x-axis, sampling millions of pseudo-particles according to the physical volume of each computational cell. Each point carries interpolated quantities such as velocity components, density, Mach number, dynamic viscosity, and Knudsen number.

The implementation uses CuPy, a GPU-accelerated drop-in replacement for NumPy, enabling massive vectorized computations directly on NVIDIA GPUs. The workload is automatically split across all available GPUs on the HPC node using Python’s multiprocessing with per-device isolation, ensuring near-linear scaling.

To maximize throughput:

Each GPU handles a spatial chunk of the grid independently.

Random azimuthal angles (θ) are generated in parallel for 3D reconstruction.

Domain clipping and sampling are fully vectorized on GPU memory.

The output is merged and equalized to a uniform total number of points.

This script supports both single-timestep and multi-timestep processing, automatically switching modes based on user input. It is optimized for modern multi-GPU nodes (such as NVIDIA H100/H200 systems) and can process tens of millions of points per timestep with full reproducibility and minimal CPU overhead
