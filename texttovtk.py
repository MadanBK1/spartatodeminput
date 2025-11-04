#!/usr/bin/env python3
"""
============================================================
 TXT → VTK Converter for DSMC 3D Point Clouds
   (Extended for viscosity + Kn fields)
============================================================

Author: Madan B. K. (WMU CFDLAB)
Date:   2025-10-01

Description:
------------
This script converts `.txt` files produced by the
SPARTA-to-3D converter into `.vtk` PolyData files for
visualization (e.g., ParaView) or DEM coupling.

Key Features:
-------------
1. Parallel VTK Writer:
   - Splits large `.txt` files into chunks.
   - Each chunk is processed in parallel using multiprocessing.
   - Each worker builds a `.vtk` part file.

2. Data Fields:
   - Cartesian coordinates (x, y, z)
   - Velocity vector (vx, vy, vz)
   - Scalar density
   - Scalar Mach number
   - Scalar viscosity
   - Scalar Knudsen number

3. Robust Parsing:
   - Skips header lines (with column names or domain bounds).
   - Ignores malformed/non-numeric rows gracefully.

4. Merging:
   - All part `.vtk` files are merged into a single final `.vtk`.
   - Intermediate files are deleted in parallel.

Inputs:
-------
- `.txt` file with 10 numeric columns:
    x, y, z, vx, vy, vz, density, Mach, viscosity, Kn

Outputs:
--------
- Final `.vtk` file containing all points and arrays.

Usage:
------
$ python3 txttovtk_parallel.py

============================================================
"""

import os
import vtk
import multiprocessing
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor

# === CONFIG ===
input_file = "/mnt/research/WMUCFDLAB/2Daxisto3D/30m/30m0.txt"
output_dir = "/mnt/research/WMUCFDLAB/2Daxisto3D/30m"
final_output = os.path.join(output_dir, "30m0.vtk")
num_procs = min(110, multiprocessing.cpu_count())

os.makedirs(output_dir, exist_ok=True)

# === Efficiently split lines ===
def split_lines(filepath, n_chunks):
    with open(filepath, 'r') as f:
        # skip first line (domain info) and second line (header)
        lines = f.read().splitlines()[2:]
    total = len(lines)
    chunk_size = (total + n_chunks - 1) // n_chunks
    chunks = [(lines[i * chunk_size : (i + 1) * chunk_size], i)
              for i in range(n_chunks) if i * chunk_size < total]
    return chunks

# === Worker: build and write one VTK chunk ===
def write_chunk(args):
    lines, idx = args
    points = vtk.vtkPoints()
    velocity = vtk.vtkFloatArray(); velocity.SetNumberOfComponents(3); velocity.SetName("velocity")
    density = vtk.vtkFloatArray(); density.SetNumberOfComponents(1); density.SetName("density")
    mach = vtk.vtkFloatArray(); mach.SetNumberOfComponents(1); mach.SetName("Mach")
    viscosity = vtk.vtkFloatArray(); viscosity.SetNumberOfComponents(1); viscosity.SetName("viscosity")
    kn = vtk.vtkFloatArray(); kn.SetNumberOfComponents(1); kn.SetName("Kn")
    verts = vtk.vtkCellArray()

    for line in lines:
        # Skip empty lines, comments, or headers
        if not line.strip() or line.startswith("#") or line[0].isalpha():
            continue
        try:
            # Expect 10 numeric columns
            x, y, z, u, v, w, rho, M, mu, Kn = map(float, line.split())
        except ValueError:
            continue  # skip malformed rows

        pt_id = points.InsertNextPoint(x, y, z)
        velocity.InsertNextTuple((u, v, w))
        density.InsertNextValue(rho)
        mach.InsertNextValue(M)
        viscosity.InsertNextValue(mu)
        kn.InsertNextValue(Kn)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pt_id)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(verts)
    polydata.GetPointData().SetVectors(velocity)
    polydata.GetPointData().AddArray(density)
    polydata.GetPointData().AddArray(mach)
    polydata.GetPointData().AddArray(viscosity)
    polydata.GetPointData().AddArray(kn)

    out_file = os.path.join(output_dir, f"part_{idx:03d}.vtk")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_file)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()
    print(f"[PID {os.getpid()}] Wrote: {out_file}")
    return out_file

# === Merge helper ===
def merge_vtk_files(file_list, output_filename):
    appender = vtk.vtkAppendPolyData()
    for f in sorted(file_list):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(f)
        reader.Update()
        appender.AddInputData(reader.GetOutput())
    appender.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(appender.GetOutput())
    writer.SetFileTypeToBinary()
    writer.Write()

# === Parallel file deletion ===
def parallel_delete(file_list, max_workers=32):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(os.remove, file_list)

# === MAIN ===
if __name__ == '__main__':
    print(f"Detected {multiprocessing.cpu_count()} CPUs, using {num_procs} processes...")
    task_count = num_procs * 4
    chunks = split_lines(input_file, task_count)
    print(f"Processing {len(chunks)} chunks...")

    ctx = get_context("fork")
    with ctx.Pool(processes=num_procs) as pool:
        out_files = pool.map(write_chunk, chunks)

    print("Merging chunks into final VTK...")
    merge_vtk_files(out_files, final_output)
    print(f"✅ Merged VTK written to: {final_output}")

    print("Deleting intermediate files in parallel...")
    parallel_delete(out_files)
    print("✅ All chunk files removed.")
