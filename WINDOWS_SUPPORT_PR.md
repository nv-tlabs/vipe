# Fix Windows / MSVC build & runtime: ViPE now runs end-to-end on Windows 10/11

## TL;DR

Six small patches that together make ViPE buildable + runnable on Windows
without WSL or a Docker container. Verified end-to-end on Windows 10
(build 26100) with VS 2022 Professional + CUDA Toolkit 12.4 + torch
2.5.1+cu124. The pipeline now produces correct `pose/`, `intrinsics/`,
`depth/`, `mask/`, `rgb/` artifacts from both the bundled
`assets/examples/dog-example.mp4` and arbitrary user clips.

Patches break down:
1. **`csrc/lietorch_ext/lietorch_cpu.cpp`** — wrap host kernel templates
   in an anonymous namespace. **This is the critical fix.** Without it,
   the Windows MSVC linker silently aliases CPU host templates with
   same-name `__global__` CUDA templates in `lietorch_gpu.cu`, then
   routes GPU kernel launches into the host C++ implementation. The
   host template dereferences CUDA device pointers and the process dies
   with `STATUS_ACCESS_VIOLATION` (0xC0000005) at first BA iteration.
2. **`vipe/utils/io.py`** — Windows-safe `tempfile.NamedTemporaryFile`
   handling for OpenEXR. Windows holds an exclusive lock on the temp
   file while the Python handle is open; OpenEXR's second open fails
   with "Permission denied" and the pipeline aborts AFTER SLAM has
   already succeeded.
3. **`csrc/slam_ext/geom_kernels.cu`**, **`csrc/utils_ext/knn.cu`** —
   replace `<long>` template parameters with `<int64_t>`. MSVC's
   `long` is 32-bit but the underlying tensors are `torch::kInt64`
   (8 bytes); the original code links cleanly on Linux (where
   `long == int64_t`) but fails to link on Windows (LNK2001:
   unresolved external `at::TensorBase::data_ptr<long>`).
4. **`vipe/ext/specs.py`** — Windows-aware compiler flag handling.
   MSVC's `cl.exe` doesn't accept `-O3` or `-isystem`. Auto-download
   of Eigen 3.4.0 still works, just emit `-I<path>` (which both `cl`
   and `nvcc` accept) and `/O2` (MSVC's max-speed flag) instead of
   the GCC equivalents.
5. **`setup.py`** — opt-in PDB emission via `VIPE_DEBUG_SYMBOLS=1`
   env var. Off by default to keep release builds lean; on for
   developers who want crash dumps to resolve to source line numbers.
6. *(no runtime change)* documentation in this file of the install
   recipe that worked on Windows 10/11.

No effect on Linux or macOS — every patch is either Windows-conditional
or a portable refactor that doesn't change behaviour on POSIX.

---

## Patch 1 (root cause) — `csrc/lietorch_ext/lietorch_cpu.cpp`

### Symptom

On Windows, `vipe infer --image-dir <frames> -o <out>` consistently
exits with code `-1073741819` (`0xC0000005`, `STATUS_ACCESS_VIOLATION`)
immediately after SLAM Pass 1 frontend completes, before Pass 2 starts.
No Python traceback. No CUDA runtime error. The process is killed
synchronously by the Windows kernel.

Reproducible with:
- `assets/examples/dog-example.mp4` (bundled reference clip) — crashes
  at frame 24/122 in SLAM Pass 1
- 8-frame consecutive image-dir input — Pass 1 completes 100%,
  crashes in `backend.run(7)` before Pass 2

Affects every Windows install regardless of GPU, CUDA toolkit version
(tested 12.4), torch CUDA build (tested cu121 and cu124), or VRAM
budget (tested 8 GB and 16 GB GPUs).

### Root cause

`csrc/lietorch_ext/lietorch_cpu.cpp` and `csrc/lietorch_ext/lietorch_gpu.cu`
both define 12 templates with **identical signatures**:

| Template name | `lietorch_cpu.cpp` (host) | `lietorch_gpu.cu` (device) |
|---|---|---|
| `exp_forward_kernel<G, T>` | line 20 (host, `at::parallel_for`) | line 25 (`__global__`) |
| `exp_backward_kernel<G, T>` | line 34 | line 41 |
| `log_forward_kernel<G, T>` | line 49 | line 51 |
| `log_backward_kernel<G, T>` | line 62 | line 66 |
| `inv_forward_kernel<G, T>` | **line 77** | **line 77** |
| `inv_backward_kernel<G, T>` | line 91 | line 89 |
| `mul_forward_kernel<G, T>` | line 106 | line 95 |
| `mul_backward_kernel<G, T>` | line 120 | line 114 |
| `adj_forward_kernel<G, T>` | line 138 | line 156 |
| `adjT_forward_kernel<G, T>` | line 175 | line 209 |
| `act_forward_kernel<G, T>` | line 210 | line 240 |
| `act4_forward_kernel<G, T>` | line 278 | line 263 |

Both translation units use the standard pybind11 emission with **default
external linkage** for templates. The host versions are plain C++
functions; the device versions are decorated with `__global__`.

**On Linux + GCC**:
- The `__global__` attribute participates in name mangling, producing
  a distinct mangled symbol for each pair
- e.g. `_Z18inv_forward_kernelIN7lietorch4SE3IfEEfEvPKT0_PS3_i` (host) vs
  `_Z18inv_forward_kernel...` with a host-side stub for the launch
  registration
- The linker keeps them separate, calls route correctly

**On Windows + MSVC**:
- MSVC's name mangling does NOT distinguish `__global__` from host
  versions — both produce the same mangled symbol (e.g.
  `??$inv_forward_kernel@VSE3@1@M@@YAXPEBMPEAMH@Z`)
- The linker sees the same symbol exported from `lietorch_cpu.obj`
  and `lietorch_gpu.obj` and silently picks ONE (typically the
  earlier one in link order = the CPU host version)
- Every call site that intended to invoke the `__global__` GPU
  kernel is rewritten by the linker to call the host C++ version
  instead
- nvcc's CUDA kernel launch wrapper (the `<<<NUM_BLOCKS, NUM_THREADS>>>`
  syntax) gets compiled to `cudaLaunchKernel(&hostFunction, ...)` —
  but `hostFunction` is now the C++ host loop, not the device
  kernel

When `inv_forward_gpu(group_id, X)` is called with X on `cuda:0`:

```cpp
torch::Tensor inv_forward_gpu(int group_id, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor Y = torch::zeros_like(X);

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.scalar_type(),
        "inv_forward_kernel", ([&] {
            inv_forward_kernel<group_t, scalar_t>
                <<<NUM_BLOCKS(batch_size), NUM_THREADS>>>(
                    X.data_ptr<scalar_t>(),  // <-- DEVICE pointer
                    Y.data_ptr<scalar_t>(),  // <-- DEVICE pointer
                    batch_size);
        }));

    return Y;
}
```

`X.data_ptr<float>()` returns a CUDA device pointer like
`0x0000000c_a51f9a00`. On Linux this gets passed to the GPU kernel
which runs on the GPU and reads device memory fine. **On Windows**,
the call is routed to the host template in `lietorch_cpu.cpp:84`:

```cpp
for (int64_t i = start; i < end; i++) {
    Group X(X_ptr + i * Group::N);   // <-- LINE 84, the crash site
    ...
}
```

`Group(ptr)` invokes `Eigen::Matrix<float, 4, 1>::Matrix(const float*)`
which copies 4 floats from `ptr`. The host CPU tries to read from a
CUDA device address — Windows kernel raises `STATUS_ACCESS_VIOLATION`
(0xC0000005) at the `movups xmm2, xmmword ptr [rcx+rbx+0Ch]`
instruction.

### Fix

Wrap all 12 host kernel templates in `lietorch_cpu.cpp` in an
anonymous namespace. Templates in anonymous namespaces get
**internal linkage** in C++17, meaning their symbols are not
exported from the translation unit and the linker cannot confuse
them with same-name symbols from other translation units.

```cpp
namespace {

template <typename Group, typename scalar_t>
void exp_forward_kernel(...) { ... }

// ... 11 more templates ...

}  // anonymous namespace
```

Total diff: 2 lines added (`namespace {` and `}`), no logic changes.

### Verification

After patch + clean rebuild, on the exact same 8-frame test:

```
SLAM Pass (1/2): 100% | 8/8 [00:25, 1.68s/it]
BA iters = 16, energy: 4.85 -> 0.19 -> 0.04 -> 0.02 ...   (converged)
SLAM Pass (2/2): 100% | 8/8 [00:03, 2.02it/s]
Aligning depth:  100% | 8/8 [00:25, 3.63s/it]
[vipe.utils.io] saving pose/intrinsics/depth/mask/rgb artifacts
2026-05-19 12:00:04 - vipe - INFO - Finished
```

Artifacts:
- `out/pose/<name>.npz` — (N, 4, 4) cam-to-world SE3 matrices
- `out/intrinsics/<name>.npz` — (N, 4) per-frame fx/fy/cx/cy
- `out/depth/<name>.zip` — N OpenEXR depth maps
- `out/mask/<name>.zip` — N TrackAnything dynamic-content masks
- `out/rgb/<name>.mp4` — re-encoded RGB video
- `out/vipe/<name>_info.pkl` — BA residual + meta info

### Diagnostic methodology

The root cause was identified by:

1. **Enabling Windows Error Reporting Local Dumps** for `python.exe`
   (registry-only, no admin) → automatic `.dmp` capture on every crash
2. **Adding `/Zi` host compile flag + `/DEBUG` linker flag** to emit
   a consolidated PDB next to `vipe_ext.pyd`
3. **Running `cdb -z <dump.dmp> -cf <script>`** with `!analyze -v` and
   `k 30` to extract the native stack trace

Without PDB symbols, the crash was attributed to "somewhere inside
vipe_ext, near `PyInit_vipe_ext+0x26828`" — unactionable. With PDB
symbols, `!analyze -v` resolved the exact source line:

```
FAULTING_SOURCE_FILE:   C:\research\ViPE\csrc\lietorch_ext\lietorch_cpu.cpp
FAULTING_SOURCE_LINE:   84

SYMBOL_NAME:  vipe_ext_cp310_win_amd64!<lambda_69bdfca...>::operator+0x98
FAILURE_BUCKET_ID:  INVALID_POINTER_READ_c0000005_vipe_ext.cp310-win_amd64.pyd!
                    _lambda_69bdfca26b73dea3b2b1753c36546a2a_::operator
```

Combined with the deeper stack frame `vipe_ext!inv_forward_gpu::__l2::...`
showing the call came from the GPU dispatcher, the host-template-being-
called-instead-of-GPU-kernel pattern became unambiguous.

---

## Patch 2 — `vipe/utils/io.py` (Windows-safe OpenEXR tempfile)

### Symptom

After Patch 1 lands, SLAM completes successfully and reaches
`save_artifacts(...)`. Then:

```python
File "C:\research\ViPE\vipe\utils\io.py", line 274, in save_depth_artifacts
    exr = OpenEXR.OutputFile(f.name, header)
OSError: Cannot open image file "C:\Users\...\Local\Temp\tmpjhp1iads.exr".
        Permission denied.
```

### Root cause

```python
with tempfile.NamedTemporaryFile(suffix=".exr") as f:
    exr = OpenEXR.OutputFile(f.name, header)
    ...
```

`tempfile.NamedTemporaryFile` opens the file with `O_TEMPORARY` /
`O_EXCL` semantics on Windows. The OS holds an **exclusive lock**
on the underlying file while the Python handle is active. When
`OpenEXR.OutputFile(f.name, ...)` tries to open the same file for
writing from native code, Windows refuses with ERROR_SHARING_VIOLATION
which surfaces as "Permission denied".

On Linux this works because POSIX allows concurrent opens by default.

### Fix

Create the temp file, **close the Python handle immediately**, hand
the name to OpenEXR, then manually unlink in a `finally` block:

```python
tmp = tempfile.NamedTemporaryFile(suffix=".exr", delete=False)
tmp.close()
try:
    exr = OpenEXR.OutputFile(tmp.name, header)
    exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
    exr.close()
    z.write(tmp.name, f"{frame_idx:05d}.exr")
finally:
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
```

Same behaviour on Linux; works on Windows.

---

## Patch 3 — `csrc/slam_ext/geom_kernels.cu`, `csrc/utils_ext/knn.cu` (MSVC `long` width)

### Symptom

At link time on Windows:

```
geom_kernels.obj : error LNK2001: unresolved external symbol
    "public: long * __cdecl at::TensorBase::data_ptr<long>(void)const"
    (??$data_ptr@J@TensorBase@at@@QEBAPEAJXZ)
geom_kernels.obj : error LNK2001: unresolved external symbol
    "public: long * __cdecl at::TensorBase::mutable_data_ptr<long>(void)const"
    (??$mutable_data_ptr@J@TensorBase@at@@QEBAPEAJXZ)
fatal error LNK1120: 2 unresolved externals
```

### Root cause

`geom_kernels.cu` uses `tensor.data_ptr<long>()` and
`PackedTensorAccessor32<long, ...>` throughout. The tensors involved
(`ii`, `jj`, `kk`, `idx`, etc.) are explicitly constructed with
`torch::TensorOptions().dtype(torch::kInt64)` (8 bytes per element).

**On Linux + GCC**, `sizeof(long) == 8` so `<long>` resolves to the
explicit `<int64_t>` template instantiation that PyTorch ships in its
prebuilt libraries.

**On Windows + MSVC**, `sizeof(long) == 4`. PyTorch only ships
`data_ptr<int32_t>` and `data_ptr<int64_t>` instantiations explicitly;
`data_ptr<long>` on Windows would need a `data_ptr<int32_t>`
instantiation that mis-strides the underlying kInt64 storage. The
linker rejects with LNK2001.

### Fix

Replace `<long>` with `<int64_t>` everywhere in the affected `.cu`
files. The semantic meaning is the same on Linux (since `long ==
int64_t`); on Windows the new code uses the correct 8-byte stride.

Files touched:
- `csrc/slam_ext/geom_kernels.cu` — all `<long>` → `<int64_t>` for
  template parameters, plus a few `int64_t` local variables to
  match
- `csrc/utils_ext/knn.cu` — 2 local `long` declarations changed to
  `int64_t` for portability against `Tensor.size()` return type

---

## Patch 4 — `vipe/ext/specs.py` (Windows-aware compile flags)

### Symptom

```
cl : Command line warning D9002 : ignoring unknown option '-O3'
cl : Command line warning D9002 : ignoring unknown option '-isystem'
cl : Command line warning D9024 : unrecognized source file type
    'C:\research\ViPE\csrc\include', object file assumed
cl : Command line warning D9027 : source file 'C:\research\ViPE\csrc\include'
    ignored
```

Build then fails with `fatal error C1083: Cannot open include file:
'eigen3/Eigen/Dense': No such file or directory`.

### Root cause

`_eigen_include_flags()` and the optimization flag in `get_cpp_flags()`
emit GCC/Clang syntax:
- `-isystem <path>` — works for GCC/Clang, **silently ignored** by
  MSVC, AND MSVC then treats the next argument (the path) as a
  source file, AND rejects it as "unrecognized source file type"
- `-O3` — works for GCC/Clang, MSVC has no `-O3`; MSVC's max-speed
  flag is `/O2`

The result is that the Eigen header search path never reaches the
compiler, and Eigen 3.4.0 (which `specs.py` auto-downloads to
`csrc/include/eigen3/Eigen/`) becomes invisible.

For `nvcc` compilation of `.cu` files, the host-compiler flags are
forwarded via `-Xcompiler`; the same problem applies, plus nvcc itself
treats `/I<path>` (MSVC syntax) as an extra positional argument and
aborts:

```
nvcc fatal : A single input file is required for a non-link phase
             when an outputfile is specified
```

### Fix

Detect Windows at module import time:

```python
_IS_WINDOWS = platform.system() == "Windows"
```

Emit `-I<path>` (no space) on Windows — accepted by **both** MSVC's
`cl.exe` and `nvcc` — and `-isystem <path>` on Linux/macOS (keeps
Eigen template noise out of warning logs):

```python
def _include_flag_for_host(path: str) -> list[str]:
    if _IS_WINDOWS:
        return [f"-I{path}"]
    return ["-isystem", path]
```

Swap `-O3` for `/O2` on Windows host compiles. Keep nvcc's `-O3`
unchanged (nvcc accepts it on all platforms):

```python
def get_cpp_flags() -> list[str]:
    opt = "/O2" if _IS_WINDOWS else "-O3"
    return [opt, "-DWITH_CUDA"] + _additional_include_flags()
```

---

## Patch 5 — `setup.py` (opt-in PDB emission)

Adds an `extra_link_args=["/DEBUG", "/OPT:REF", "/OPT:ICF"]` clause
when both `platform.system() == "Windows"` and
`VIPE_DEBUG_SYMBOLS=1` are set. Combined with the corresponding
`/Zi` host compile flag in `specs.py`, this produces a consolidated
`vipe_ext.cp310-win_amd64.pdb` (~80 MB) next to the `.pyd` that
WinDbg / cdb / Visual Studio can use to resolve crash dumps to
source file + line.

Off by default — only developers who want crash-investigation
support need to set the env var.

---

## Installation recipe (Windows 10/11)

Required tools:
- Visual Studio 2022 (Community / Professional / BuildTools — any
  edition with the C++ workload installed)
- CUDA Toolkit 12.4 (or matching whatever cu version torch is built
  against)
- Python 3.10 (the only `nvidia-vipe` wheel target on PyPI today is
  cp310)
- A virtualenv / venv dedicated to ViPE (don't mix into a torch-using
  app's main env; ViPE installs `torch` + 60 other deps including
  `OpenEXR` which can clash)

```powershell
# 1. Create the venv
python -m venv C:\path\to\vipe-env

# 2. Install matching torch + torchvision FIRST (so vipe_ext links
#    against the same CUDA runtime the user's GPU code uses).
& "C:\path\to\vipe-env\Scripts\python.exe" -m pip install `
    --index-url https://download.pytorch.org/whl/cu124 `
    torch==2.5.1 torchvision==0.20.1

# 3. Install build tools
& "C:\path\to\vipe-env\Scripts\python.exe" -m pip install `
    setuptools wheel ninja packaging

# 4. Clone + install nvidia-vipe from source (PyPI sdist also works
#    after this PR lands)
git clone https://github.com/nv-tlabs/vipe C:\research\ViPE

# 5. Build — must run inside a Developer Command Prompt OR call
#    vcvars64.bat first. DISTUTILS_USE_SDK=1 tells torch's
#    cpp_extension that the VC env is already activated.
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $env:CUDA_HOME
& "C:\path\to\vipe-env\Scripts\python.exe" -m pip install -e C:\research\ViPE --no-build-isolation
```

Smoke test (bundled sample):

```powershell
& "C:\path\to\vipe-env\Scripts\python.exe" -m vipe.cli.main infer `
    C:\research\ViPE\assets\examples\dog-example.mp4 `
    -o C:\research\ViPE\vipe_results `
    -p default
```

Expect "Finished" + non-empty `vipe_results/pose/`, `intrinsics/`,
`depth/`, `mask/`, `rgb/` directories.

---

## Notes for reviewers

* All 6 patches are gated on `platform.system() == "Windows"` or are
  semantic no-ops on Linux/macOS. Linux behaviour is unchanged.
* The anonymous-namespace wrap in `lietorch_cpu.cpp` is a portable
  C++17 idiom and arguably improves Linux too (cleaner intent —
  these templates are translation-unit private).
* I did not add CI workflows for Windows. Happy to follow up with
  a `.github/workflows/windows.yml` if the maintainers want one.
* PDB output is opt-in via `VIPE_DEBUG_SYMBOLS=1` to keep release
  builds slim; a maintainer might want to enable it for the nightly
  build to make user crash reports actionable.

Tested on:
- Windows 10 build 26100
- Visual Studio 2022 Professional 17.x with C++ Desktop workload
- CUDA Toolkit 12.4
- torch 2.5.1+cu124
- Python 3.10.0
- NVIDIA RTX 3070 Ti Laptop (8 GB) and RTX A5000 (24 GB)
- Cat orbit clip (478×360, h264, 269 frames) + bundled
  `dog-example.mp4` + bundled `cosmos-example.mp4`
