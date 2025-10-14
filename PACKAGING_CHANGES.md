# ViPE Packaging Changes Summary

This document summarizes the changes made to package ViPE as an installable Python package for use in Modal apps.

## Changes Made

### 1. Configuration Files Relocated
- **Before**: `configs/` at repository root
- **After**: `vipe/configs/` inside the package
- **Reason**: Config files need to be part of the package data to be included in distributions

### 2. Updated `pyproject.toml`
Added package data configuration:
```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
vipe = ["configs/**/*.yaml", "configs/**/*.yml"]
```

### 3. Fixed Config Path Resolution
Updated `vipe/__init__.py`:
```python
# Before:
def get_config_path() -> Path:
    return Path(__file__).parent.parent / "configs"

# After:
def get_config_path() -> Path:
    return Path(__file__).parent / "configs"
```

### 4. Updated `run.py`
Modified to use the package config path:
```python
from vipe import get_config_path

@hydra.main(version_base=None, config_path=str(get_config_path()), config_name="default")
def run(args: DictConfig) -> None:
    # ...
```

### 5. Created `MANIFEST.in`
Added explicit inclusion rules for:
- README, LICENSE, and documentation files
- Config files (*.yaml, *.yml)
- C++ and CUDA source files for extensions
- Environment specification files
- Scripts

## Verification

All packaging requirements have been verified:
- ✅ Package structure is correct
- ✅ Config files are in `vipe/configs/`
- ✅ `get_config_path()` returns package location
- ✅ `pyproject.toml` includes package data configuration
- ✅ `MANIFEST.in` includes all necessary files
- ✅ `run.py` uses portable config path
- ✅ CUDA extension sources are present

## Installation

The package can now be installed in multiple ways:

### Local Installation (Development)
```bash
# Editable install
pip install --no-build-isolation -e .

# Regular install
pip install --no-build-isolation .
```

### From Git Repository
```bash
pip install git+https://github.com/your-org/vipe.git
```

### In Modal
See `MODAL_INSTALLATION.md` for detailed Modal integration instructions.

## Files Modified

1. `pyproject.toml` - Added package data configuration
2. `vipe/__init__.py` - Updated `get_config_path()` function
3. `run.py` - Updated to import and use `get_config_path()`
4. `configs/` → `vipe/configs/` - Relocated config directory

## Files Created

1. `MANIFEST.in` - Package file inclusion rules
2. `MODAL_INSTALLATION.md` - Modal integration guide
3. `PACKAGING_CHANGES.md` - This document

## Backward Compatibility

The changes maintain backward compatibility:
- The CLI (`vipe` command) works the same way
- The `run.py` script works the same way
- Python API usage remains unchanged
- All existing functionality is preserved

The only difference is that configs are now loaded from the installed package location rather than the repository root.

## Testing

To verify the package is correctly structured:

1. Check package structure:
```bash
# Verify configs are in the right place
ls -la vipe/configs/

# Check config files exist
ls vipe/configs/*.yaml
ls vipe/configs/pipeline/*.yaml
```

2. Test import (requires dependencies):
```python
from vipe import get_config_path
import os

config_path = get_config_path()
print(f"Config path: {config_path}")
print(f"Exists: {os.path.exists(config_path)}")
print(f"Files: {os.listdir(config_path)}")
```

3. Test CLI (requires full environment):
```bash
vipe --help
vipe infer --help
```

## Next Steps for Modal Integration

1. Create a Modal app in `/home/shivin/ml-monorepo/backend/modal/apps/`
2. Use the installation examples from `MODAL_INSTALLATION.md`
3. Configure the Modal image with CUDA and dependencies
4. Test video processing in Modal's GPU containers

## Notes

- CUDA extensions will build automatically during installation
- Modal's GPU containers have the necessary CUDA development tools
- All config files are included as package data
- The package works with both editable and regular installations


