# NanoOWL Setup Guide (Desktop Ubuntu + RTX GPU)

> Adapted from [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl) for desktop Ubuntu with pip-installed TensorRT.
> Tested on: Ubuntu 24, NVIDIA RTX 4080, CUDA 12.8 driver, PyTorch cu124.

---

## Prerequisites

- NVIDIA GPU with CUDA support (driver ≥ 525)
- [Miniforge / Mamba](https://github.com/conda-forge/miniforge) installed
- Internet access for downloading model weights (~613MB on first run)

---

## Directory Layout

All repos live under a single folder to keep things tidy:

```
~/Documents/Projects/nanoowl-stuff/
├── torch2trt/      # NVIDIA torch-to-TensorRT converter
└── nanoowl/        # NanoOWL repo
```

---

## Step 1 — Create the Conda Environment

Use Python **3.10.11** specifically. PyTorch 2.6+ has a regression with Python 3.10's
`inspect` module (`posixpath UnboundLocalError`) that is avoided by pinning to 3.10.11
and PyTorch 2.5.1.

```bash
mamba create -n nanoowl python=3.10.11 -y
mamba activate nanoowl
pip install uv
```

---

## Step 2 — Install PyTorch 2.5.1 (pinned)

```bash
uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# Expected: True  2.5.1+cu124
```

---

## Step 3 — Install TensorRT (cu12 explicitly)

Always specify `tensorrt-cu12` — without it, pip auto-selects `cu13` which mismatches
the PyTorch cu124 build.

```bash
uv pip install tensorrt-cu12 \
    --extra-index-url https://pypi.nvidia.com
```

Verify:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Step 4 — Set LD_LIBRARY_PATH (env-scoped, not global)

Use conda's activation hooks so the library path is set **only when the env is active**
and restored on deactivate. No `~/.bashrc` changes.

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

TRT_LIBS=$(find $CONDA_PREFIX -name "libnvinfer.so*" 2>/dev/null | head -1 | xargs dirname)

cat > $CONDA_PREFIX/etc/conda/activate.d/trt_libs.sh << EOF
export _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TRT_LIBS:\$LD_LIBRARY_PATH
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/trt_libs.sh << EOF
export LD_LIBRARY_PATH=\$_OLD_LD_LIBRARY_PATH
unset _OLD_LD_LIBRARY_PATH
EOF

# Reload the env for hooks to take effect
mamba deactivate && mamba activate nanoowl
```

---

## Step 5 — Install Python Dependencies

Install **all** dependencies before building torch2trt or nanoowl.

```bash
uv pip install \
    transformers \
    pillow \
    numpy \
    onnx \
    matplotlib \
    opencv-python \
    aiohttp

# Required for tree_predict.py and tree_demo.py (not on PyPI, install from GitHub)
uv pip install git+https://github.com/openai/CLIP.git
```

---

## Step 6 — Create a Python trtexec Wrapper

NanoOWL calls `/usr/src/tensorrt/bin/trtexec` (a Jetson-specific hardcoded path).
On desktop with pip-installed TensorRT, this binary does not exist. The fix is:

1. Patch the hardcoded path out of NanoOWL source
2. Drop a Python-based `trtexec` wrapper into the conda env's `bin/`

### 6a — Create the wrapper

```bash
cat > $CONDA_PREFIX/bin/trtexec << 'EOF'
#!/usr/bin/env python3
import argparse
import tensorrt as trt
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--saveEngine', required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--shapes', default=None)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    trt_parser = trt.OnnxParser(network, logger)

    print(f"[trtexec-py] Parsing ONNX: {args.onnx}")
    with open(args.onnx, 'rb') as f:
        if not trt_parser.parse(f.read()):
            for i in range(trt_parser.num_errors):
                print(trt_parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if args.shapes:
        profile = builder.create_optimization_profile()
        for shape_str in args.shapes.split(','):
            name, dims_str = shape_str.split(':')
            dims = tuple(int(d) for d in dims_str.split('x'))
            profile.set_shape(name, dims, dims, dims)
        config.add_optimization_profile(profile)

    print("[trtexec-py] Building TensorRT engine (this may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        print("[trtexec-py] Engine build FAILED.")
        sys.exit(1)

    with open(args.saveEngine, 'wb') as f:
        f.write(serialized)

    print(f"[trtexec-py] Engine saved to: {args.saveEngine}")

if __name__ == '__main__':
    main()
EOF

chmod +x $CONDA_PREFIX/bin/trtexec
```

---

## Step 7 — Clone and Install torch2trt

`torch2trt` must be built from source. Always use `--no-build-isolation` so the build
can see the already-installed `tensorrt` package.

```bash
mkdir -p ~/Documents/Projects/nanoowl-stuff
cd ~/Documents/Projects/nanoowl-stuff

git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
uv pip install -e . --no-build-isolation
```

---

## Step 8 — Clone and Install NanoOWL

```bash
cd ~/Documents/Projects/nanoowl-stuff

git clone https://github.com/NVIDIA-AI-IOT/nanoowl
cd nanoowl
uv pip install -e . --no-build-isolation
```

### Patch the hardcoded trtexec path

```bash
sed -i 's|/usr/src/tensorrt/bin/trtexec|trtexec|g' \
    ~/Documents/Projects/nanoowl-stuff/nanoowl/nanoowl/owl_predictor.py
```

### Patch read-only numpy array issues (OpenCV 4.9+ + numpy 2.x)

OpenCV 4.9+ strictly enforces numpy's read-only flag. PIL images converted via
`np.asarray()` are read-only by design, which causes `cv2.rectangle()` to crash.
The fix patches all affected drawing files at once:

```bash
# Fix all *_drawing.py and image_preprocessor.py in one shot
grep -rl "np.asarray(image)$" \
    ~/Documents/Projects/nanoowl-stuff/nanoowl/nanoowl/ \
    | xargs sed -i 's/image = np.asarray(image)$/image = np.asarray(image).copy()/'

# Fix image_preprocessor.py separately (different pattern)
sed -i 's/torch.from_numpy(np.asarray(image))/torch.from_numpy(np.asarray(image).copy())/' \
    ~/Documents/Projects/nanoowl-stuff/nanoowl/nanoowl/image_preprocessor.py
```

Verify all patches landed:

```bash
grep -rn "\.copy()" ~/Documents/Projects/nanoowl-stuff/nanoowl/nanoowl/
# Should show hits in: owl_drawing.py, tree_drawing.py, image_preprocessor.py
```

---

## Step 9 — Full Import Sanity Check

```bash
python -c "import torch, tensorrt, torch2trt, transformers, nanoowl; print('All OK')"
```

All 5 must print `All OK` before proceeding.

---

## Step 10 — Build the TensorRT Engine

This compiles OWL-ViT's image encoder specifically for your GPU. Takes 3–8 minutes.
Only needs to be done once.

```bash
cd ~/Documents/Projects/nanoowl-stuff/nanoowl
mkdir -p data

python -m nanoowl.build_image_encoder_engine \
    data/owl_image_encoder_patch32.engine
```

You will see `[trtexec-py] Engine saved to: data/owl_image_encoder_patch32.engine` on success.

---

## Step 11 — Run Inference

All examples are run from the `examples/` directory with the engine path pointing to `data/`.

### Basic object detection

```bash
cd ~/Documents/Projects/nanoowl-stuff/nanoowl/examples

python owl_predict.py \
    --image /path/to/your/image.jpg \
    --prompt "[a person, a chair, a laptop]" \
    --threshold 0.1 \
    --image_encoder_engine ../data/owl_image_encoder_patch32.engine \
    --output ../data/predict_out.jpg

eog ../data/predict_out.jpg
```

### Hierarchical / nested detection (NanoOWL's unique feature)

Detects objects *within* objects using a bracket syntax:

```bash
python tree_predict.py \
    --image /path/to/your/image.jpg \
    --prompt "[a person [a face, a hand], a chair]" \
    --threshold 0.1 \
    --image_encoder_engine ../data/owl_image_encoder_patch32.engine \
    --output ../data/tree_out.jpg

eog ../data/tree_out.jpg
```

### Interactive web demo

Serves a local web UI where you can type prompts live against your webcam:

```bash
cd ~/Documents/Projects/nanoowl-stuff/nanoowl/examples/tree_demo

python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
# Then open http://localhost:8080 in your browser
```

### Live webcam (headless)

```bash
cd ~/Documents/Projects/nanoowl-stuff/nanoowl/examples

python owl_predict_camera.py \
    --prompt "[a person, a monitor, a chair]" \
    --threshold 0.1 \
    --image_encoder_engine ../data/owl_image_encoder_patch32.engine
```

---

## Warnings You Will See (All Safe to Ignore)

| Warning | Cause | Action |
|---|---|---|
| `UNEXPECTED` keys (`position_ids`) | Cross-architecture weight loading | Ignore |
| `OwlViTImageProcessor fast processor` | Transformers 5.x default change | Ignore |
| `torch.meshgrid indexing argument` | Deprecated API in PyTorch 2.5 | Ignore |
| `Using default stream in enqueueV3()` | TRT prefers explicit CUDA streams | Ignore for single-image; worth fixing for video loops |

---

## Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `posixpath UnboundLocalError` | PyTorch 2.6 regression on Python 3.10 | Pin `torch==2.5.1` |
| `tensorrt-cu13` auto-installed | NVIDIA index defaults to latest CUDA | Install `tensorrt-cu12` explicitly |
| `No module named tensorrt` during build | uv build isolation sandbox | Always use `--no-build-isolation` for torch2trt and nanoowl |
| `FileNotFoundError: trtexec` | Hardcoded Jetson path in source | Patch `owl_predictor.py` + create Python wrapper in `$CONDA_PREFIX/bin/` |
| `cv2.error: readonly NumPy array` | OpenCV 4.9+ strict read-only enforcement | Patch all `*_drawing.py` files and `image_preprocessor.py` with `.copy()` |
| `No module named transformers` | Missing dep before engine build | Install all deps (Step 5) before cloning repos |
| `No module named clip` | OpenAI CLIP not on PyPI | `uv pip install git+https://github.com/openai/CLIP.git` |
| `No module named aiohttp` | Web demo dependency | `uv pip install aiohttp` |

---

## Environment Summary

| Component | Version |
|---|---|
| Python | 3.10.11 (mamba) |
| PyTorch | 2.5.1+cu124 |
| TensorRT | 10.15.x (tensorrt-cu12) |
| torch2trt | 0.5.0 (from source) |
| nanoowl | 0.0.0 (from source) |
| transformers | 5.x |
| opencv-python | 4.13.x |
| openai-clip | latest (from GitHub) |
| aiohttp | latest |
