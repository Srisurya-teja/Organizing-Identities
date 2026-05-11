

Automatically separates mixed face images inside identity folders into per-person subfolders using a  InsightFace buffalo_l — ResNet50 ArcFace —model

---

## What it does

Given a root directory where each subfolder contains a mix of face images belonging to potentially more than one person, the script:

1. Reads every image in each identity folder
2. Rejects blurry images (Laplacian variance check)
3. Embeds each image into a 512-dimensional vector using your ONNX model
4. Clusters those vectors by cosine distance (greedy nearest-neighbour)
5. Moves images into subfolders named `<identity>_1`, `<identity>_2`, etc.
6. Quarantines unreadable or blurry images into `<identity>_unreadable/`

**Before:**
```
identities/
    john_doe/
        img1.jpg    ← person A
        img2.jpg    ← person B
        img3.jpg    ← person A
    jane_smith/
        a.png
        b.png
```

**After:**
```
identities/
    john_doe/
        john_doe_1/
            img1.jpg
            img3.jpg
        john_doe_2/
            img2.jpg
    jane_smith/
        jane_smith_1/
            a.png
            b.png
```



---


Inspected model specs:

| Property | Value |
|----------|-------|
| Input tensor name | `input.1` |
| Input shape | `[N, 3, 112, 112]` float32 |
| Input format | BGR, normalised to `[-1, 1]` via `(pixel − 127.5) / 127.5` |
| Output tensor name | `683` |
| Output shape | `[1, 512]` float32 |
| Embedding type | L2-normalised 512-d ArcFace vector |
| Parameters | ~43.6 million |
| Architecture | ResNet-50 (24 residual blocks, PReLU activations) |

Images must already be **face-cropped and aligned** before being placed in the identity folders. The script does not perform face detection.

### Python packages

**CPU inference:**
```bash
pip install onnxruntime opencv-python-headless numpy tqdm
```

**GPU inference (CUDA):**
```bash
pip install onnxruntime-gpu opencv-python-headless numpy tqdm
```

> `onnxruntime` and `onnxruntime-gpu` cannot coexist in the same environment — install only one. GPU inference requires a compatible CUDA and cuDNN installation.

---

## Usage

```bash
python organize_identities.py \
    --root  /path/to/identities \
    --model /path/to/model.onnx
```

Always do a dry run first to preview moves without touching any files:

```bash
python organize_identities.py \
    --root  /path/to/identities \
    --model /path/to/model.onnx \
    --dry-run
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--root` | *(required)* | Root directory containing identity subfolders |
| `--model` | *(required)* | Path to the ArcFace `.onnx` model file |
| `--device` | `auto` | Execution device: `auto` / `cpu` / `cuda` / `cuda:0` / `cuda:1` |
| `--batch-size` | `32` | Images per ONNX forward pass — larger is faster on GPU |
| `--tolerance` | `0.5` | Cosine-distance threshold for clustering (see below) |
| `--blur-threshold` | `25` | Laplacian variance cutoff — images below this score are skipped |
| `--dry-run` | off | Print all planned moves without touching any files |

---

## Tolerance

Tolerance is the cosine-distance threshold that decides whether two face embeddings belong to the same person.

Cosine distance ranges from 0 (identical vectors) to 2 (opposite). In practice, same-person pairs score well below 0.5 and different-person pairs score above it.

| Value | Behaviour |
|---|---|
| `0.3` | Strict — same person in varying lighting may split into two clusters |
| `0.5` | Balanced — handles sunglasses, mild aging, different expressions *(default)* |
| `0.6` | Lenient — more forgiving; look-alike faces may occasionally merge |

**How clustering works:**

The algorithm processes images in alphabetical order. For each image:
- Compute its cosine distance to every existing cluster's representative (running mean embedding)
- If the closest cluster is within tolerance → join it; update the representative
- If no cluster is within tolerance → create a new cluster

This is a greedy nearest-neighbour approach — each image is assigned once and not revisited.

---

## Blur filtering

Before embedding, every image is scored for sharpness using the Laplacian variance method:

```python
gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
```

A sharp image has strong edges → high variance. A blurry image has soft edges → low variance. Images scoring below `--blur-threshold` are treated as unreadable and moved to `<folder>_unreadable/` instead of being embedded.

| Score | Typical image |
|---|---|
| 0 – 30 | Heavy motion blur or severely out of focus |
| 30 – 80 | Mild blur, degraded detail |
| 80 – 200 | Acceptable sharpness |
| 200+ | Very sharp, high detail |

To disable blur filtering entirely:

```bash
--blur-threshold 0
```

---

## Device selection

| `--device` value | Behaviour |
|---|---|
| `auto` *(default)* | Uses CUDA (GPU 0) if available, falls back to CPU |
| `cpu` | Forces CPU execution |
| `cuda` | Forces CUDA on GPU 0 |
| `cuda:0`, `cuda:1` | Forces CUDA on a specific GPU by index |

GPU inference is significantly faster for large datasets due to batched ONNX calls. The batch size (`--batch-size`) controls how many images are stacked into one ONNX forward pass. Increase it on high-VRAM GPUs for better throughput.

---

## Preprocessing pipeline

Each image goes through the following steps before being fed to the model:

```
BGR image  →  blur check  →  resize 112×112  →  (pixel − 127.5) / 127.5  →  NCHW reshape  →  model.onnx  →  512-d embedding  →  L2 normalise
```

The model expects BGR channel order (OpenCV default) — no channel swap is performed.

---

## Supported image formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp`

---

## Output folder naming

| Subfolder | Contents |
|---|---|
| `<identity>_1/` | Images belonging to the first detected cluster |
| `<identity>_2/` | Images belonging to the second detected cluster |
| `<identity>_N/` | … and so on |
| `<identity>_unreadable/` | Images OpenCV could not open, or images rejected by blur filter |

---

## Example runs

```bash
# Basic run on CPU
python organize_identities.py \
    --root /data/identities \
    --model /models/model.onnx

# GPU, strict clustering, strict blur filter
python organize_identities.py \
    --root /data/identities \
    --model /models/model.onnx \
    --device cuda \
    --tolerance 0.4 \
    --blur-threshold 120

# Lenient clustering, no blur filter, large batch
python organize_identities.py \
    --root /data/identities \
    --model /models/model.onnx \
    --device cuda:1 \
    --batch-size 64 \
    --tolerance 0.6 \
    --blur-threshold 0

# Dry run — preview only
python organize_identities.py \
    --root /data/identities \
    --model /models/model.onnx \
    --dry-run
```

---

## Caveats and known limitations

**Greedy assignment is permanent.** Each image is assigned to a cluster once, in alphabetical order, and never re-evaluated. If a blurry or unusual image is processed first, it becomes the cluster's founding representative and may inflate distances for subsequent images. Mitigation: use `--blur-threshold` to skip low-quality images, or rename files so the best-quality images sort first alphabetically.

**No re-clustering.** The algorithm does not run multiple passes or optimise globally. For a perfect separation, review the `_unreadable/` folder and re-run after cleaning up low-quality images.
