import sys
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found.\nRun: pip install opencv-python-headless")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found.\nRun: pip install onnxruntime")


# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ArcFace / InsightFace preprocessing — matches your model.onnx exactly
INPUT_SIZE       = (112, 112)       # model input spatial size
INPUT_NAME       = "input.1"        # input  tensor name  (inspected)
OUTPUT_NAME      = "683"            # output tensor name  (inspected)
MEAN             = np.float32(127.5)
STD              = np.float32(127.5)

# Blur detection — Laplacian variance threshold.
# Images scoring below this are considered too blurry to embed reliably
# and are moved to <folder>_unreadable/ instead of being clustered.
# Raise this value to reject more images; lower it to be more permissive.
# Typical range: 50 (lenient) – 150 (strict).
BLUR_THRESHOLD   = 25
DEFAULT_BATCH    = 32               # images per ONNX forward pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)




def build_providers(device: str) -> list:
    """
    Return an ordered list of ONNX Runtime execution providers for the
    requested device string.

    device values
    -------------
    "auto"      CUDA if available, CPU otherwise (default)
    "cpu"       Force CPU
    "cuda"      CUDA on GPU 0 with default options
    "cuda:N"    CUDA on GPU N  (e.g. "cuda:1" for the second GPU)

    Provider options
    ----------------
    CUDAExecutionProvider is configured with:
      - arena_extend_strategy = kNextPowerOfTwo  (efficient memory growth)
      - gpu_mem_limit          = 2 GiB           (guards against OOM on small GPUs)
      - cudnn_conv_algo_search = EXHAUSTIVE       (finds fastest conv algo once)
      - do_copy_in_default_stream = True          (thread-safe D→H copies)
    """
    available = ort.get_available_providers()

    cuda_options = {
        "arena_extend_strategy":    "kNextPowerOfTwo",
        "gpu_mem_limit":            2 * 1024 ** 3,   # 2 GiB
        "cudnn_conv_algo_search":   "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }

    if device == "auto":
        if "CUDAExecutionProvider" in available:
            cuda_options["device_id"] = 0
            log.info("Device: CUDA (GPU 0)  [auto-selected]")
            return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        log.info("Device: CPU  [CUDA not available, auto-fallback]")
        return ["CPUExecutionProvider"]

    if device == "cpu":
        log.info("Device: CPU  [forced]")
        return ["CPUExecutionProvider"]

    if device.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            sys.exit(
                "ERROR: CUDA requested but CUDAExecutionProvider is not available.\n"
                "Make sure onnxruntime-gpu is installed and CUDA drivers are present.\n"
                "Run: pip install onnxruntime-gpu"
            )
        gpu_id = int(device.split(":")[1]) if ":" in device else 0
        cuda_options["device_id"] = gpu_id
        log.info("Device: CUDA (GPU %d)  [explicit]", gpu_id)
        return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]

    sys.exit(
        f"ERROR: Unknown device '{device}'. "
        "Valid values: auto | cpu | cuda | cuda:0 | cuda:1 …"
    )


# ── Embedder ──────────────────────────────────────────────────────────────────

class FaceEmbedder:
    """
    Loads the ONNX model on the requested device and produces 512-d
    L2-normalised embeddings for pre-cropped face images.

    Supports batched inference — multiple images are stacked into a single
    ONNX call, which is especially beneficial on GPU where individual small
    forward passes leave most of the device idle.
    """

    def __init__(self, model_path: Path, device: str, batch_size: int):
        providers    = build_providers(device)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.batch_size = batch_size

        active = self.session.get_providers()[0]
        log.info("Model loaded: %s  |  Active provider: %s", model_path.name, active)
        if "CUDA" in active:
            log.info("Running on GPU — batch size: %d", batch_size)
        else:
            log.info("Running on CPU — batch size: %d", batch_size)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_one(self, bgr: np.ndarray) -> np.ndarray:
        """
        Single image: HxWx3 BGR uint8  →  3x112x112 float32  (CHW, normalised).
        """
        resized = cv2.resize(bgr, INPUT_SIZE)                        # 112×112×3
        normed  = (resized.astype(np.float32) - MEAN) / STD         # [-1, 1]
        return np.transpose(normed, (2, 0, 1))                       # 3×112×112

    # ── Single image (used as a fallback) ─────────────────────────────────────

    def get_embedding(self, image_path: Path):
        """
        Embed one image. Returns a 512-d L2-normalised ndarray, or None if:
          - OpenCV cannot read the file, or
          - the image is too blurry (Laplacian variance < BLUR_THRESHOLD).
        Both cases send the image to <folder>_unreadable/.
        """
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            log.warning("Cannot read: %s", image_path.name)
            return None

        # ── Blur check ────────────────────────────────────────────────────────
        # Laplacian highlights edges. A sharp image has strong edges → high
        # variance. A blurry image has soft/no edges → variance near zero.
        # Computed on the original resolution before resizing for best accuracy.
        gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < BLUR_THRESHOLD:
            log.warning(
                "Too blurry (score %.1f < %.1f): %s",
                blur_score, BLUR_THRESHOLD, image_path.name,
            )
            return None
        # ── End blur check ────────────────────────────────────────────────────

        blob   = self._preprocess_one(bgr)[np.newaxis, ...]          # 1×3×112×112
        output = self.session.run([OUTPUT_NAME], {INPUT_NAME: blob})
        return self._normalise(output[0][0])

    # ── Batched inference (primary path) ──────────────────────────────────────

    def get_embeddings_batch(self, image_paths: list) -> dict:
        """
        Embed a list of image paths in batches of self.batch_size.

        Returns
        -------
        embeddings_map : {Path: ndarray}   successfully embedded images
        unreadable     : [Path]            images OpenCV could not open
        """
        embeddings_map = {}
        unreadable     = []

        # Pre-load images, skip unreadable ones
        loaded = []
        for p in image_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                log.warning("Cannot read: %s", p.name)
                unreadable.append(p)
            else:
                loaded.append((p, bgr))

        # Run in batches
        for batch_start in range(0, len(loaded), self.batch_size):
            batch = loaded[batch_start : batch_start + self.batch_size]

            # Stack into [B, 3, 112, 112]
            batch_tensor = np.stack(
                [self._preprocess_one(bgr) for _, bgr in batch]
            )                                                         # B×3×112×112

            outputs = self.session.run([OUTPUT_NAME], {INPUT_NAME: batch_tensor})
            batch_embs = outputs[0]                                   # B×512

            for (path, _), emb in zip(batch, batch_embs):
                embeddings_map[path] = self._normalise(emb.astype(np.float32))

        return embeddings_map, unreadable

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(emb: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D embedding vector in place."""
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ── Clustering ────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical vectors."""
    return float(1.0 - np.dot(a, b))


def cluster_embeddings(embeddings_map: dict, tolerance: float) -> dict:
    """
    Greedy nearest-neighbour clustering by cosine distance.

    Parameters
    ----------
    embeddings_map : {Path: np.ndarray}   L2-normalised 512-d embeddings
    tolerance      : cosine distance threshold — same person if dist ≤ tolerance

    Returns
    -------
    {Path: int}   cluster id (0-based) for each image
    """
    paths = list(embeddings_map.keys())
    embs  = [embeddings_map[p] for p in paths]

    cluster_ids             = [-1] * len(paths)
    next_cluster            = 0
    cluster_representatives = []   # running-mean embedding per cluster

    for i, emb in enumerate(embs):
        if not cluster_representatives:
            cluster_ids[i] = next_cluster
            cluster_representatives.append(emb.copy())
            next_cluster += 1
            continue

        distances = np.array([cosine_distance(emb, r) for r in cluster_representatives])
        best_idx  = int(np.argmin(distances))

        if distances[best_idx] <= tolerance:
            cluster_ids[i] = best_idx
            # Update representative as running mean and re-normalise
            n = sum(1 for c in cluster_ids if c == best_idx)
            rep = (cluster_representatives[best_idx] * (n - 1) + emb) / n
            rep_norm = np.linalg.norm(rep)
            cluster_representatives[best_idx] = rep / rep_norm if rep_norm > 0 else rep
        else:
            cluster_ids[i] = next_cluster
            cluster_representatives.append(emb.copy())
            next_cluster += 1

    return {paths[i]: cluster_ids[i] for i in range(len(paths))}


# ── File helpers ──────────────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def safe_move(src: Path, dst_dir: Path, dry_run: bool):
    """Move src into dst_dir, creating it if needed, resolving name collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        stem, suffix = src.stem, src.suffix
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    if dry_run:
        log.info("[DRY-RUN]  %s  →  %s/", src.name, dst_dir.name)
    else:
        shutil.move(str(src), str(dst))
        log.debug("Moved  %s  →  %s/", src.name, dst_dir.name)


# ── Per-folder logic ──────────────────────────────────────────────────────────

def process_identity_folder(
    folder: Path,
    embedder: FaceEmbedder,
    tolerance: float,
    dry_run: bool,
):
    """
    Embed every image in `folder` (in batches), cluster by cosine distance,
    and move each cluster to <folder_name>_1/, <folder_name>_2/, etc.
    """
    folder_name = folder.name
    images      = sorted(p for p in folder.iterdir() if is_image(p))

    if not images:
        log.info("  [%s]  No images — skipping.", folder_name)
        return

    log.info("  [%s]  %d image(s) found.", folder_name, len(images))

    embeddings_map, unreadable = embedder.get_embeddings_batch(
        tqdm(images, desc=f"    {folder_name}", leave=False)
    )

    log.info(
        "  [%s]  Embedded: %d  |  Unreadable: %d",
        folder_name, len(embeddings_map), len(unreadable),
    )

    # ── Cluster and move ──────────────────────────────────────────────────────
    if embeddings_map:
        assignment = cluster_embeddings(embeddings_map, tolerance)

        clusters: dict[int, list[Path]] = defaultdict(list)
        for path, cid in assignment.items():
            clusters[cid].append(path)

        log.info(
            "  [%s]  Distinct person(s) detected: %d", folder_name, len(clusters)
        )

        for cid in sorted(clusters.keys()):
            subfolder_name = f"{folder_name}_{cid + 1}"   # e.g. john_doe_1
            dst_dir        = folder / subfolder_name
            log.info(
                "  [%s]  → %s/  (%d image(s))",
                folder_name, subfolder_name, len(clusters[cid]),
            )
            for img_path in clusters[cid]:
                safe_move(img_path, dst_dir, dry_run)

    # ── Quarantine unreadable images ──────────────────────────────────────────
    if unreadable:
        bad_dir = folder / f"{folder_name}_unreadable"
        log.warning(
            "  [%s]  Moving %d unreadable image(s) → %s/",
            folder_name, len(unreadable), bad_dir.name,
        )
        for img_path in unreadable:
            safe_move(img_path, bad_dir, dry_run)


# ── Orchestration ─────────────────────────────────────────────────────────────

def organise(root: Path, model_path: Path, device: str, batch_size: int, tolerance: float, dry_run: bool):
    embedder         = FaceEmbedder(model_path, device, batch_size)
    identity_folders = sorted(p for p in root.iterdir() if p.is_dir())

    if not identity_folders:
        log.warning("No subdirectories found in '%s'.", root)
        return

    log.info(
        "Root: '%s'  |  Folders: %d  |  Tolerance: %.2f  |  Batch: %d",
        root, len(identity_folders), tolerance, batch_size,
    )
    if dry_run:
        log.info("*** DRY-RUN — no files will be moved ***")

    for folder in identity_folders:
        loose_images = [p for p in folder.iterdir() if is_image(p)]
        if not loose_images:
            log.info("  [%s]  No loose images — skipping.", folder.name)
            continue
        process_identity_folder(folder, embedder, tolerance, dry_run)

    log.info("All done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster pre-cropped face images by ArcFace embedding and organise "
            "into subfolders named <identity>_1, <identity>_2, …"
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing identity subfolders.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the ArcFace ONNX model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Execution device: auto | cpu | cuda | cuda:0 | cuda:1 …  "
            "(default: auto — uses CUDA if available, CPU otherwise)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=(
            f"Number of images per ONNX forward pass (default: {DEFAULT_BATCH}). "
            "Larger values are faster on GPU but use more VRAM."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help=(
            "Cosine-distance threshold for clustering. "
            "0.3 = strict · 0.5 = balanced (default) · 0.7 = lenient"
        ),
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=30.0,
        help=(
            "Laplacian variance threshold for blur detection. Images scoring "
            "below this are skipped and moved to <folder>_unreadable/. "
            "50 = lenient · 80 = balanced (default) · 150 = strict. "
            "Set to 0 to disable blur filtering entirely."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving any files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply blur threshold from CLI to the module-level constant
    BLUR_THRESHOLD = args.blur_threshold

    if not args.root.is_dir():
        sys.exit(f"ERROR: '{args.root}' is not a valid directory.")
    if not args.model.is_file():
        sys.exit(f"ERROR: '{args.model}' is not a valid file.")

    organise(args.root, args.model, args.device, args.batch_size, args.tolerance, args.dry_run)