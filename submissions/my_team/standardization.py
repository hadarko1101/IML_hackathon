from PIL import Image, ImageOps
from dataclasses import dataclass, field
import torch
import numpy as np


TARGET_SIZE = 224
VALID_MODES = {"RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"}



# ---------------------------------------------------------------------------
# Corrupted Data Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Holds the outcome of a batch image validation pass."""
    valid_images: list       = field(default_factory=list)
    valid_indices: list      = field(default_factory=list)
    corrupted_indices: list  = field(default_factory=list)
    corruption_reasons: dict = field(default_factory=dict)  # index -> reason str

    @property
    def num_valid(self) -> int:
        return len(self.valid_images)

    @property
    def num_corrupted(self) -> int:
        return len(self.corrupted_indices)

    def summary(self) -> str:
        lines = [
            f"Total images : {self.num_valid + self.num_corrupted}",
            f"Valid        : {self.num_valid}",
            f"Corrupted    : {self.num_corrupted}",
        ]
        if self.corruption_reasons:
            lines.append("Corruption details:")
            for idx, reason in self.corruption_reasons.items():
                lines.append(f"  [index {idx}] {reason}")
        return "\n".join(lines)


def validate_images(images: list) -> ValidationResult:
    """
    Validate a list of images for common corruption signals.

    Checks performed on every image:
      1. Not None         - image object must exist.
      2. Valid type       - must be a PIL.Image.Image instance.
      3. Non-zero size    - width and height must both be > 0.
      4. Recognised mode  - e.g. RGB, RGBA, L, CMYK, etc.
      5. Pixel integrity  - forces a full decode to catch truncated /
                            structurally broken files.

    Args:
        images: List of objects to validate (expected PIL.Image.Image).

    Returns:
        ValidationResult containing clean images, their original indices,
        and a per-image corruption report.
    """
    result = ValidationResult()

    for idx, img in enumerate(images):
        reason = _check_image(img)
        if reason:
            result.corrupted_indices.append(idx)
            result.corruption_reasons[idx] = reason
        else:
            result.valid_images.append(img)
            result.valid_indices.append(idx)

    return result


def _check_image(img) -> str:
    """Return a non-empty reason string if the image is corrupted, else ''."""

    # 1. Existence check
    if img is None:
        return "Image is None"

    # 2. Type check
    if not isinstance(img, Image.Image):
        return f"Expected PIL.Image, got {type(img).__name__}"

    # 3. Dimension check
    try:
        w, h = img.size
    except Exception as exc:
        return f"Cannot read dimensions: {exc}"

    if w == 0 or h == 0:
        return f"Zero dimension: width={w}, height={h}"

    # 4. Mode check
    if img.mode not in VALID_MODES:
        return f"Unrecognised image mode: '{img.mode}'"

    # 5. Pixel-data integrity check (force full decode)
    try:
        img.load()
    except Exception as exc:
        return f"Pixel data unreadable: {exc}"

    return ""  # no corruption found


# ---------------------------------------------------------------------------
# EXIF Orientation Fix
# ---------------------------------------------------------------------------

def fix_orientation(images: list) -> list:
    """
    Apply EXIF orientation tags so pixel data matches visual orientation.

    Phone cameras often store photos in a fixed pixel orientation and record
    the actual rotation in EXIF metadata.  PIL does NOT auto-rotate, so
    without this fix the model would see sideways / upside-down images.

    Args:
        images: List of PIL.Image objects.

    Returns:
        List of PIL.Image objects with correct pixel orientation.
    """
    fixed = []
    for img in images:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass  # no EXIF data or unsupported tag — keep image as-is
        fixed.append(img)
    return fixed


# ---------------------------------------------------------------------------
# Color Space Conversion
# ---------------------------------------------------------------------------

def convert_to_rgb(images: list) -> list:
    """
    Convert every image to the RGB color space.

    Handles:
      - RGBA  → composites alpha over a white background, then converts.
      - L / P → converts directly to RGB.
      - CMYK / YCbCr / LAB / etc. → converts directly to RGB.
      - RGB   → returned unchanged.

    Args:
        images: List of PIL.Image objects (any mode).

    Returns:
        List of PIL.Image objects, all in mode 'RGB'.
    """
    converted = []
    for img in images:
        if img.mode == "RGB":
            converted.append(img)
        elif img.mode == "RGBA":
            # Composite over white to avoid black artifacts from alpha
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 = alpha channel
            converted.append(background)
        else:
            converted.append(img.convert("RGB"))
    return converted


# ---------------------------------------------------------------------------
# Short-Edge Resize + Center Crop
# ---------------------------------------------------------------------------

def short_edge_resize_and_crop(images: list, target_size: int = TARGET_SIZE) -> list:
    """
    Resize a batch of PIL Images to (target_size x target_size) using
    Short-Edge Resize + Center Crop:

      1. Scale so the shorter edge == target_size (aspect ratio preserved).
      2. Center-crop to target_size x target_size.

    Args:
        images:      List of PIL.Image objects.
        target_size: Output resolution (default 224).

    Returns:
        List of PIL.Image objects each sized (target_size, target_size).
    """
    processed = []

    for img in images:
        w, h = img.size

        # Step 1 — Short-edge resize
        if w <= h:
            new_w = target_size
            new_h = int(round(h * target_size / w))
        else:
            new_h = target_size
            new_w = int(round(w * target_size / h))

        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Step 2 — Center crop
        left   = (new_w - target_size) // 2
        top    = (new_h - target_size) // 2
        right  = left + target_size
        bottom = top  + target_size

        img = img.crop((left, top, right, bottom))
        processed.append(img)

    return processed


# ---------------------------------------------------------------------------
# Tensor Conversion
# ---------------------------------------------------------------------------

def to_tensors(images: list) -> torch.Tensor:
    """
    Convert a list of PIL Images to a batched float32 torch tensor.

    Each image is converted to a numpy array, scaled from [0, 255] to
    [0.0, 1.0], and rearranged from (H, W, C) to (C, H, W).

    Args:
        images: List of PIL.Image objects (should all be the same size).

    Returns:
        torch.Tensor of shape (N, C, H, W) with dtype float32, values in [0, 1].
    """
    tensors = []
    for img in images:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0   # (H, W, 3)
        arr = np.transpose(arr, (2, 0, 1))                  # (3, H, W)
        tensors.append(torch.from_numpy(arr))

    return torch.stack(tensors)  # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(batch: torch.Tensor) -> torch.Tensor:
    """
    Channel-wise normalize a batch of image tensors using batch statistics.

    Computes the per-channel mean and std from the batch itself, then applies:
        output[c] = (input[c] - mean[c]) / std[c]

    Args:
        batch: Tensor of shape (N, C, H, W), values in [0, 1].

    Returns:
        Normalized tensor of the same shape and dtype.
    """
    # Compute mean and std across batch, height, and width (dims 0, 2, 3)
    mean = batch.mean(dim=[0, 2, 3]).view(1, -1, 1, 1)
    std  = batch.std(dim=[0, 2, 3]).view(1, -1, 1, 1)

    # Avoid division by zero for constant channels
    std = torch.clamp(std, min=1e-8)

    return (batch - mean) / std


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def preprocess_batch(images: list, target_size: int = TARGET_SIZE) -> torch.Tensor:
    # Step 1 — Validate
    result = validate_images(images)

    if result.num_corrupted > 0:
        print(result.summary())

    # Step 2 — Fix EXIF orientation
    oriented = fix_orientation(result.valid_images)

    # Step 3 — Convert all images to RGB
    rgb_images = convert_to_rgb(oriented)

    # Step 4 — Resize to target_size x target_size
    resized = short_edge_resize_and_crop(rgb_images, target_size)

    # Step 5 — Convert to float32 tensors (values in [0, 1])
    batch = to_tensors(resized)

    # Step 6 — Normalize (batch mean/std)
    batch = normalize(batch)

    return batch


# ---------------------------------------------------------------------------
# Tensor → PIL Image (for visual inspection)
# ---------------------------------------------------------------------------

def tensor_to_images(batch: torch.Tensor) -> list:
    """
    Convert a batch tensor back to a list of PIL Images for visual inspection.

    Handles both normalized and unnormalized tensors:
      - If values are outside [0, 1] (i.e. normalized), rescales each image
        to [0, 255] using min-max scaling.
      - If values are in [0, 1], simply scales to [0, 255].

    Args:
        batch: Tensor of shape (N, C, H, W) or (C, H, W) for a single image.

    Returns:
        List of PIL.Image objects in RGB mode.
    """
    # Handle single image (C, H, W) → add batch dimension
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)

    images = []
    for i in range(batch.shape[0]):
        img_tensor = batch[i].detach().cpu().float()  # (C, H, W)

        # Undo normalization via min-max rescale to [0, 1]
        for c in range(img_tensor.shape[0]):
            c_min = img_tensor[c].min()
            c_max = img_tensor[c].max()
            if c_max - c_min > 1e-8:
                img_tensor[c] = (img_tensor[c] - c_min) / (c_max - c_min)
            else:
                img_tensor[c] = torch.zeros_like(img_tensor[c])

        # (C, H, W) → (H, W, C) → numpy → uint8
        arr = img_tensor.permute(1, 2, 0).numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr, mode="RGB"))

    return images


def save_debug_grid(
    batch: torch.Tensor,
    path: str = "debug_grid.png",
    max_images: int = 16,
    cols: int = 4,
    labels: list = None,
) -> None:
    """
    Save a grid of images from a batch tensor to disk for quick visual inspection.

    Args:
        batch:      Tensor of shape (N, C, H, W).
        path:       Output file path (e.g. 'debug_grid.png').
        max_images: Maximum number of images to include in the grid.
        cols:       Number of columns in the grid.
        labels:     Optional list of label strings/ints for each image.
    """
    images = tensor_to_images(batch[:max_images])
    n = len(images)
    rows = (n + cols - 1) // cols

    w, h = images[0].size
    padding = 4
    cell_w = w + padding
    cell_h = h + padding + (20 if labels else 0)

    grid = Image.new("RGB", (cols * cell_w + padding, rows * cell_h + padding), (40, 40, 40))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * cell_w + padding
        y = r * cell_h + padding
        grid.paste(img, (x, y))

    grid.save(path)
    print(f"Saved debug grid ({n} images, {rows}x{cols}) -> {path}")
