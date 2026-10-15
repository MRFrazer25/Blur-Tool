import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import atexit
import io
import os
import shutil
import tempfile
import logging
import math
import secrets
import threading
import time

# Apply the 25 MB upload cap even when this module is imported (e.g. `gradio app.py`
# or a Hugging Face Space) rather than started with `python app.py`.
os.environ.setdefault("GRADIO_MAX_FILE_SIZE", "25mb")

# Never let Ultralytics pip-install packages at runtime (e.g. when an upload fails to decode).
os.environ["YOLO_AUTOINSTALL"] = "False"
# Ultralytics replaces PIL's Image.open on import with a version that tries to install an
# extra plugin whenever a file fails to open. Keep the original and restore it below.
_pil_image_open = Image.open
from ultralytics import YOLO
Image.open = _pil_image_open

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Turn off Ultralytics' anonymous usage analytics for this process only
# (does not change the user's global Ultralytics settings file).
try:
    from ultralytics.utils.events import events as ultralytics_events
    ultralytics_events.enabled = False
except Exception as e:
    logger.warning(f"Could not disable Ultralytics analytics: {e}")

# Windows-only workaround for a Gradio upload race. Uploads are cached in a folder named
# after the file's content hash. When the image editor re-uploads an identical file,
# Windows refuses to rename over the existing copy, so Gradio rewrites it in a background
# task after the request returns - and the next event can read it half-written ("cannot
# identify image file"). An existing file at that path already has identical content,
# so the rewrite is skipped and the duplicate temp file is just removed.
if os.name == "nt":
    try:
        import gradio.routes as gradio_routes
        _gradio_move_uploads = gradio_routes.move_uploaded_files_to_cache

        def _move_uploads_skip_existing(files, destinations):
            new_files, new_destinations = [], []
            for file, dest in zip(files, destinations):
                if os.path.exists(dest):
                    try:
                        os.remove(file)
                    except OSError:
                        pass
                else:
                    new_files.append(file)
                    new_destinations.append(dest)
            _gradio_move_uploads(new_files, new_destinations)

        gradio_routes.move_uploaded_files_to_cache = _move_uploads_skip_existing
    except Exception as e:
        logger.warning(f"Could not apply Gradio upload workaround: {e}")

# Privacy Suggestions detection modes and the YOLO26 model each one uses
# (nano versions for the best speed/efficiency balance).
MODE_OBJECTS = "Whole objects"
MODE_OUTLINES = "Exact outlines"
MODE_FACES = "Faces only"
DETECTION_MODELS = {
    MODE_OBJECTS: "yolo26n.pt",        # Bounding boxes
    MODE_OUTLINES: "yolo26n-seg.pt",   # Segmentation masks (exact shapes)
    MODE_FACES: "yolo26n-pose.pt",     # Body keypoints, used to locate faces
}

# Models are loaded on first use (each downloads from Ultralytics on first run).
# If one fails, only that mode is unavailable.
yolo_models = {}
_yolo_failed_modes = set()
_yolo_load_lock = threading.Lock()

def get_yolo_model(mode):
    """Loads the YOLO weights for a detection mode on first use. Returns None if unavailable."""
    with _yolo_load_lock:
        if mode in yolo_models:
            return yolo_models[mode]
        if mode in _yolo_failed_modes:
            return None
        weights = DETECTION_MODELS.get(mode)
        if not weights:
            return None
        try:
            logger.info(f"Attempting to load {weights}...")
            yolo_models[mode] = YOLO(weights)
            logger.info(f"{weights} loaded successfully.")
            return yolo_models[mode]
        except Exception as e:
            _yolo_failed_modes.add(mode)
            logger.error(f"Failed to load {weights}: {e}. The '{mode}' suggestion mode will be unavailable.", exc_info=True)
            return None

# Object classes (COCO) suggested for blurring, and the minimum detection confidence
PRIVACY_TARGET_CLASSES = {
    'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'boat', 'airplane',
    'cell phone', 'laptop', 'tv', 'handbag', 'backpack', 'suitcase'
}
SUGGESTION_CONFIDENCE = 0.4
SUGGESTION_FILL = (255, 0, 0, 100)  # Semi-transparent red (~40% opacity)

# Pose keypoint indices (COCO order): nose, eyes and ears locate the face; shoulders give scale
FACE_KEYPOINTS = range(0, 5)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
KEYPOINT_CONFIDENCE = 0.5

# Upload limits: only real PNG/JPEG files are decoded, up to 40 megapixels
ALLOWED_IMAGE_FORMATS = ["PNG", "JPEG"]
MAX_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # Pillow rejects decompression bombs beyond 2x this

# Blur strength is a percent of image size (must match the slider). The old 1-101
# pixel kernel treated 1 as a no-op and could not anonymize faces on large photos.
MIN_BLUR_STRENGTH = 10
MAX_BLUR_STRENGTH = 100
DEFAULT_BLUR_STRENGTH = 50
MAX_EDITOR_LAYERS = 10
MAX_UPLOAD_FILE_SIZE = "25mb"
# ImageEditor on phones reloads the tab if it has to hold a full camera photo or screenshot.
# The editor gets a copy no wider/taller than this; blur still uses the original pixels.
MAX_EDITOR_SIDE = 1280
# On-screen previews (upload + result) stay smaller so phones can actually paint them.
MAX_SCREEN_SIDE = 960
MODE_PHONE = "Phone"
MODE_COMPUTER = "Computer"
PHONE_DRAWING_LIMITATION = (
    "On phones, tap the photo to mark areas to blur (each tap marks a circle; set its size with "
    "Tap Mark Size). Privacy Suggestions also work. Freehand drawing needs the Computer layout."
)
# Phone layout: tapping marks round areas instead of the drawing editor, which reloads phone tabs.
MIN_TAP_SIZE = 3        # Tap circle diameter, as a percent of the photo's shorter side
MAX_TAP_SIZE = 30
DEFAULT_TAP_SIZE = 15
MAX_TAP_MARKS = 200
_PHONE_UA_MARKERS = (
    "iphone", "ipod", "ipad", "android", "mobile", "webos", "blackberry", "iemobile", "opera mini",
)

# Resource limits for a public deployment
MAX_QUEUED_REQUESTS = 20         # Further requests get a "busy" error instead of piling up
# Every page view creates a session (even without an upload), so the cap must leave room for
# passive visitors; sessions without an upload hold almost nothing.
MAX_STORED_SESSIONS = 100        # Least recently used sessions beyond this are dropped from memory
SESSION_DATA_TTL_SECONDS = 1800  # Per-session data expires 30 minutes after the last upload, blur or suggestion

# Blurred results go in a per-process folder so another user cannot pre-create a
# shared name under the system temp directory. Files older than an hour are swept.
OUTPUT_DIR = tempfile.mkdtemp(prefix="blur-tool-outputs-")
OUTPUT_MAX_AGE_SECONDS = 3600
OUTPUT_SWEEP_INTERVAL_SECONDS = 600
atexit.register(lambda: shutil.rmtree(OUTPUT_DIR, ignore_errors=True))

# Helper Functions
def is_within_temp_dir(path):
    """Returns True if path resolves to a location inside the system temp directory,
    this process's output folder, or Gradio's upload folder (GRADIO_TEMP_DIR)."""
    resolved_path = os.path.realpath(path)
    allowed_dirs = [tempfile.gettempdir(), OUTPUT_DIR, os.environ.get("GRADIO_TEMP_DIR")]
    for allowed_dir in filter(None, allowed_dirs):
        allowed_dir = os.path.realpath(allowed_dir)
        try:
            if os.path.commonpath([resolved_path, allowed_dir]) == allowed_dir:
                return True
        except ValueError:  # Paths on different drives (Windows)
            continue
    return False

def encode_stored_image(img):
    """Compresses an uploaded image to PNG bytes (RGB unless alpha is actually used).
    Palette and grayscale PNGs with transparency keep it (stored as RGBA)."""
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img if img.mode == "RGBA" else img.convert("RGBA")
        alpha_min = rgba.getextrema()[3][0]
        stored = rgba if alpha_min < 255 else rgba.convert("RGB")
    elif img.mode == "RGB":
        stored = img
    else:
        stored = img.convert("RGB")
    buf = io.BytesIO()
    stored.save(buf, format="PNG")
    return buf.getvalue()

def capped_preview(img, max_side):
    """Smaller copy of img. Same aspect ratio; long side at most max_side."""
    if img is None:
        return None
    width, height = img.size
    long_side = max(width, height)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def editor_preview(img):
    """Smaller copy for ImageEditor. Same aspect ratio; long side at most MAX_EDITOR_SIDE."""
    return capped_preview(img, MAX_EDITOR_SIDE)


def screen_preview(img):
    """RGB copy for on-screen Image components. JPEG-safe and smaller than the editor copy."""
    preview = capped_preview(img, MAX_SCREEN_SIDE)
    if preview is None:
        return None
    if preview.mode in ("RGB", "L"):
        return preview
    return preview.convert("RGB")


def screen_size(size):
    """(width, height) of screen_preview for an image of this size."""
    width, height = size
    long_side = max(width, height)
    if long_side <= MAX_SCREEN_SIDE:
        return width, height
    scale = MAX_SCREEN_SIDE / float(long_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def normalize_tap_marks(taps):
    """Validated list of (x, y, radius) tuples in original-image pixels; [] if missing or malformed."""
    if not isinstance(taps, (list, tuple)):
        return []
    clean = []
    for tap in list(taps)[:MAX_TAP_MARKS]:
        try:
            x, y, radius = (float(v) for v in tap)
        except (TypeError, ValueError):
            continue
        if radius > 0 and all(math.isfinite(v) for v in (x, y, radius)):
            clean.append((x, y, radius))
    return clean


def taps_to_mask(taps, size):
    """0/1 mask of (width, height) with a filled circle for each tap."""
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, radius in normalize_tap_marks(taps):
        cv2.circle(mask, (int(round(x)), int(round(y))), max(1, int(round(radius))), 1, thickness=-1)
    return mask


def tap_view(source, taps):
    """Phone preview of the photo with tapped areas tinted red (drawn at preview size)."""
    preview = screen_preview(source)
    taps = normalize_tap_marks(taps)
    if preview is None or not taps:
        return preview
    scale = preview.width / float(source.width)
    scaled = [(x * scale, y * scale, r * scale) for x, y, r in taps]
    return render_marks_preview(preview, taps_to_mask(scaled, preview.size)).convert("RGB")


def is_phone_user_agent(user_agent):
    """True when the browser string looks like a phone or tablet."""
    if not user_agent:
        return False
    ua = user_agent.lower()
    return any(marker in ua for marker in _PHONE_UA_MARKERS)


def same_aspect(size_a, size_b, tol=0.02):
    """True when two (width, height) sizes have about the same aspect ratio."""
    width_a, height_a = size_a
    width_b, height_b = size_b
    if min(width_a, height_a, width_b, height_b) <= 0:
        return False
    return abs((width_a / height_a) - (width_b / height_b)) <= tol


def scale_mask_to_image(mask, size):
    """Resizes a 0/1 mask to (width, height) with nearest-neighbor so brush marks stay binary."""
    width, height = size
    if mask is None:
        return None
    if mask.shape == (height, width):
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def decode_stored_image(data):
    """Rebuilds a PIL image from compact session bytes (or returns a PIL image unchanged)."""
    if isinstance(data, Image.Image):
        return data
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as img:
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError):
        return None

def pack_marks(mask):
    """Stores a 0/1 mask as (height, width, packed bits) instead of a full-size array."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    return (height, width, np.packbits(arr.astype(bool)).tobytes())

def unpack_marks(packed, expected_shape=None):
    """Restores a packed marks tuple to a uint8 0/1 mask, or None if it is missing/wrong."""
    if packed is None:
        return None
    if isinstance(packed, np.ndarray):
        if packed.ndim != 2:
            return None
        if expected_shape is not None and packed.shape != expected_shape:
            return None
        return packed.astype(np.uint8)
    if not (isinstance(packed, (tuple, list)) and len(packed) == 3):
        return None
    height, width, blob = packed
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0 or not isinstance(blob, (bytes, bytearray)):
        return None
    bits = np.frombuffer(blob, dtype=np.uint8)
    mask = np.unpackbits(bits, count=height * width).reshape((height, width))
    if expected_shape is not None and mask.shape != expected_shape:
        return None
    return mask.astype(np.uint8)

def editor_matches_upload(flag):
    """False only when the server knows the editor does not hold the current upload (e.g. after a
    Phone-layout upload, which leaves the hidden editor with an older image and its brush strokes).
    None (expired or never set) keeps the old behavior, so brush strokes are never silently dropped."""
    return flag is not False

def suggestions_were_requested(flag):
    """True when the client says Privacy Suggestions are still expected to be applied."""
    return flag in (True, 1, "1", "true", "True")

def validate_editor_payload(editor_data, original_image):
    """Re-applies upload limits to blur/suggest payloads.

    Returns (source_image, layers, error_message). error_message is set on failure.
    The validated original (set only by handle_file_upload) is required; the editor
    background is never used as a fallback.
    """
    source = decode_stored_image(original_image)
    if source is None:
        return None, None, "Session expired — please upload the image again."
    if source.width * source.height > MAX_IMAGE_PIXELS:
        return None, None, (
            f"Image is too large ({source.width}x{source.height}). "
            f"Please resize it to under {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
        )
    if not editor_data or not isinstance(editor_data.get("background"), Image.Image):
        # Phone uploads never load the drawing editor. Suggestions and blur still work.
        return source, [], None
    background = editor_data["background"]
    if background.size != source.size and not same_aspect(background.size, source.size):
        return None, None, "The editor image does not match the uploaded image. Please upload again."
    layers = editor_data.get("layers") or []
    if not isinstance(layers, list):
        return None, None, "Internal error: drawing layers are invalid."
    if len(layers) > MAX_EDITOR_LAYERS:
        return None, None, f"Too many drawing layers (maximum is {MAX_EDITOR_LAYERS})."
    for layer in layers:
        if isinstance(layer, Image.Image) and layer.size != background.size:
            return None, None, "A drawing layer does not match the uploaded image size. Please upload again."
    return source, layers, None

def blur_sigma_for_image(strength, height, width):
    """Maps slider percent to a Gaussian sigma that scales with the shorter image side."""
    min_dim = max(1, min(int(height), int(width)))
    span = MAX_BLUR_STRENGTH - MIN_BLUR_STRENGTH
    t = 0.0 if span <= 0 else (float(strength) - MIN_BLUR_STRENGTH) / span
    t = min(1.0, max(0.0, t))
    # 1.5% to 8% of the shorter side; floor of 2 px so the weakest setting is never a no-op
    return max(2.0, (0.015 + 0.065 * t) * min_dim)

# Blurred output files created by this app, keyed by a random ID. Sessions only ever hold the
# ID, never a file path, so only files the app itself created can be deleted.
_output_files = {}
_output_files_lock = threading.Lock()

def register_output_file(file_path):
    """Records a blurred output file this app created and returns its random ID."""
    file_id = secrets.token_hex(16)
    with _output_files_lock:
        _output_files[file_id] = file_path
    return file_id

def remove_temp_file(file_id):
    """Deletes the blurred output file registered under file_id. Unknown IDs are ignored."""
    if not isinstance(file_id, str):
        return
    with _output_files_lock:
        file_path = _output_files.pop(file_id, None)
    if file_path is None:
        return
    try:
        os.remove(file_path)
        logger.info(f"Removed previous temporary download file: {file_path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Error removing old temp file {file_path}: {e}")

def sweep_old_outputs(max_age_seconds=OUTPUT_MAX_AGE_SECONDS):
    """Deletes blurred output files in OUTPUT_DIR older than max_age_seconds. Returns how many were deleted."""
    cutoff = time.time() - max_age_seconds
    deleted = 0
    try:
        entries = list(os.scandir(OUTPUT_DIR))
    except FileNotFoundError:
        return 0
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False) and entry.stat().st_mtime < cutoff:
                os.remove(entry.path)
                deleted += 1
        except OSError:
            pass
    # Forget registry entries whose files are gone
    with _output_files_lock:
        for file_id in [i for i, p in _output_files.items() if not os.path.exists(p)]:
            del _output_files[file_id]
    if deleted:
        logger.info(f"Swept {deleted} old blurred output file(s).")
    return deleted

def _sweep_outputs_forever():
    while True:
        sweep_old_outputs()
        time.sleep(OUTPUT_SWEEP_INTERVAL_SECONDS)

_sweeper_lock = threading.Lock()
_sweeper_thread = None

def _ensure_output_sweeper():
    """Starts the output sweeper once, including when the app is imported rather than run as main."""
    global _sweeper_thread
    with _sweeper_lock:
        if _sweeper_thread is not None and _sweeper_thread.is_alive():
            return
        _sweeper_thread = threading.Thread(
            target=_sweep_outputs_forever, daemon=True, name="output-sweeper"
        )
        _sweeper_thread.start()

# Core Functions
def layers_to_mask(layers, size):
    """Combines ImageEditor drawing layers into a binary mask (1 = marked) of the given (width, height).
    Any pixel with non-zero alpha on any layer counts as marked. Layers whose size
    does not match are ignored; callers must reject those before blurring."""
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in layers or []:
        if isinstance(layer, Image.Image) and layer.size == size:
            layer_alpha = layer.convert("RGBA").split()[-1]
            mask |= (np.array(layer_alpha) > 0).astype(np.uint8)
    return mask

def render_marks_preview(base_image, marks):
    """Returns the image with marked areas tinted semi-transparent red, for the Privacy Suggestions preview."""
    overlay = np.zeros((base_image.height, base_image.width, 4), dtype=np.uint8)
    overlay[marks.astype(bool)] = SUGGESTION_FILL
    return Image.alpha_composite(base_image.convert("RGBA"), Image.fromarray(overlay, "RGBA"))

def apply_gaussian_blur(image_np, mask_np_binary, blur_strength):
    """
    Applies a size-scaled Gaussian blur to marked regions.
    Strength is a percent (MIN_BLUR_STRENGTH–MAX_BLUR_STRENGTH), not a pixel kernel.
    Preserves an alpha channel when the input has one.
    """
    if image_np.shape[:2] != mask_np_binary.shape:
        logger.error("Image and mask dimensions mismatch. Cannot apply blur.")
        return image_np

    height, width = image_np.shape[:2]
    sigma = blur_sigma_for_image(blur_strength, height, width)
    has_alpha = image_np.ndim == 3 and image_np.shape[2] == 4
    rgb_image_np = image_np[:, :, :3]
    alpha_channel_np = image_np[:, :, 3] if has_alpha else None

    # Downscale before a large blur so 40 MP images stay interactive
    down = max(1, int(sigma / 8.0))
    if down > 1:
        small = cv2.resize(
            rgb_image_np,
            (max(1, width // down), max(1, height // down)),
            interpolation=cv2.INTER_AREA,
        )
        blurred_small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma / down))
        blurred_rgb_np = cv2.resize(blurred_small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        k_val = max(3, int(round(sigma * 6)) | 1)
        blurred_rgb_np = cv2.GaussianBlur(rgb_image_np, (k_val, k_val), sigma)

    mask_expanded_rgb = np.stack([mask_np_binary.astype(bool)] * 3, axis=-1)
    blended_rgb_np = np.where(mask_expanded_rgb, blurred_rgb_np, rgb_image_np)
    if has_alpha:
        return np.dstack((blended_rgb_np, alpha_channel_np))
    return blended_rgb_np

def _upload_return(editor, status, download, output, temp, stored, marks, preview, flag, uploaded, panel, editor_synced, taps):
    """Shared 13-value return for upload/clear, in the order of the upload event's outputs."""
    return (editor, status, download, output, temp, stored, marks, preview, flag, uploaded, panel, editor_synced, taps)

def _upload_error(message, current_temp_file_for_download, css_class="status-error"):
    """Upload/clear return that only shows a message: keeps the current image, marks and editor."""
    return _upload_return(
        gr.update(),
        gr.HTML(message, elem_classes=css_class),
        gr.DownloadButton(visible=False),
        None,
        current_temp_file_for_download,
        *([gr.skip()] * 8),
    )


# Gradio Event Handlers
def handle_file_upload(uploaded_file_path, current_temp_file_for_download, ui_mode=MODE_PHONE):
    """Handles new file uploads. Phone layout never loads ImageEditor (that reloads phones).
    Computer layout loads a capped copy into the drawing editor, matching the desktop flow."""
    computer = ui_mode == MODE_COMPUTER
    if uploaded_file_path:
        try:
            # Validate uploaded_file_path
            resolved_upload_path = os.path.realpath(uploaded_file_path)

            if not is_within_temp_dir(resolved_upload_path):
                logger.error(f"Security alert: Upload path '{uploaded_file_path}' resolves to '{resolved_upload_path}', which is outside the allowed temp directories. Aborting upload.")
                return _upload_error("Invalid file path detected. Upload failed.", current_temp_file_for_download)
            
            # Only decode PNG/JPEG regardless of file extension, and check size before decoding pixels
            with Image.open(resolved_upload_path, formats=ALLOWED_IMAGE_FORMATS) as opened_img:
                if opened_img.width * opened_img.height > MAX_IMAGE_PIXELS:
                    logger.warning(f"Rejected upload of {opened_img.width}x{opened_img.height} image (over pixel limit).")
                    return _upload_error(
                        f"Image is too large ({opened_img.width}x{opened_img.height}). "
                        f"Please resize it to under {MAX_IMAGE_PIXELS // 1_000_000} megapixels.",
                        current_temp_file_for_download,
                    )
                # Apply camera rotation so phone photos are not shown sideways
                img = ImageOps.exif_transpose(opened_img)
                img.load()
            stored_original = encode_stored_image(img)
            source = img  # Same pixels as the stored copy; no need to decode it again
            logger.info("Image loaded successfully.")
            
            # Clean up previous temporary file for download, if one exists
            remove_temp_file(current_temp_file_for_download)

            if computer:
                return _upload_return(
                    gr.update(value=editor_preview(source), visible=True),
                    gr.HTML(
                        "Image loaded successfully! You can now draw on it or get AI suggestions.",
                        elem_classes="status-success",
                    ),
                    gr.DownloadButton(visible=False),
                    None,
                    None,
                    stored_original,
                    None,
                    gr.update(value=None, visible=False),
                    False,
                    gr.update(value=None, visible=False),
                    gr.update(visible=True),
                    True,  # The editor now holds this upload
                    [],  # New image: clear phone taps
                )
            return _upload_return(
                gr.skip(),
                gr.HTML(
                    "Image loaded successfully. " + PHONE_DRAWING_LIMITATION,
                    elem_classes="status-success",
                ),
                gr.DownloadButton(visible=False),
                None,
                None,
                stored_original,
                None,
                gr.update(value=None, visible=False),
                False,
                gr.update(value=tap_view(source, []), visible=True),
                gr.update(visible=False),
                False,  # The hidden editor still holds any earlier image: ignore it
                [],  # New image: clear phone taps
            )
        except (UnidentifiedImageError, Image.DecompressionBombError) as e:
            logger.warning(f"Rejected uploaded file: {e}")
            return _upload_error(
                "Could not open this file. Please upload a valid PNG or JPG image under 40 megapixels.",
                current_temp_file_for_download,
            )
        except Exception as e:
            # Details are logged server-side only, so file paths are never shown to the user
            logger.error(f"Error processing uploaded file: {e}", exc_info=True)
            return _upload_error("Error loading image. Please try a different file.", current_temp_file_for_download)
    
    # Case: No file uploaded or file cleared - discard the result and any pending download.
    # The editor is left as-is: emptying it after drawing leaves the Gradio 6 editor stuck
    # loading, and the next upload replaces its contents anyway.
    remove_temp_file(current_temp_file_for_download)
    return _upload_error("No file provided or file cleared.", None, css_class="status-info")

def _blur_status_return(output_image, status_html, download_button, temp_id, preview_update=None, suggestions_flag=None, refreshed_state=None):
    """Shared return shape for Apply Blur (includes preview + client suggestion flag).
    refreshed_state = (original, marks, editor_synced, taps) re-saves those values so their
    30-minute expiry restarts while the session is in use."""
    original, marks, editor_synced, taps = refreshed_state if refreshed_state is not None else (gr.skip(),) * 4
    return (
        output_image,
        status_html,
        download_button,
        temp_id,
        gr.skip() if preview_update is None else preview_update,
        gr.skip() if suggestions_flag is None else suggestions_flag,
        original,
        marks,
        editor_synced,
        taps,
    )

def handle_blur_click(editor_data, original_image, saved_marks, suggestions_pending, current_temp_file_for_download, blur_strength_slider_value, editor_synced=None, tap_marks=None):
    """Applies blur to the marked areas (editor brush strokes, phone taps, and saved Privacy Suggestions marks)
    and prepares the result for download. Uses the validated original upload, never the
    editor background as a fallback. Refuses if suggestions were requested but the
    server-side marks have expired or been evicted."""
    try:
        blur_strength = float(blur_strength_slider_value)
    except (TypeError, ValueError):
        return _blur_status_return(
            None,
            gr.HTML(f"Blur strength must be a number from {MIN_BLUR_STRENGTH} to {MAX_BLUR_STRENGTH}.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download,
        )
    if not editor_matches_upload(editor_synced):
        editor_data = None  # Hidden editor holds an older image (Phone upload): use suggestions only
    source_pil, layers_pil, error = validate_editor_payload(editor_data, original_image)
    if error:
        return _blur_status_return(
            None,
            gr.HTML(error, elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download,
        )

    mask_shape = (source_pil.height, source_pil.width)
    suggestion_marks = unpack_marks(saved_marks, expected_shape=mask_shape)
    if suggestions_were_requested(suggestions_pending) and suggestion_marks is None:
        return _blur_status_return(
            None,
            gr.HTML(
                "Privacy Suggestions expired — please run them again.",
                elem_classes="status-error",
            ),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download,
            gr.update(value=None, visible=False),
            True,
        )

    background = editor_data.get("background") if editor_data else None
    if isinstance(background, Image.Image):
        final_mask_np_binary = layers_to_mask(layers_pil, background.size)
        final_mask_np_binary = scale_mask_to_image(final_mask_np_binary, source_pil.size)
    else:
        final_mask_np_binary = np.zeros(mask_shape, dtype=np.uint8)
    if suggestion_marks is not None:
        final_mask_np_binary |= suggestion_marks
    final_mask_np_binary |= taps_to_mask(tap_marks, source_pil.size)

    if np.sum(final_mask_np_binary) == 0:
        return _blur_status_return(
            None,
            gr.HTML("No areas marked for blurring. Draw or tap on the photo, or use Privacy Suggestions first.", elem_classes="status-info"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download,
        )

    remove_temp_file(current_temp_file_for_download)

    if source_pil.mode == "RGBA":
        source_np = np.array(source_pil)
        result_mode = "RGBA"
    else:
        source_np = np.array(source_pil.convert("RGB"))
        result_mode = "RGB"

    blurred_image_np = apply_gaussian_blur(source_np, final_mask_np_binary, blur_strength)
    blurred_image_pil = Image.fromarray(blurred_image_np, result_mode)

    new_temp_file_for_download_path = None
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="blurred_", dir=OUTPUT_DIR) as tmp_file:
            blurred_image_pil.save(tmp_file.name, "PNG")
            new_temp_file_for_download_path = tmp_file.name
        logger.info(f"Blurred image saved to temporary file for download: {new_temp_file_for_download_path}")

        return _blur_status_return(
            screen_preview(blurred_image_pil),
            gr.HTML(
                "Blur applied successfully! Download your privacy-protected image below. "
                "The file is a new PNG without the original photo's location or camera data.",
                elem_classes="status-success",
            ),
            gr.DownloadButton(value=new_temp_file_for_download_path, visible=True, label="Download Blurred Image"),
            register_output_file(new_temp_file_for_download_path),
            refreshed_state=(
                original_image,
                saved_marks,
                editor_synced if editor_synced is not None else gr.skip(),
                tap_marks if tap_marks is not None else gr.skip(),
            ),
        )
    except Exception as e:
        logger.error(f"Error saving blurred image to temp file: {e}", exc_info=True)
        if new_temp_file_for_download_path and os.path.exists(new_temp_file_for_download_path):
            try:
                os.remove(new_temp_file_for_download_path)
            except Exception as e_rem_fail:
                logger.error(f"Failed to remove temp file {new_temp_file_for_download_path} after error: {e_rem_fail}")
        return _blur_status_return(
            None,
            gr.HTML("Error processing blur. Please try again.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            None,
        )

def _privacy_detections(result):
    """Yields the indices of detections that are privacy targets with enough confidence."""
    if result.boxes is None:
        return
    for i, (class_id, confidence) in enumerate(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())):
        if result.names[int(class_id)] in PRIVACY_TARGET_CLASSES and confidence >= SUGGESTION_CONFIDENCE:
            yield i

def draw_object_boxes(draw, result):
    """Draws a filled rectangle over each detected privacy object. Returns how many were drawn."""
    count = 0
    for i in _privacy_detections(result):
        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], fill=SUGGESTION_FILL)
        count += 1
    return count

def draw_object_outlines(draw, result, image_size):
    """Fills the exact shape of each detected privacy object. Returns how many were drawn."""
    if result.masks is None:
        return 0
    # Grow each shape slightly so the blur also covers the object's edges
    edge_px = max(2, round(max(image_size) / 250))
    count = 0
    for i in _privacy_detections(result):
        polygon = [tuple(point) for point in result.masks.xy[i].tolist()]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=SUGGESTION_FILL, outline=SUGGESTION_FILL, width=edge_px)
            count += 1
    return count

def draw_faces(draw, result):
    """Draws an oval over each face, located from the pose model's face keypoints
    (nose, eyes, ears). Returns how many were drawn."""
    if result.keypoints is None or result.boxes is None:
        return 0
    count = 0
    for i, confidence in enumerate(result.boxes.conf.tolist()):
        if confidence < SUGGESTION_CONFIDENCE:
            continue
        keypoints = result.keypoints.data[i].tolist()
        face_points = [(x, y) for x, y, c in (keypoints[k] for k in FACE_KEYPOINTS) if c >= KEYPOINT_CONFIDENCE]
        if len(face_points) < 2:  # Face not visible (e.g. person facing away)
            continue
        xs, ys = [p[0] for p in face_points], [p[1] for p in face_points]
        center_x, center_y = sum(xs) / len(xs), sum(ys) / len(ys)
        # Estimate head size from the spread of face points, and from shoulder width when
        # visible (eyes alone sit close together, so they understate the head size)
        half_width = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.9
        (lsx, lsy, lsc), (rsx, rsy, rsc) = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
        if lsc >= KEYPOINT_CONFIDENCE and rsc >= KEYPOINT_CONFIDENCE:
            half_width = max(half_width, ((lsx - rsx) ** 2 + (lsy - rsy) ** 2) ** 0.5 * 0.34)
        half_width = max(half_width, 8)
        half_height = half_width * 1.4  # Heads are taller than wide; covers hair and chin
        draw.ellipse([center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height],
                     fill=SUGGESTION_FILL)
        count += 1
    return count

SUGGESTION_MESSAGES = {
    MODE_OBJECTS: ("Found {n} object(s)! Red boxes in the preview show detected people, vehicles, and other items that will be blurred.",
                   "No common privacy objects (people, cars, etc.) detected with high confidence. Try manual drawing instead."),
    MODE_OUTLINES: ("Found {n} object(s)! Red shapes in the preview outline detected people, vehicles, and other items that will be blurred.",
                    "No common privacy objects (people, cars, etc.) detected with high confidence. Try manual drawing instead."),
    MODE_FACES: ("Found {n} face(s)! Red ovals in the preview mark them for blurring. Very small or turned-away faces can be missed, so check and add any with the brush.",
                 "No faces detected. Faces that are very small, hidden, or turned away can be missed. Try manual drawing instead."),
}

def handle_suggest_click(editor_data, original_image, detection_mode=MODE_OBJECTS, editor_synced=None):
    """Uses YOLO26 to detect privacy targets (whole objects, exact outlines, or faces).

    Each run replaces the previous suggestions. The detections are kept server-side as a mask and
    shown in a separate preview image; Apply Blur combines them with the brush strokes in the
    editor. The editor itself is never updated here: pushing new content into the Gradio 6 editor
    while it is still syncing recent brush strokes can leave it stuck loading.

    Returns (status, download button, marks, preview image update, suggestions-pending flag,
    original image). The original is re-saved on success so its 30-minute expiry restarts."""
    model = get_yolo_model(detection_mode)
    if model is None:
        return (
            gr.HTML("Privacy Suggestions unavailable: the model for this detection mode is not loaded.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip()
        )

    if not editor_matches_upload(editor_synced):
        editor_data = None  # Hidden editor holds an older image (Phone upload)
    base_pil, _layers, error = validate_editor_payload(editor_data, original_image)
    if error:
        return (
            gr.HTML(error, elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip()
        )

    try:
        # YOLO typically works best with RGB images
        result = model(base_pil.convert("RGB"), conf=SUGGESTION_CONFIDENCE, verbose=False)[0]

        # Draw the detections on a transparent layer, then turn it into a mask
        suggestion_layer_pil = Image.new("RGBA", base_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(suggestion_layer_pil)

        if detection_mode == MODE_OUTLINES:
            found_count = draw_object_outlines(draw, result, base_pil.size)
        elif detection_mode == MODE_FACES:
            found_count = draw_faces(draw, result)
        else:
            found_count = draw_object_boxes(draw, result)

        found_msg, none_msg = SUGGESTION_MESSAGES.get(detection_mode, SUGGESTION_MESSAGES[MODE_OBJECTS])
        if found_count == 0:
            # Nothing found: previous suggestions are replaced too, so they won't be blurred
            return (
                gr.HTML(none_msg, elem_classes="status-info"),
                gr.DownloadButton(visible=False),
                None,
                gr.update(value=None, visible=False),
                False,
                original_image,
            )

        # New suggestions replace the previous ones (brush strokes stay in the editor)
        mask = layers_to_mask([suggestion_layer_pil], base_pil.size)

        return (
            gr.HTML(found_msg.format(n=found_count), elem_classes="status-success"),
            gr.DownloadButton(visible=False),
            pack_marks(mask),
            gr.update(value=screen_preview(render_marks_preview(base_pil, mask)), visible=True),
            True,
            original_image,
        )

    except Exception as e:
        logger.error(f"Error during AI suggestion generation: {e}", exc_info=True)
        return (
            gr.HTML("Error with Privacy Suggestions. Please try again or mark areas manually.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip()
        )

def handle_clear_marks():
    """Removes all saved Privacy Suggestions areas (brush strokes are managed in the editor itself)."""
    return (
        gr.update(value=None, visible=False),
        gr.HTML("Privacy Suggestions removed - those red areas will no longer be blurred. Your brush strokes are kept; use the editor's eraser or undo to remove them.", elem_classes="status-info"),
        None,
        False
    )

def _tap_result(source, taps, message, css_class="status-info"):
    """(phone preview update, tap marks, status) for the tap handlers."""
    return gr.update(value=tap_view(source, taps)), taps, gr.HTML(message, elem_classes=css_class)


def _tap_count_message(taps):
    if not taps:
        return "No tapped areas. Tap the photo to mark areas to blur."
    return f"{len(taps)} tapped area(s) marked in red. Tap more, or press Apply Blur."


def handle_phone_tap(original_image, tap_marks, tap_size, evt: gr.SelectData):
    """Phone layout: adds a round mark where the user tapped the photo."""
    source = decode_stored_image(original_image)
    if source is None:
        return gr.skip(), gr.skip(), gr.HTML("Session expired — please upload the image again.", elem_classes="status-error")
    try:
        tap_x, tap_y = float(evt.index[0]), float(evt.index[1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return gr.skip(), gr.skip(), gr.HTML("Could not read where you tapped. Please try again.", elem_classes="status-error")
    taps = normalize_tap_marks(tap_marks)
    if len(taps) >= MAX_TAP_MARKS:
        return gr.skip(), gr.skip(), gr.HTML(
            f"You can mark up to {MAX_TAP_MARKS} areas. Use a bigger Tap Mark Size or Undo Last Tap.",
            elem_classes="status-error",
        )
    # The tap is in screen-preview pixels; map it back to the original photo
    preview_width, preview_height = screen_size(source.size)
    x = min(max(tap_x * source.width / preview_width, 0), source.width - 1)
    y = min(max(tap_y * source.height / preview_height, 0), source.height - 1)
    try:
        size_percent = min(max(float(tap_size), MIN_TAP_SIZE), MAX_TAP_SIZE)
    except (TypeError, ValueError):
        size_percent = DEFAULT_TAP_SIZE
    radius = size_percent / 100.0 * min(source.width, source.height) / 2.0
    taps.append((x, y, radius))
    return _tap_result(source, taps, _tap_count_message(taps))


def handle_undo_tap(original_image, tap_marks):
    """Phone layout: removes the most recent tap."""
    source = decode_stored_image(original_image)
    if source is None:
        return gr.skip(), gr.skip(), gr.HTML("Session expired — please upload the image again.", elem_classes="status-error")
    taps = normalize_tap_marks(tap_marks)[:-1]
    return _tap_result(source, taps, _tap_count_message(taps))


def handle_clear_taps(original_image):
    """Phone layout: removes every tap."""
    source = decode_stored_image(original_image)
    if source is None:
        return gr.skip(), [], gr.HTML("Taps cleared.", elem_classes="status-info")
    return _tap_result(source, [], "Taps cleared. Tap the photo to mark areas to blur.")


def handle_layout_change(ui_mode, original_image, tap_marks=None):
    """Shows Computer drawing editor or Phone tap preview. Does not empty ImageEditor (that sticks Gradio 6).
    Returns (editor panel, editor, phone preview, status, editor-in-sync flag, phone tools)."""
    source = decode_stored_image(original_image)
    if ui_mode == MODE_COMPUTER:
        editor = (
            gr.update(value=editor_preview(source), visible=True)
            if source is not None
            else gr.update(visible=True)
        )
        return (
            gr.update(visible=True),
            editor,
            gr.update(visible=False),
            gr.HTML(
                "Computer layout. Draw with the brush or use Privacy Suggestions, then Apply Blur.",
                elem_classes="status-info",
            ),
            True if source is not None else gr.skip(),  # Editor reloaded with the current upload
            gr.update(visible=False),
        )
    uploaded = (
        gr.update(value=tap_view(source, tap_marks), visible=True)
        if source is not None
        else gr.update(visible=False)
    )
    return (
        gr.update(visible=False),
        gr.skip(),
        uploaded,
        gr.HTML(PHONE_DRAWING_LIMITATION, elem_classes="status-info"),
        gr.skip(),  # Editor keeps the current upload's strokes
        gr.update(visible=True),
    )


def detect_device_layout(request: gr.Request):
    """Picks Phone or Computer from the browser user-agent on first load."""
    user_agent = ""
    if request is not None:
        user_agent = request.headers.get("user-agent") or request.headers.get("User-Agent") or ""
    if is_phone_user_agent(user_agent):
        return (
            MODE_PHONE,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.HTML(PHONE_DRAWING_LIMITATION, elem_classes="status-info"),
            gr.update(visible=True),
        )
    return (
        MODE_COMPUTER,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.HTML("Ready. Upload an image to begin.", elem_classes="status-info status-bar"),
        gr.update(visible=False),
    )

# Custom CSS for a cool looking dark theme
css = """
body, html {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #e2e8f0 !important;
    background-color: #1a202c !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    padding: 20px;
    background-color: #1a202c !important;
}
/* Dark theme text colors - excellent contrast */
p, span, div {
    color: #e2e8f0 !important;
}
/* Input components - high contrast for readability */
input {
    color: #ffffff !important;
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    border-radius: 6px !important;
}
/* Focus states for inputs */
input:focus {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1) !important;
    outline: none !important;
}
/* Placeholder text */
::placeholder {
    color: #a0aec0 !important;
    opacity: 0.8;
}
/* Headers */
h1 {
    color: #f7fafc !important;
    font-weight: 600;
}
/* Status message styling */
.status-success, .status-error, .status-info, .status-bar {
    min-height: 22px;
    padding: 8px 16px !important;
    margin: 22px 0 12px 0 !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 1.02em !important;
    opacity: 0.98;
    text-align: center;
}
/* Section headers - complementary blue gradient */
.section-header {
    background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
    color: #ffffff !important;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 15px 0 10px 0;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
/* Privacy notice styling - dark theme */
.privacy-notice {
    padding: 16px;
    margin-top: 20px;
    border: 1px solid #4a5568;
    border-radius: 8px;
    background-color: #2d3748;
    color: #cbd5e0 !important;
    font-size: 0.9em;
    line-height: 1.4;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}
img {
    max-width: 100%;
    height: auto;
}
#drawing-editor-panel.hidden,
#drawing-editor-panel[hidden] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
#drawing-editor-panel canvas {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
    border-radius: 8px;
    background-color: #2d3748 !important;
}
/* Hide default Gradio footer */
footer {
    display: none !important;
}
/* Accordion styling - dark theme */
.gr-accordion {
    border-radius: 8px !important;
    border: 1px solid #4a5568 !important;
    background-color: #2d3748 !important;
}
/* Label improvements */
label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
}
"""

# The CSS above makes text light, so Gradio has to draw its panels dark too. Without this, a browser in
# light mode draws light panels and labels, radio choices and help text become light on light.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

# Build the Gradio interface using Blocks for layout flexibility.
def on_hugging_face_space():
    """True when this process is a Hugging Face Space. Spaces sets SPACE_ID; a local run does not."""
    return bool(os.environ.get("SPACE_ID"))

def privacy_notice_html(on_space=None):
    """Footer copy for where the photo is processed. The Space and a local run are different."""
    if on_space is None:
        on_space = on_hugging_face_space()
    if on_space:
        body = (
            "<strong>Privacy First:</strong> This page runs in your browser. "
            "The photo is uploaded to this Hugging Face Space, and blur and suggestions run on that server. "
            "Hugging Face has the photo because it is running the app. It is not sent to any other service. "
            "The download is a new PNG without the original photo's location or camera data. "
            "For a photo that should never leave your computer, run Blur Tool locally."
        )
    else:
        body = (
            "<strong>Privacy First:</strong> Images are processed on this computer and are not sent to another service. "
            "Downloads are new PNG files without the original photo's location or camera data. "
            "The AI models download once from Ultralytics if needed."
        )
    return "<div class='privacy-notice'>" + body + "</div>"

def processing_location_line(on_space=None):
    """One How to Use line, matching privacy_notice_html."""
    if on_space is None:
        on_space = on_hugging_face_space()
    if on_space:
        return (
            "The page is in your browser. The photo is uploaded to this Hugging Face Space, where the blur runs. "
            "To keep a photo on your own computer, run Blur Tool locally."
        )
    return "Images are processed on this computer and are not sent to another service."


def how_to_use_markdown(on_space=None):
    """How to Use accordion copy, including the phone drawing limitation."""
    return f"""
        **Quick Start Guide:**
        
        1. **Layout**: Phone marks areas by tapping the photo. Computer has the freehand drawing editor.
        2. **Upload**: Click 'Upload Image' or drag & drop your JPG/PNG file
        3. **Mark Areas**:
           - **Phone**: {PHONE_DRAWING_LIMITATION}
           - **Computer**: Draw with the brush, or pick what to detect and click 'Privacy Suggestions'
           - **Both**: Tapped areas, brush strokes, and Privacy Suggestions are all blurred together
        4. **Apply**: Adjust blur strength and click 'Apply Blur'
        5. **Download**: Save your processed image
        
        **Pro Tips:**
        - Use red brush for clear visibility on most images
        - **Whole objects** marks people, vehicles, electronics and bags with boxes
        - **Exact outlines** traces the shape of those objects, so less of the background gets blurred
        - **Faces only** marks just people's faces - very small or turned-away faces can be missed
        - License plates and text are not detected - mark those with the brush
        - Suggestions appear in red in a preview below the editor; running them again (e.g. after switching mode) replaces them
        - 'Remove Privacy Suggestions' deletes the suggested areas so they won't be blurred (your brush strokes stay)
        - Higher blur values create stronger effects (the blur scales with image size, so large photos stay unreadable)
        - If you wait a long time after Privacy Suggestions, Apply Blur will ask you to run them again rather than silently skip those areas
        - {processing_location_line(on_space)}
        """

# delete_cache=(3600, 3600) deletes Gradio's cached uploads about an hour after
# creation while this process is running, and again on a clean shutdown.
# analytics_enabled=False stops Gradio from sending usage statistics.
with gr.Blocks(title="Blur Tool", delete_cache=(3600, 3600), analytics_enabled=False) as demo:
    gr.Markdown("# Blur Tool")
    gr.Markdown("*Draw to blur images then Download them. Powered by Gradio, OpenCV, and YOLO26.*")
    
    # Browser compatibility warning
    gr.HTML("""
    <div style="background: #2d4a22; border: 1px solid #38a169; color: #68d391; padding: 12px 16px; margin: 15px 0; border-radius: 8px; font-weight: 500;">
        <strong> Firefox Users:</strong> If the image editor doesn't work, try enabling hardware acceleration in Firefox Settings → Performance, 
        or use Google Chrome for guaranteed compatibility.
    </div>
    """)
    
    # Collapsible instructions
    with gr.Accordion("How to Use", open=False):
        gr.Markdown(how_to_use_markdown())
    
    # Status messages for user feedback (dynamic, including ready state)
    status_html = gr.HTML("Ready. Upload an image to begin.", elem_classes="status-info status-bar")

    # ID of the temporary blurred image for the download button (never a file path).
    # The file is deleted when the session data expires. The hourly sweep runs only while this process is up.
    temp_file_path_for_download_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS, delete_callback=remove_temp_file)
    # Compact PNG bytes of the uploaded image (kept server-side; re-saved on each blur or
    # suggestion so the 30-minute expiry counts from the last use, not the upload).
    original_image_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)
    # Packed bitmask from the latest Privacy Suggestions run (brush strokes stay in the editor).
    marks_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)
    # Lives in the browser, so it still tells the server suggestions were requested after
    # server-side marks expire or the session is evicted.
    suggestions_pending = gr.Checkbox(value=False, visible=False, interactive=False, label="")
    # Whether the drawing editor holds the current upload (False after a Phone-layout upload,
    # when the hidden editor may still hold an older image and its brush strokes).
    editor_synced_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)
    # Phone layout taps: list of (x, y, radius) circles in original-image pixels.
    tap_marks_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)

    layout_radio = gr.Radio(
        choices=[MODE_PHONE, MODE_COMPUTER],
        value=MODE_PHONE,
        label="Layout",
    )

    with gr.Row():
        with gr.Column(scale=3): # Main interactive column
            gr.HTML("<div class='section-header'>Upload & Edit</div>")
            
            file_uploader = gr.File(
                label="Upload Image (PNG, JPG)", 
                type="filepath", 
                file_types=[".png", ".jpg", ".jpeg"],
                file_count="single"
            )

            uploaded_view = gr.Image(
                label="Uploaded Image (tap to mark areas to blur)",
                interactive=False,
                type="pil",
                format="jpeg",
                height=420,
                visible=False,
            )

            with gr.Column(visible=False) as phone_tools:
                tap_size_slider = gr.Slider(
                    minimum=MIN_TAP_SIZE,
                    maximum=MAX_TAP_SIZE,
                    value=DEFAULT_TAP_SIZE,
                    step=1,
                    label="Tap Mark Size (% of photo)",
                )
                with gr.Row():
                    undo_tap_button = gr.Button("Undo Last Tap", size="sm", variant="secondary")
                    clear_taps_button = gr.Button("Clear Taps", size="sm", variant="secondary")

            with gr.Column(visible=False, elem_id="drawing-editor-panel") as editor_panel:
                image_editor = gr.ImageEditor(
                    label="Image Editor",
                    type="pil",
                    sources=[],
                    interactive=True,
                    brush=gr.Brush(
                        default_size=25,
                        colors=["#FF0000", "#0066CC"],
                        color_mode="fixed"
                    ),
                    eraser=gr.Eraser(default_size=25),
                    canvas_size=(800, 600),
                )

            marks_preview = gr.Image(
                label="Privacy Suggestions Preview (red areas will be blurred along with your brush strokes)",
                interactive=False,
                type="pil",
                height=420,
                visible=False
            )
            
        with gr.Column(scale=2): # Actions and results column
            gr.HTML("<div class='section-header'>Controls</div>")
            
            blur_strength_slider = gr.Slider(
                minimum=MIN_BLUR_STRENGTH,
                maximum=MAX_BLUR_STRENGTH,
                value=DEFAULT_BLUR_STRENGTH,
                step=5,
                label="Blur Strength"
            )

            detection_mode_radio = gr.Radio(
                choices=list(DETECTION_MODELS),
                value=MODE_OBJECTS,
                label="Privacy Suggestions Detect"
            )

            with gr.Row():
                suggest_button = gr.Button("Privacy Suggestions", size="sm", variant="primary")
                blur_button = gr.Button("Apply Blur", variant="primary", size="sm")
            clear_marks_button = gr.Button("Remove Privacy Suggestions", size="sm", variant="secondary")
            gr.Markdown(
                "<small>Each Privacy Suggestions run replaces the previous one, so switch the mode and run it again "
                "to compare. **Remove Privacy Suggestions** deletes the red areas they found (shown in the preview), "
                "so they won't be blurred. It does not touch what you drew with the brush - "
                "use the editor's eraser or undo for that.</small>"
            )

            gr.HTML("<div class='section-header'>Results</div>")
            
            output_image = gr.Image(
                label="Processed Image",
                interactive=False,
                type="pil",
                format="jpeg",
                height=420,
            )
            
            download_button = gr.DownloadButton("Download Blurred Image", visible=False, size="lg")
        
    gr.Markdown(privacy_notice_html())
    
    demo.load(
        fn=detect_device_layout,
        inputs=None,
        outputs=[layout_radio, editor_panel, uploaded_view, status_html, phone_tools],
    )
    layout_radio.change(
        fn=handle_layout_change,
        inputs=[layout_radio, original_image_state, tap_marks_state],
        outputs=[editor_panel, image_editor, uploaded_view, status_html, editor_synced_state, phone_tools],
    )
    # Phone layout: tap the photo to mark areas
    uploaded_view.select(
        fn=handle_phone_tap,
        inputs=[original_image_state, tap_marks_state, tap_size_slider],
        outputs=[uploaded_view, tap_marks_state, status_html],
    )
    undo_tap_button.click(
        fn=handle_undo_tap,
        inputs=[original_image_state, tap_marks_state],
        outputs=[uploaded_view, tap_marks_state, status_html],
    )
    clear_taps_button.click(
        fn=handle_clear_taps,
        inputs=[original_image_state],
        outputs=[uploaded_view, tap_marks_state, status_html],
    )
    # File uploader actions
    file_uploader.upload(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state, layout_radio],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state, marks_state, marks_preview, suggestions_pending, uploaded_view, editor_panel, editor_synced_state, tap_marks_state]
    )
    file_uploader.clear(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state, layout_radio],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state, marks_state, marks_preview, suggestions_pending, uploaded_view, editor_panel, editor_synced_state, tap_marks_state]
    )
    
    # Blur button actions
    blur_button.click(
        fn=handle_blur_click,
        inputs=[image_editor, original_image_state, marks_state, suggestions_pending, temp_file_path_for_download_state, blur_strength_slider, editor_synced_state, tap_marks_state],
        outputs=[output_image, status_html, download_button, temp_file_path_for_download_state, marks_preview, suggestions_pending, original_image_state, marks_state, editor_synced_state, tap_marks_state]
    )
    
    # Suggest button actions
    suggest_button.click(
        fn=handle_suggest_click,
        inputs=[image_editor, original_image_state, detection_mode_radio, editor_synced_state],
        outputs=[status_html, download_button, marks_state, marks_preview, suggestions_pending, original_image_state]
    )

    # Clear marks button actions
    clear_marks_button.click(
        fn=handle_clear_marks,
        inputs=[],
        outputs=[marks_preview, status_html, marks_state, suggestions_pending]
    )

# Limit how many requests can wait at once, so a flood is turned away instead of piling up
demo.queue(max_size=MAX_QUEUED_REQUESTS)

# Start cleanup and inject launch limits even when this file is imported
# (`gradio app.py`, Hugging Face Spaces) rather than run as `__main__`.
_ensure_output_sweeper()
_original_demo_launch = demo.launch

def launch_with_limits(*args, **kwargs):
    kwargs.setdefault("max_file_size", MAX_UPLOAD_FILE_SIZE)
    kwargs.setdefault("state_session_capacity", MAX_STORED_SESSIONS)
    kwargs.setdefault("theme", "base")
    kwargs.setdefault("css", css)
    kwargs.setdefault("js", FORCE_DARK_JS)
    _ensure_output_sweeper()
    return _original_demo_launch(*args, **kwargs)

demo.launch = launch_with_limits

# When main is run, start the application
if __name__ == "__main__":
    logger.info("Starting Gradio Blur Tool app...")
    # Runs locally only by default. Set the environment variable GRADIO_SHARE=True
    # to also create a temporary public Gradio link.
    demo.launch()