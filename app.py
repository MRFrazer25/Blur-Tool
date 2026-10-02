import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import os
import tempfile
import logging
import secrets
import threading
import time

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

# Load the models when the application starts. Each downloads from Ultralytics on first run
# (requires internet once). If one fails, only that mode is unavailable.
yolo_models = {}
for mode, weights in DETECTION_MODELS.items():
    try:
        logger.info(f"Attempting to load {weights}...")
        yolo_models[mode] = YOLO(weights)
        logger.info(f"{weights} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load {weights}: {e}. The '{mode}' suggestion mode will be unavailable.", exc_info=True)

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

# Blur strength range (must match the slider)
MIN_BLUR_STRENGTH = 1
MAX_BLUR_STRENGTH = 101

# Resource limits for a public deployment
MAX_QUEUED_REQUESTS = 20         # Further requests get a "busy" error instead of piling up
MAX_STORED_SESSIONS = 100        # Least recently used sessions beyond this are dropped from memory
SESSION_DATA_TTL_SECONDS = 1800  # Per-session images/marks expire after 30 minutes without updates

# Blurred results are written to the app's own folder; files older than an hour are swept
# regularly, which also catches files left behind by crashes, restarts or dropped sessions.
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "blur-tool-outputs")
OUTPUT_MAX_AGE_SECONDS = 3600
OUTPUT_SWEEP_INTERVAL_SECONDS = 600

# Helper Functions
def is_within_temp_dir(path):
    """Returns True if path resolves to a location inside the system temp directory
    or Gradio's upload folder (which can be moved with the GRADIO_TEMP_DIR variable)."""
    resolved_path = os.path.realpath(path)
    allowed_dirs = [tempfile.gettempdir(), os.environ.get("GRADIO_TEMP_DIR")]
    for allowed_dir in filter(None, allowed_dirs):
        allowed_dir = os.path.realpath(allowed_dir)
        try:
            if os.path.commonpath([resolved_path, allowed_dir]) == allowed_dir:
                return True
        except ValueError:  # Paths on different drives (Windows)
            continue
    return False

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

# Core Functions
def layers_to_mask(layers, size):
    """Combines ImageEditor drawing layers into a binary mask (1 = marked) of the given (width, height).
    Any pixel with non-zero alpha on any layer counts as marked."""
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in layers or []:
        if isinstance(layer, Image.Image):
            layer_alpha = layer.convert("RGBA").split()[-1]
            if layer_alpha.size != size:
                layer_alpha = layer_alpha.resize(size, Image.NEAREST)
            mask |= (np.array(layer_alpha) > 0).astype(np.uint8)
    return mask

def render_marks_preview(base_image, marks):
    """Returns the image with marked areas tinted semi-transparent red, for the Privacy Suggestions preview."""
    overlay = np.zeros((base_image.height, base_image.width, 4), dtype=np.uint8)
    overlay[marks.astype(bool)] = SUGGESTION_FILL
    return Image.alpha_composite(base_image.convert("RGBA"), Image.fromarray(overlay, "RGBA"))

def apply_gaussian_blur(image_np_rgba, mask_np_binary, blur_radius_odd):
    """
    Applies Gaussian blur to an RGBA image in regions specified by a binary mask.
    Uses the original alpha channel for the final image.
    """
    # Input validation for security and stability
    if image_np_rgba.shape[:2] != mask_np_binary.shape:
        logger.error("Image and mask dimensions mismatch. Cannot apply blur.")
        return image_np_rgba # Return original image on dimension error

    # Ensure blur_radius is odd and within range for cv2.GaussianBlur kernel size
    k_val = min(max(MIN_BLUR_STRENGTH, int(blur_radius_odd)), MAX_BLUR_STRENGTH)
    if k_val % 2 == 0:
        k_val += 1
    ksize = (k_val, k_val)

    # Separate RGB and alpha channels for processing
    rgb_image_np = image_np_rgba[:, :, :3]
    alpha_channel_np = image_np_rgba[:, :, 3] # Preserve original transparency

    # Apply Gaussian blur to RGB channels
    blurred_rgb_np = cv2.GaussianBlur(rgb_image_np, ksize, 0)

    # Expand mask to 3 channels for RGB blending
    mask_expanded_rgb = np.stack([mask_np_binary] * 3, axis=-1)

    # Blend original and blurred RGB based on mask
    blended_rgb_np = np.where(mask_expanded_rgb == 1, blurred_rgb_np, rgb_image_np)
    
    # Recombine with original alpha channel
    final_rgba_np = np.dstack((blended_rgb_np, alpha_channel_np))
    
    return final_rgba_np

# Gradio Event Handlers
def handle_file_upload(uploaded_file_path, current_temp_file_for_download):
    """Handles new file uploads, prepares image for editor, and cleans up old temp files.
    Also returns the full-quality original, which is what gets blurred (the editor only
    holds a compressed preview copy), and resets any saved Privacy Suggestions marks."""
    if uploaded_file_path:
        try:
            # Validate uploaded_file_path
            resolved_upload_path = os.path.realpath(uploaded_file_path)

            if not is_within_temp_dir(resolved_upload_path):
                logger.error(f"Security alert: Upload path '{uploaded_file_path}' resolves to '{resolved_upload_path}', which is outside the allowed temp directories. Aborting upload.")
                return (
                    gr.update(),
                    gr.HTML("Invalid file path detected. Upload failed.", elem_classes="status-error"),
                    gr.DownloadButton(visible=False),
                    None,
                    current_temp_file_for_download,
                    gr.skip(),
                    gr.skip(),
                    gr.skip()
                )
            
            # Only decode PNG/JPEG regardless of file extension, and check size before decoding pixels
            with Image.open(resolved_upload_path, formats=ALLOWED_IMAGE_FORMATS) as opened_img:
                if opened_img.width * opened_img.height > MAX_IMAGE_PIXELS:
                    logger.warning(f"Rejected upload of {opened_img.width}x{opened_img.height} image (over pixel limit).")
                    return (
                        gr.update(),
                        gr.HTML(f"Image is too large ({opened_img.width}x{opened_img.height}). Please resize it to under {MAX_IMAGE_PIXELS // 1_000_000} megapixels.", elem_classes="status-error"),
                        gr.DownloadButton(visible=False),
                        None,
                        current_temp_file_for_download,
                        gr.skip(),
                        gr.skip(),
                        gr.skip()
                    )
                # Apply camera rotation so phone photos are not shown sideways
                img = ImageOps.exif_transpose(opened_img).convert("RGBA")
            logger.info("Image loaded successfully.")
            
            # Clean up previous temporary file for download, if one exists
            remove_temp_file(current_temp_file_for_download)

            return (
                gr.update(value=img), 
                gr.HTML("Image loaded successfully! You can now draw on it or get AI suggestions.", elem_classes="status-success"),
                gr.DownloadButton(visible=False), 
                None, 
                None,
                img,
                None,  # New image: clear any saved marks
                gr.update(value=None, visible=False)
            )
        except (UnidentifiedImageError, Image.DecompressionBombError) as e:
            logger.warning(f"Rejected uploaded file: {e}")
            return (
                gr.update(),
                gr.HTML("Could not open this file. Please upload a valid PNG or JPG image under 40 megapixels.", elem_classes="status-error"),
                gr.DownloadButton(visible=False),
                None,
                current_temp_file_for_download,
                gr.skip(),
                gr.skip(),
                gr.skip()
            )
        except Exception as e:
            # Details are logged server-side only, so file paths are never shown to the user
            logger.error(f"Error processing uploaded file: {e}", exc_info=True)
            return (
                gr.update(),
                gr.HTML("Error loading image. Please try a different file.", elem_classes="status-error"),
                gr.DownloadButton(visible=False),
                None,
                current_temp_file_for_download,
                gr.skip(),
                gr.skip(),
                gr.skip()
            )
    
    # Case: No file uploaded or file cleared - discard the result and any pending download.
    # The editor is left as-is: emptying it after drawing leaves the Gradio 6 editor stuck
    # loading, and the next upload replaces its contents anyway.
    remove_temp_file(current_temp_file_for_download)
    return (
        gr.update(),
        gr.HTML("No file provided or file cleared.", elem_classes="status-info"),
        gr.DownloadButton(visible=False),
        None,
        None,
        gr.skip(),  # Keep the original, marks and preview in sync with the editor, which is also left as-is
        gr.skip(),
        gr.skip()
    )

def handle_blur_click(editor_data, original_image, saved_marks, current_temp_file_for_download, blur_strength_slider_value):
    """Applies blur to the marked areas (editor brush strokes plus saved Privacy Suggestions marks)
    and prepares the result for download. Blurs the full-quality original upload when available,
    since the editor only holds a compressed copy."""
    if not editor_data or not editor_data.get('background'):
        return (
            None, 
            gr.HTML("Please upload and select an image first.", elem_classes="status-error"), 
            gr.DownloadButton(visible=False),
            current_temp_file_for_download
        )

    background_pil = editor_data['background'] # ImageEditor's 'type' is 'pil'
    layers_pil = editor_data.get('layers', []) # Drawings are in 'layers'
    
    if not isinstance(background_pil, Image.Image):
        logger.error("Background is not a PIL image. This indicates an issue with ImageEditor's output type.")
        return None, gr.HTML("Internal error: Background image format incorrect.", elem_classes="status-error"), gr.DownloadButton(visible=False), current_temp_file_for_download

    # The editor's copy is compressed, so blur the original upload instead
    if isinstance(original_image, Image.Image) and original_image.size == background_pil.size:
        source_pil = original_image
    else:
        source_pil = background_pil
    background_np_rgba = np.array(source_pil.convert("RGBA"))

    # Marked areas = brush strokes in the editor + saved marks from Privacy Suggestions
    final_mask_np_binary = layers_to_mask(layers_pil, source_pil.size)
    if isinstance(saved_marks, np.ndarray) and saved_marks.shape == final_mask_np_binary.shape:
        final_mask_np_binary |= saved_marks

    if np.sum(final_mask_np_binary) == 0: # Check if mask is empty
         return (
            None,
            gr.HTML("No areas marked for blurring. Use the brush tools or AI suggestions first.", elem_classes="status-info"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download
        )

    # Clean up previous temporary file for download, validating its path first
    remove_temp_file(current_temp_file_for_download)

    blur_radius = int(blur_strength_slider_value)
    blurred_image_np = apply_gaussian_blur(background_np_rgba, final_mask_np_binary, blur_radius)
    blurred_image_pil = Image.fromarray(blurred_image_np, 'RGBA')
    
    new_temp_file_for_download_path = None
    try:
        # Save blurred image to a new temporary file for download
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="blurred_", dir=OUTPUT_DIR) as tmp_file:
            blurred_image_pil.save(tmp_file.name, "PNG")
            new_temp_file_for_download_path = tmp_file.name
        logger.info(f"Blurred image saved to temporary file for download: {new_temp_file_for_download_path}")
        
        return (
            blurred_image_pil, # Display in output_image component
            gr.HTML("Blur applied successfully! Download your privacy-protected image below.", elem_classes="status-success"),
            gr.DownloadButton(value=new_temp_file_for_download_path, visible=True, label="Download Blurred Image"),
            register_output_file(new_temp_file_for_download_path) # State holds only the file's ID
        )
    except Exception as e:
        logger.error(f"Error saving blurred image to temp file: {e}", exc_info=True)
        # Attempt to clean up if temp file was created but save failed
        if new_temp_file_for_download_path and os.path.exists(new_temp_file_for_download_path):
            try: os.remove(new_temp_file_for_download_path)
            except Exception as e_rem_fail: logger.error(f"Failed to remove temp file {new_temp_file_for_download_path} after error: {e_rem_fail}")
        return (
            None, 
            gr.HTML("Error processing blur. Please try again.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            None # Reset temp file state on error
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

def handle_suggest_click(editor_data, original_image, detection_mode=MODE_OBJECTS):
    """Uses YOLO26 to detect privacy targets (whole objects, exact outlines, or faces).

    Each run replaces the previous suggestions. The detections are kept server-side as a mask and
    shown in a separate preview image; Apply Blur combines them with the brush strokes in the
    editor. The editor itself is never updated here: pushing new content into the Gradio 6 editor
    while it is still syncing recent brush strokes can leave it stuck loading.

    Returns (status, download button, marks, preview image update)."""
    model = yolo_models.get(detection_mode)
    if model is None:
        return (
            gr.HTML("Privacy Suggestions unavailable: the model for this detection mode is not loaded.", elem_classes="status-error"),
            gr.DownloadButton(visible=False), # Ensure download button is hidden
            gr.skip(),
            gr.skip()
        )

    if not editor_data or not editor_data.get('background'):
        return (
            gr.HTML("Please upload an image first to use Privacy Suggestions.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            gr.skip(),
            gr.skip()
        )

    background_pil = editor_data['background']
    if not isinstance(background_pil, Image.Image):
        logger.error("Background for suggestion is not a PIL image.")
        return gr.HTML("Internal error: Image format incorrect for Privacy Suggestions.", elem_classes="status-error"), gr.DownloadButton(visible=False), gr.skip(), gr.skip()

    # Detect on the full-quality original (the editor's copy is compressed)
    if isinstance(original_image, Image.Image) and original_image.size == background_pil.size:
        base_pil = original_image
    else:
        base_pil = background_pil

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
                gr.update(value=None, visible=False)
            )

        # New suggestions replace the previous ones (brush strokes stay in the editor)
        marks = layers_to_mask([suggestion_layer_pil], base_pil.size)

        return (
            gr.HTML(found_msg.format(n=found_count), elem_classes="status-success"),
            gr.DownloadButton(visible=False),
            marks,
            gr.update(value=render_marks_preview(base_pil, marks), visible=True)
        )

    except Exception as e:
        logger.error(f"Error during AI suggestion generation: {e}", exc_info=True)
        return (
            gr.HTML("Error with Privacy Suggestions. Please try again or mark areas manually.", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            gr.skip(),
            gr.skip()
        )

def handle_clear_marks():
    """Removes all saved Privacy Suggestions areas (brush strokes are managed in the editor itself)."""
    return (
        gr.update(value=None, visible=False),
        gr.HTML("Privacy Suggestions removed - those red areas will no longer be blurred. Your brush strokes are kept; use the editor's eraser or undo to remove them.", elem_classes="status-info"),
        None
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
canvas {
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

# Build the Gradio interface using Blocks for layout flexibility.
# delete_cache removes uploaded/processed images from Gradio's cache after an hour.
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
        gr.Markdown("""
        **Quick Start Guide:**
        
        1. **Upload**: Click 'Upload Image' or drag & drop your JPG/PNG file
        2. **Mark Areas**: Choose your method:
           - **Manual**: Use brush tools to draw on areas you want blurred
           - **AI Assist**: Pick what to detect, then click 'Privacy Suggestions' - uses YOLO26 AI to find privacy-sensitive areas automatically
        3. **Apply**: Adjust blur strength and click 'Apply Blur'
        4. **Download**: Save your processed image
        
        **Pro Tips:**
        - Use red brush for clear visibility on most images
        - **Whole objects** marks people, vehicles, electronics and bags with boxes
        - **Exact outlines** traces the shape of those objects, so less of the background gets blurred
        - **Faces only** marks just people's faces - very small or turned-away faces can be missed
        - License plates and text are not detected - mark those with the brush
        - Suggestions appear in red in a preview below the editor; running them again (e.g. after switching mode) replaces them
        - 'Remove Privacy Suggestions' deletes the suggested areas so they won't be blurred (your brush strokes stay)
        - Higher blur values create stronger effects
        - Images are processed by the server running this app (your own machine when run locally)
        """)
    
    # Status messages for user feedback (dynamic, including ready state)
    status_html = gr.HTML("Ready. Upload an image to begin.", elem_classes="status-info status-bar")

    # ID of the temporary blurred image for the download button (never a file path).
    # The file is deleted when the session data expires (and swept after an hour regardless).
    temp_file_path_for_download_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS, delete_callback=remove_temp_file)
    # Full-quality copy of the uploaded image (kept server-side, freed when the session ends).
    original_image_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)
    # Areas found by the latest Privacy Suggestions run, as a 0/1 mask (brush strokes stay in the editor).
    marks_state = gr.State(None, time_to_live=SESSION_DATA_TTL_SECONDS)

    with gr.Row():
        with gr.Column(scale=3): # Main interactive column
            gr.HTML("<div class='section-header'>Upload & Edit</div>")
            
            file_uploader = gr.File(
                label="Upload Image (PNG, JPG)", 
                type="filepath", 
                file_types=[".png", ".jpg", ".jpeg"],
                file_count="single"
            )
            
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
                canvas_size=(800, 600)
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
                value=51,
                step=2, 
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
                format="png"
            )
            
            download_button = gr.DownloadButton("Download Blurred Image", visible=False, size="lg")
        
    gr.Markdown(
        "<div class='privacy-notice'>"
        "<strong>Privacy First:</strong> Images are processed only by the server running this app "
        "(your own machine when run locally) and are not sent to any third-party service. "
        "The AI models download once from Ultralytics if needed."
        "</div>"
    )
    
    # File uploader actions
    file_uploader.upload(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state, marks_state, marks_preview]
    )
    file_uploader.clear(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state, marks_state, marks_preview]
    )
    
    # Blur button actions
    blur_button.click(
        fn=handle_blur_click,
        inputs=[image_editor, original_image_state, marks_state, temp_file_path_for_download_state, blur_strength_slider],
        outputs=[output_image, status_html, download_button, temp_file_path_for_download_state]
    )
    
    # Suggest button actions
    suggest_button.click(
        fn=handle_suggest_click,
        inputs=[image_editor, original_image_state, detection_mode_radio],
        outputs=[status_html, download_button, marks_state, marks_preview]
    )

    # Clear marks button actions
    clear_marks_button.click(
        fn=handle_clear_marks,
        inputs=[],
        outputs=[marks_preview, status_html, marks_state]
    )

# Limit how many requests can wait at once, so a flood is turned away instead of piling up
demo.queue(max_size=MAX_QUEUED_REQUESTS)

# When main is run, start the application
if __name__ == "__main__":
    threading.Thread(target=_sweep_outputs_forever, daemon=True, name="output-sweeper").start()
    logger.info("Starting Gradio Blur Tool app...")
    # Runs locally only by default. Set the environment variable GRADIO_SHARE=True
    # to also create a temporary public Gradio link.
    demo.launch(theme='base', css=css, max_file_size="25mb", state_session_capacity=MAX_STORED_SESSIONS)