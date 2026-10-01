import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import os
import tempfile
import logging

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

# Load YOLO26 model when the application starts.
yolo_model = None
try:
    logger.info("Attempting to load YOLO26 model...")
    # Download model from Ultralytics if not cached (requires internet on first run)
    yolo_model = YOLO("yolo26n.pt")  # Using nano version for better speed/efficiency balance
    logger.info("YOLO26 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO26 model: {e}. The 'Privacy Suggestions' feature will be unavailable.", exc_info=True)
    # Application continues to function without AI features if model loading fails

# Object classes (COCO) suggested for blurring, and the minimum detection confidence
PRIVACY_TARGET_CLASSES = {
    'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'boat', 'airplane',
    'cell phone', 'laptop', 'tv', 'handbag', 'backpack', 'suitcase'
}
SUGGESTION_CONFIDENCE = 0.4

# Upload limits: only real PNG/JPEG files are decoded, up to 40 megapixels
ALLOWED_IMAGE_FORMATS = ["PNG", "JPEG"]
MAX_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # Pillow rejects decompression bombs beyond 2x this

# Blur strength range (must match the slider)
MIN_BLUR_STRENGTH = 1
MAX_BLUR_STRENGTH = 101

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

def remove_temp_file(path):
    """Deletes a previously generated temp file, but only if it lives inside an allowed temp directory."""
    if not path:
        return
    if not is_within_temp_dir(path):
        logger.error(f"Security alert: Temp path '{path}' is outside the allowed temp directories. Skipping cleanup of this path.")
        return
    resolved_path = os.path.realpath(path)
    if os.path.exists(resolved_path):
        try:
            os.remove(resolved_path)
            logger.info(f"Removed previous temporary download file: {resolved_path}")
        except Exception as e:
            logger.error(f"Error removing old temp file {resolved_path}: {e}")

# Core Functions
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
    holds a compressed preview copy)."""
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
                img
            )
        except (UnidentifiedImageError, Image.DecompressionBombError) as e:
            logger.warning(f"Rejected uploaded file: {e}")
            return (
                gr.update(),
                gr.HTML("Could not open this file. Please upload a valid PNG or JPG image under 40 megapixels.", elem_classes="status-error"),
                gr.DownloadButton(visible=False),
                None,
                current_temp_file_for_download,
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
        gr.skip()  # Keep the original in sync with the editor, which is also left as-is
    )

def handle_blur_click(editor_data, original_image, current_temp_file_for_download, blur_strength_slider_value):
    """Applies blur to areas drawn by user in ImageEditor, prepares for download.
    Blurs the full-quality original upload when available, using the editor only for the marked areas."""
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

    # The editor's copy is compressed for display, so blur the original upload instead
    if isinstance(original_image, Image.Image) and original_image.size == background_pil.size:
        source_pil = original_image
    else:
        source_pil = background_pil
    background_np_rgba = np.array(source_pil.convert("RGBA"))

    if not layers_pil: # No drawing layers found
        return (
            None,
            gr.HTML("No areas marked for blurring. Use the brush tools or AI suggestions first.", elem_classes="status-info"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download
        )

    # Combine all drawing layers into a single binary mask: any pixel with
    # non-zero alpha on any layer is marked. AI suggestion boxes are drawn
    # semi-transparent, so they must count as marked too.
    combined_alpha_np = np.zeros((background_pil.height, background_pil.width), dtype=np.uint8)
    for layer_pil_rgba in layers_pil:
        if layer_pil_rgba and isinstance(layer_pil_rgba, Image.Image):
            # Ensure layer is RGBA to get alpha, and aligned with the background
            layer_alpha = layer_pil_rgba.convert("RGBA").split()[-1]
            if layer_alpha.size != background_pil.size:
                layer_alpha = layer_alpha.resize(background_pil.size, Image.NEAREST)
            combined_alpha_np = np.maximum(combined_alpha_np, np.array(layer_alpha))

    # Threshold to binary mask: 1 where drawn, 0 otherwise
    final_mask_np_binary = (combined_alpha_np > 0).astype(np.uint8)

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="blurred_") as tmp_file:
            blurred_image_pil.save(tmp_file.name, "PNG")
            new_temp_file_for_download_path = tmp_file.name
        logger.info(f"Blurred image saved to temporary file for download: {new_temp_file_for_download_path}")
        
        return (
            blurred_image_pil, # Display in output_image component
            gr.HTML("Blur applied successfully! Download your privacy-protected image below.", elem_classes="status-success"),
            gr.DownloadButton(value=new_temp_file_for_download_path, visible=True, label="Download Blurred Image"),
            new_temp_file_for_download_path # Update state with new temp file path
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

def handle_suggest_click(editor_data):
    """Uses YOLO26 to detect objects and adds them as a new layer in ImageEditor."""
    if not yolo_model:
        return (
            editor_data, # Return original data, no changes
            gr.HTML("Privacy Suggestions unavailable: YOLO26 model not loaded.", elem_classes="status-error"),
            gr.DownloadButton(visible=False) # Ensure download button is hidden
        )
        
    if not editor_data or not editor_data.get('background'):
        return (
            editor_data, 
            gr.HTML("Please upload an image first to use Privacy Suggestions.", elem_classes="status-error"),
            gr.DownloadButton(visible=False)
        )

    background_pil = editor_data['background']
    if not isinstance(background_pil, Image.Image):
         logger.error("Background for suggestion is not a PIL image.")
         return editor_data, gr.HTML("Internal error: Image format incorrect for Privacy Suggestions.", elem_classes="status-error"), gr.DownloadButton(visible=False)

    # YOLO typically works best with RGB images
    background_for_yolo = background_pil.convert("RGB")

    try:
        results = yolo_model(background_for_yolo, conf=SUGGESTION_CONFIDENCE, verbose=False)

        # Create a new transparent layer for suggestions
        suggestion_layer_pil = Image.new("RGBA", background_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(suggestion_layer_pil)

        suggestion_made = False

        # Process YOLO26 results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in PRIVACY_TARGET_CLASSES and confidence >= SUGGESTION_CONFIDENCE:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw semi-transparent red rectangle for suggestion
                        draw.rectangle([xmin, ymin, xmax, ymax], fill=(255, 0, 0, 100)) # RGBA: Red, ~40% opacity
                        suggestion_made = True
        
        # Merge existing drawings and the new suggestion boxes into a single layer.
        # Sending several layers back makes the Gradio 6 editor hang while loading.
        current_layers = editor_data.get('layers') or []
        merged_layer_pil = Image.new("RGBA", background_pil.size, (0, 0, 0, 0))
        for layer_pil in current_layers:
            if isinstance(layer_pil, Image.Image):
                layer_rgba = layer_pil.convert("RGBA")
                if layer_rgba.size != background_pil.size:
                    layer_rgba = layer_rgba.resize(background_pil.size, Image.NEAREST)
                merged_layer_pil = Image.alpha_composite(merged_layer_pil, layer_rgba)
        if suggestion_made:
            merged_layer_pil = Image.alpha_composite(merged_layer_pil, suggestion_layer_pil)
        updated_layers = [merged_layer_pil] if (current_layers or suggestion_made) else []

        # Update the ImageEditor value with the new layer
        composite_pil = background_pil.convert("RGBA")
        for layer_pil in updated_layers:
            composite_pil = Image.alpha_composite(composite_pil, layer_pil)
        updated_editor_value = {
            "background": background_pil,
            "layers": updated_layers,
            "composite": composite_pil
        }
        
        if suggestion_made:
            status_msg = "Objects found! Red boxes show detected people, vehicles, and other items you might want to blur for privacy."
            status_class = "status-success"
        else:
            status_msg = "No common privacy objects (people, cars, etc.) detected with high confidence. Try manual drawing instead."
            status_class = "status-info"
            
        return (
            gr.update(value=updated_editor_value), 
            gr.HTML(status_msg, elem_classes=status_class),
            gr.DownloadButton(visible=False)
        )

    except Exception as e:
        logger.error(f"Error during AI suggestion generation: {e}", exc_info=True)
        return (
            editor_data, 
            gr.HTML("Error with Privacy Suggestions. Please try again or mark areas manually.", elem_classes="status-error"),
            gr.DownloadButton(visible=False)
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
           - **AI Assist**: Click 'Privacy Suggestions' - uses YOLO26 AI to automatically detect people, cars, and other privacy-sensitive objects
        3. **Apply**: Adjust blur strength and click 'Apply Blur'
        4. **Download**: Save your processed image
        
        **Pro Tips:**
        - Use red brush for clear visibility on most images
        - Privacy Suggestions detects common privacy targets: people, vehicles, electronics, bags
        - It does not detect faces or license plates on their own - mark those with the brush
        - AI suggestions appear as red overlays that you can edit or use as-is
        - Higher blur values create stronger effects
        - Images are processed by the server running this app (your own machine when run locally)
        """)
    
    # Status messages for user feedback (dynamic, including ready state)
    status_html = gr.HTML("Ready. Upload an image to begin.", elem_classes="status-info status-bar")

    # State variable to hold the path of the temporary blurred image for the download button.
    # The file is deleted when the user's session ends.
    temp_file_path_for_download_state = gr.State(None, delete_callback=remove_temp_file)
    # Full-quality copy of the uploaded image (kept server-side, freed when the session ends).
    original_image_state = gr.State(None)

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
            
        with gr.Column(scale=2): # Actions and results column
            gr.HTML("<div class='section-header'>Controls</div>")
            
            blur_strength_slider = gr.Slider(
                minimum=MIN_BLUR_STRENGTH,
                maximum=MAX_BLUR_STRENGTH,
                value=51,
                step=2, 
                label="Blur Strength"
            )

            with gr.Row():
                suggest_button = gr.Button("Privacy Suggestions", size="sm", variant="primary")
                blur_button = gr.Button("Apply Blur", variant="primary", size="sm")

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
        "The AI model downloads once from Ultralytics if needed."
        "</div>"
    )
    
    # File uploader actions
    file_uploader.upload(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state]
    )
    file_uploader.clear(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state, original_image_state]
    )
    
    # Blur button actions
    blur_button.click(
        fn=handle_blur_click,
        inputs=[image_editor, original_image_state, temp_file_path_for_download_state, blur_strength_slider],
        outputs=[output_image, status_html, download_button, temp_file_path_for_download_state]
    )
    
    # Suggest button actions
    suggest_button.click(
        fn=handle_suggest_click,
        inputs=[image_editor],
        outputs=[image_editor, status_html, download_button]
    )

# When main is run, start the application
if __name__ == "__main__":
    logger.info("Starting Gradio Blur Tool app...")
    # Runs locally only by default. Set the environment variable GRADIO_SHARE=True
    # to also create a temporary public Gradio link.
    demo.launch(theme='base', css=css, max_file_size="25mb")