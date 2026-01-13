import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import os
import tempfile
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load YOLOv11 model when the application starts.
yolo_model = None
try:
    logger.info("Attempting to load YOLOv11 model...")
    # Download model from Ultralytics if not cached (requires internet on first run)
    yolo_model = YOLO("yolo11n.pt")  # Using nano version for better speed/efficiency balance
    logger.info("YOLOv11 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLOv11 model: {e}. The 'Privacy Suggestions' feature will be unavailable.", exc_info=True)
    # Application continues to function without AI features if model loading fails

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

    # Ensure blur_radius is odd and positive for cv2.GaussianBlur kernel size
    k_val = max(1, int(blur_radius_odd))
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
    """Handles new file uploads, prepares image for editor, and cleans up old temp files."""
    if uploaded_file_path:
        try:
            # Validate uploaded_file_path
            resolved_upload_path = os.path.realpath(uploaded_file_path)
            system_temp_dir = os.path.realpath(tempfile.gettempdir())

            if not resolved_upload_path.startswith(system_temp_dir):
                logger.error(f"Security alert: Upload path '{uploaded_file_path}' resolves to '{resolved_upload_path}', which is outside the system temp directory '{system_temp_dir}'. Aborting upload.")
                return (
                    gr.update(value=None),
                    gr.HTML("Invalid file path detected. Upload failed.", elem_classes="status-error"),
                    gr.DownloadButton(visible=False),
                    None,
                    current_temp_file_for_download 
                )
            
            img = Image.open(resolved_upload_path).convert("RGBA") # Use resolved path
            logger.info(f"Image '{os.path.basename(resolved_upload_path)}' loaded successfully by user.")
            
            # Clean up previous temporary file for download, if one exists
            if current_temp_file_for_download:
                resolved_temp_download_path = os.path.realpath(current_temp_file_for_download)
                if not resolved_temp_download_path.startswith(system_temp_dir):
                    logger.error(f"Security alert: Temp download path '{current_temp_file_for_download}' resolves to '{resolved_temp_download_path}', which is outside the system temp directory '{system_temp_dir}'. Skipping cleanup of this path.")
                # Proceed with cleanup only if path is valid and file exists
                elif os.path.exists(resolved_temp_download_path):
                    try:
                        os.remove(resolved_temp_download_path)
                        logger.info(f"Removed previous temporary download file: {resolved_temp_download_path}")
                    except Exception as e_rem:
                        logger.error(f"Error removing old temp file {resolved_temp_download_path}: {e_rem}")
            
            return (
                gr.update(value=img), 
                gr.HTML("Image loaded successfully! You can now draw on it or get AI suggestions.", elem_classes="status-success"),
                gr.DownloadButton(visible=False), 
                None, 
                None 
            )
        except Exception as e:
            logger.error(f"Error processing uploaded file '{uploaded_file_path}': {e}", exc_info=True)
            return (
                gr.update(value=None), 
                gr.HTML(f"Error loading image: {str(e)}", elem_classes="status-error"),
                gr.DownloadButton(visible=False),
                None,
                current_temp_file_for_download 
            )
    
    # Case: No file uploaded
    return (
        gr.update(value=None), 
        gr.HTML("No file provided or file cleared.", elem_classes="status-info"), # Added more specific message
        gr.DownloadButton(visible=False),
        None,
        current_temp_file_for_download
    )

def handle_blur_click(editor_data, current_temp_file_for_download, blur_strength_slider_value):
    """Applies blur to areas drawn by user in ImageEditor, prepares for download."""
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

    background_np_rgba = np.array(background_pil.convert("RGBA"))
    
    if not layers_pil: # No drawing layers found
        return (
            None,
            gr.HTML("No areas marked for blurring. Use the brush tools or AI suggestions first.", elem_classes="status-info"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download
        )

    # Combine all drawing layers (alpha channels) into a single binary mask
    # Assumes drawings are on layers that can be composited by their alpha.
    combined_mask_pil = Image.new("L", background_pil.size, 0) # 'L' mode for 8-bit grayscale
    for layer_pil_rgba in layers_pil:
        if layer_pil_rgba and isinstance(layer_pil_rgba, Image.Image):
            # Ensure layer is RGBA to get alpha
            layer_alpha = layer_pil_rgba.convert("RGBA").split()[-1]
            combined_mask_pil.paste(layer_alpha, (0,0), mask=layer_alpha)

    # Threshold to binary mask: 1 where drawn, 0 otherwise
    final_mask_np_binary = (np.array(combined_mask_pil) > 128).astype(np.uint8) 
    
    if np.sum(final_mask_np_binary) == 0: # Check if mask is empty
         return (
            None, 
            gr.HTML("Drawn areas are too faint to detect. Try using a more opaque brush.", elem_classes="status-info"),
            gr.DownloadButton(visible=False),
            current_temp_file_for_download
        )

    # Clean up previous temporary file for download, validating its path first
    if current_temp_file_for_download:
        resolved_temp_download_path = os.path.realpath(current_temp_file_for_download)
        system_temp_dir = os.path.realpath(tempfile.gettempdir())
        if not resolved_temp_download_path.startswith(system_temp_dir):
            logger.error(f"Security alert: Temp download path '{current_temp_file_for_download}' resolves to '{resolved_temp_download_path}', which is outside the system temp directory '{system_temp_dir}'. Skipping cleanup of this path.")
        # Proceed with cleanup only if path is valid and file exists
        elif os.path.exists(resolved_temp_download_path):
            try:
                os.remove(resolved_temp_download_path)
                logger.info(f"Removed old temp download file: {resolved_temp_download_path}")
            except Exception as e:
                logger.error(f"Error removing old temp file {resolved_temp_download_path}: {e}")
    
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
            gr.HTML(f"Error processing blur: {str(e)}", elem_classes="status-error"),
            gr.DownloadButton(visible=False),
            None # Reset temp file state on error
        )

def handle_suggest_click(editor_data):
    """Uses YOLOv11 to detect objects and adds them as a new layer in ImageEditor."""
    if not yolo_model:
        return (
            editor_data, # Return original data, no changes
            gr.HTML("Privacy Suggestions unavailable: YOLOv11 model not loaded.", elem_classes="status-error"),
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
        results = yolo_model(background_for_yolo)
        
        # Create a new transparent layer for suggestions
        suggestion_layer_pil = Image.new("RGBA", background_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(suggestion_layer_pil)
        
        suggestion_made = False
        # Define common object classes to suggest for blurring (privacy-focused)
        target_classes = [
            'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'boat', 'airplane',
            'cell phone', 'laptop', 'tv', 'handbag', 'backpack', 'suitcase'
        ]

        # Process YOLOv11 results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in target_classes and confidence > 0.4:  # Confidence threshold
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw semi-transparent red rectangle for suggestion
                        draw.rectangle([xmin, ymin, xmax, ymax], fill=(255, 0, 0, 100)) # RGBA: Red, ~40% opacity
                        suggestion_made = True
        
        # Get current layers, ensure it's a list
        current_layers = editor_data.get('layers', []) if editor_data.get('layers') else []
        updated_layers = current_layers + ([suggestion_layer_pil] if suggestion_made else [])
        
        # Update the ImageEditor value with the new layer
        updated_editor_value = {
            "background": background_pil,
            "layers": updated_layers,
            "composite": None # Let Gradio rebuild composite from background and new layers
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
            gr.HTML(f"Error with Privacy Suggestions: {str(e)}", elem_classes="status-error"),
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
.gr-block-title, .gr-form label, .markdown, p, span, div {
    color: #e2e8f0 !important;
}
/* Input components - high contrast for readability */
.gr-textbox, .gr-textbox input, .gr-dropdown, .gr-dropdown select, 
.gr-slider input, .gr-number input, input, textarea, select,
.gr-file-input, .gr-file-input input, .gr-file-input label {
    color: #ffffff !important;
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    border-radius: 6px !important;
}
/* Focus states for inputs */
.gr-textbox:focus, .gr-textbox input:focus, .gr-dropdown:focus, 
.gr-dropdown select:focus, input:focus, textarea:focus, select:focus {
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
h1, h2, h3, h4, h5, h6 {
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
.status-bar {
    margin-top: 12px !important;
    margin-bottom: 12px !important;
    font-size: 1.02em !important;
    opacity: 0.98;
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
/* Button improvements - complementary colors */
.gr-button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    color: #ffffff !important;
    border: none !important;
}
.gr-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4) !important;
}
/* Primary button styling - blue theme */
.gr-button.gr-button-primary {
    background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%) !important;
}
.gr-button.gr-button-primary:hover {
    background: linear-gradient(135deg, #2c5282 0%, #2a4365 100%) !important;
}
/* Secondary button styling - complementary grey */
.gr-button:not(.gr-button-primary) {
    background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%) !important;
}
.gr-button:not(.gr-button-primary):hover {
    background: linear-gradient(135deg, #2c5282 0%, #2a4365 100%) !important;
}
/* Download button special styling */
.gr-button[size="lg"] {
    background: linear-gradient(135deg, #38a169 0%, #2f855a 100%) !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
}
.gr-button[size="lg"]:hover {
    background: linear-gradient(135deg, #2f855a 0%, #276749 100%) !important;
}
/* Slider styling - neutral dark track, blue thumb only */
.gr-slider input[type="range"] {
    background-color: #232b36 !important;
    box-shadow: none !important;
    border-radius: 4px !important;
    height: 6px !important;
}
.gr-slider input[type="range"]::-webkit-slider-thumb {
    background-color: #3182ce !important;
    border: 2px solid #ffffff !important;
}
.gr-slider input[type="range"]::-moz-range-thumb {
    background-color: #3182ce !important;
    border: 2px solid #ffffff !important;
}
.gr-slider input[type="range"]::-ms-thumb {
    background-color: #3182ce !important;
    border: 2px solid #ffffff !important;
}
/* Component containers */
.gr-block {
    margin-bottom: 15px !important;
    background-color: transparent !important;
}
/* Image components */
.gr-image, .gr-file {
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    border-radius: 8px !important;
}
/* Cross-browser compatibility */
* {
    box-sizing: border-box;
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
.gr-accordion .gr-accordion-header {
    background-color: #374151 !important;
    color: #e2e8f0 !important;
}
/* File upload styling */
.gr-file-upload {
    background-color: #2d3748 !important;
    border: 2px dashed #4a5568 !important;
    border-radius: 8px !important;
}
.gr-file-upload:hover {
    border-color: #3182ce !important;
    background-color: #374151 !important;
}
/* Label improvements */
label {
    color: #cbd5e0 !important;
    font-weight: 500 !important;
}
"""

# Build the Gradio interface using Blocks for layout flexibility.
with gr.Blocks(title="Blur Tool") as demo:
    gr.Markdown("# Blur Tool", elem_classes="main-title")
    gr.Markdown("*Draw to blur images then Download them. Powered by Gradio, OpenCV, and YOLOv11.*", elem_classes="subtitle")
    
    # Browser compatibility warning
    gr.HTML("""
    <div style="background: #2d4a22; border: 1px solid #38a169; color: #68d391; padding: 12px 16px; margin: 15px 0; border-radius: 8px; font-weight: 500;">
        <strong> Firefox Users:</strong> If the image editor doesn't work, try enabling hardware acceleration in Firefox Settings → Performance, 
        or use Google Chrome for guaranteed compatibility.
    </div>
    """, elem_classes="compatibility-notice")
    
    # Collapsible instructions
    with gr.Accordion("How to Use", open=False):
        gr.Markdown("""
        **Quick Start Guide:**
        
        1. **Upload**: Click 'Upload Image' or drag & drop your JPG/PNG file
        2. **Mark Areas**: Choose your method:
           - **Manual**: Use brush tools to draw on areas you want blurred
           - **AI Assist**: Click 'Privacy Suggestions' - uses YOLOv11 AI to automatically detect people, cars, and other privacy-sensitive objects
        3. **Apply**: Adjust blur strength and click 'Apply Blur'
        4. **Download**: Save your processed image
        
        **Pro Tips:**
        - Use red brush for clear visibility on most images
        - Privacy Suggestions detects common privacy targets: people, vehicles, faces, electronics
        - AI suggestions appear as red overlays that you can edit or use as-is
        - Higher blur values create stronger effects
        - All processing happens locally for privacy
        """)
    
    # Status messages for user feedback (dynamic, including ready state)
    status_html = gr.HTML("Ready. Upload an image to begin.", elem_classes="status-info status-bar")

    # State variable to hold the path of the temporary blurred image for the download button.
    temp_file_path_for_download_state = gr.State(None)

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
                minimum=1, 
                maximum=101, 
                value=25, 
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
                type="pil"
            )
            
            download_button = gr.DownloadButton("Download Blurred Image", visible=False, size="lg")
        
    gr.Markdown(
        "<div class='privacy-notice'>"
        "<strong>Privacy First:</strong> All processing happens locally on your machine. "
        "No images are sent to external servers. AI model downloads once from Ultralytics if needed."
        "</div>"
    )
    
    # File uploader actions
    file_uploader.upload(
        fn=handle_file_upload,
        inputs=[file_uploader, temp_file_path_for_download_state],
        outputs=[image_editor, status_html, download_button, output_image, temp_file_path_for_download_state]
    )
    
    # Blur button actions
    blur_button.click(
        fn=handle_blur_click,
        inputs=[image_editor, temp_file_path_for_download_state, blur_strength_slider],
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
    demo.launch(share=True, theme='base', css=css)