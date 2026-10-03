# Blur Tool

An intelligent image privacy application that combines manual drawing tools with AI-powered object detection for precise area blurring. Built with Gradio ImageEditor and YOLO26 via Ultralytics, it processes images locally using OpenCV with configurable Gaussian blur kernels while providing both manual control and automated privacy suggestions.

## Features

* **Intelligent Drawing Interface:** Advanced brush tools with customizable size and color for precise area marking
* **AI Privacy Suggestions:** YOLO26 finds privacy-sensitive areas for you, as whole-object boxes, exact outlines, or faces only
* **Local Processing:** When run locally, images never leave your machine
* **Real-Time Preview:** Instant blur application with adjustable strength (10–100) that scales with image size
* **Professional Dark Theme:** Modern UI with excellent contrast and accessibility
* **Browser Compatibility Notice:** Built-in guidance for Firefox users
* **Offline Capable:** AI suggestions work offline after initial model download

## Use Cases

* **Privacy Protection:** Blur faces, license plates, and personal identifiers in photos
* **Content Moderation:** Prepare sensitive images for publication or sharing
* **Social Media:** Quick privacy editing for social platform uploads
* **Professional Photography:** Artistic background blur and focus effects
* **Document Redaction:** Hide sensitive information in screenshots and documents

## Requirements

* Python 3.10+
* Gradio 6+ (installed via `requirements.txt`)
* Internet connection for AI model download (first use only)
* Modern web browser with Canvas/WebGL support (Chrome recommended)

## Setup and Installation

### Option 1: Try it on Hugging Face Spaces

**[Blur Tool](https://huggingface.co/spaces/mattrf/Blur-Tool)**

The Space uses the same `app.py` and `requirements.txt` as this repository. Its README (with the Space settings header) is kept in [huggingface/README.md](huggingface/README.md).

### Option 2: Local Installation

1. **Clone Repository:**
   ```bash
   git clone https://github.com/MRFrazer25/Blur-Tool.git
   cd Blur-Tool
   ```

2. **Install Dependencies:**
   ```bash
   pip install -U -r requirements.txt
   ```

3. **Run Application:**
   ```bash
   python app.py
   ```
   > **Note:** By default the app is only reachable from your own machine. To also generate a temporary public Gradio Live link (expires after one week), set the environment variable `GRADIO_SHARE=True` before running. Anyone with that link can use the app, and their images are processed on your machine. For a permanent public link, use Hugging Face Spaces. See [Gradio's sharing guide](https://www.gradio.app/guides/sharing-your-app) for details.

4. **Access Interface:**
   Open your browser to `http://127.0.0.1:7860` and start processing images.

## How It Works

The application workflow:
1. **Upload** your image through drag-and-drop or file picker (PNG/JPG supported)
2. **Mark Areas** using manual drawing tools or AI-generated privacy suggestions
3. **Configure** blur strength using the intensity slider (10–100; stronger on larger photos)
4. **Apply** Gaussian blur processing to marked regions with OpenCV
5. **Download** the processed image with privacy areas blurred

**AI Privacy Suggestions** has three detection modes:
- **Whole objects:** boxes over people, vehicles (cars, buses, trucks, bicycles, motorcycles, trains, boats, airplanes), electronics (cell phones, laptops, TVs), and personal items (handbags, backpacks, suitcases)
- **Exact outlines:** traces the exact shape of the same objects, so less of the background gets blurred
- **Faces only:** finds people's faces from body keypoints (eyes, nose, ears) and marks just the head

Suggestions appear in a preview below the editor, and Apply Blur blurs them together with your brush strokes. Running Privacy Suggestions again (for example after switching mode) replaces the previous suggestions. **Remove Privacy Suggestions** deletes the suggested areas so they won't be blurred. Neither affects your brush strokes; use the editor's eraser or undo for those. If the suggestion marks expire (about 30 minutes of inactivity) or the session is dropped, Apply Blur refuses and asks you to run Privacy Suggestions again instead of silently leaving those areas unblurred. Faces that are very small, hidden, or turned away can be missed, and license plates and text are not detected, so check the preview and mark anything else with the brush tool.

Choose between manual precision drawing or AI-assisted detection based on your workflow needs.

## Troubleshooting

* **Image Upload Issues:** Verify file format (JPG/PNG) and try different browser. Chrome provides best compatibility.
* **Drawing Tools Not Working:** Firefox may have Canvas/WebGL rendering limitations. Try Chrome, Safari, or Edge for full functionality.
* **AI Suggestions Unavailable:** Check internet connection for initial YOLO26 model downloads (three small models, about 20 MB total). Manual drawing tools will still work offline.
* **Application Won't Start:** Ensure Python 3.10+ and run `pip install -U -r requirements.txt`. A `TypeError` about `theme` or `css` in `launch()` means an older Gradio (5.x or earlier) is installed.
* **Image Too Large:** Images over 40 megapixels are rejected. Resize them before uploading
* **Performance Issues:** Consider resizing large images before processing
* **Blur Not Applied:** Ensure you've drawn areas or used AI suggestions before clicking "Apply Blur"

## Security & Privacy

* **No Data Collection:** Images are processed only by the server running the app (your machine when run locally) and are never sent to a third-party service
* **No Analytics:** Gradio and Ultralytics usage analytics are turned off inside the app
* **Session-Only Processing:** Uploaded images and blurred results are deleted within about an hour, even if the app crashes or restarts. Per-session data expires after 30 minutes of inactivity. Sessions keep a compressed copy of the upload and a packed suggestion mask, not full-resolution RGBA arrays
* **Abuse Limits:** At most 20 queued requests and 25 stored sessions at a time. The 25 MB upload cap, session cap, and output cleanup apply whether you start with `python app.py` or `gradio app.py`
* **Metadata Removed:** Downloads are saved as fresh PNGs without the original photo's EXIF data (such as GPS location or camera details)
* **Local AI Models:** YOLO26 runs entirely on your machine after download
* **Private by Default:** No public share link is created unless you opt in with `GRADIO_SHARE=True`
* **Upload Limits:** Only real PNG/JPG files are accepted (checked by content, not just extension), up to 25 MB and 40 megapixels. The same size and layer limits are checked again when you blur or run Privacy Suggestions
* **Safe Errors:** Error messages shown in the app never include server file paths or internal details
* **Path Validation:** File paths are checked to stay inside the temp/upload directories before any read or delete. Blurred outputs are written to a per-process temp folder

## License

MIT License - see the [LICENSE](LICENSE) file for details. 
