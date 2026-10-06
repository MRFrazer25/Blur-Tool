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

Suggestions appear in a preview below the editor, and Apply Blur blurs them together with your brush strokes. Running Privacy Suggestions again (for example after switching mode) replaces the previous suggestions. **Remove Privacy Suggestions** deletes the suggested areas so they won't be blurred. Neither affects your brush strokes; use the editor's eraser or undo for those. If the suggestion marks expire (about 30 minutes after they were last saved) or the session is dropped, Apply Blur refuses and asks you to run Privacy Suggestions again instead of silently leaving those areas unblurred. Faces that are very small, hidden, or turned away can be missed, and license plates and text are not detected, so check the preview and mark anything else with the brush tool.

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

Where the photo goes depends on where the app is running. The same `app.py` shows a different privacy notice in each place: Hugging Face sets `SPACE_ID`, and a local run does not.

**On your machine** (`python app.py`):

* The photo is processed on that computer. It is not uploaded anywhere. The YOLO files that download on first use are the models, not your photo
* Gradio and Ultralytics analytics are turned off in the app
* The stored image, suggestion marks, and download id use a 30-minute Gradio `time_to_live`, counted from the last time each value was saved. How Gradio deletes state and cached uploads is in [Gradio's resource cleanup guide](https://www.gradio.app/guides/resource-cleanup). Who can request a cached file is in [Gradio's file access guide](https://www.gradio.app/guides/file-access)
* Downloads are new PNG files without the original photo's location or camera data. The session copy used for blurring is also a PNG saved without that EXIF
* No public share link is created unless you set `GRADIO_SHARE=True`. Anyone with that link sends their photos to your computer for processing

**On the [Hugging Face Space](https://huggingface.co/spaces/mattrf/Blur-Tool):**

* The page opens in the browser. The photo is uploaded to the Space, and blur and YOLO run on the Space server. Hugging Face has the photo because Hugging Face is running the app. It is not sent to any other service
* How long files stay on a Space's disk is [Hugging Face's disk usage guide](https://huggingface.co/docs/hub/spaces-storage). This app does not attach a Storage Bucket
* The download is a new PNG without the original location or camera data, same as a local run
* To keep a photo from leaving your computer, run the app locally

**Either way:**

* **Abuse Limits:** At most 20 queued requests and 25 stored sessions at a time. The 25 MB upload cap, session cap, and output cleanup apply whether you start with `python app.py`, `gradio app.py`, or on the Space
* **Upload Limits:** Only real PNG/JPG files are accepted (checked by content, not just extension), up to 25 MB and 40 megapixels. The same size and layer limits are checked again when you blur or run Privacy Suggestions
* **Safe Errors:** Error messages shown in the app never include server file paths or internal details
* **Path Validation:** File paths are checked to stay inside the temp/upload directories before any read or delete. Blurred outputs are written to a per-process temp folder

## License

MIT License - see the [LICENSE](LICENSE) file for details. 
