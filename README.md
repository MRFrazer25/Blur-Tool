# Blur Tool

An intelligent image privacy application that combines manual drawing tools with AI-powered object detection for precise area blurring. Built with Gradio ImageEditor and YOLOv11 via Ultralytics, it processes images locally using OpenCV with configurable Gaussian blur kernels while providing both manual control and automated privacy suggestions.

## Features

* **Intelligent Drawing Interface:** Advanced brush tools with customizable size and color for precise area marking
* **Enhanced AI Privacy Suggestions:** YOLOv11 object detection automatically identifies privacy-sensitive objects including people, vehicles, electronics, and personal items with superior accuracy
* **Local Processing:** Complete privacy with 100% local image processing
* **Real-Time Preview:** Instant blur application with adjustable strength control (1-101 intensity levels)
* **Professional Dark Theme:** Modern UI with excellent contrast and accessibility
* **Browser Compatibility Warnings:** Built-in detection and guidance for optimal browser experience
* **Offline Capable:** AI suggestions work offline after initial model download

## Use Cases

* **Privacy Protection:** Blur faces, license plates, and personal identifiers in photos
* **Content Moderation:** Prepare sensitive images for publication or sharing
* **Social Media:** Quick privacy editing for social platform uploads
* **Professional Photography:** Artistic background blur and focus effects
* **Document Redaction:** Hide sensitive information in screenshots and documents

## Requirements

* Python 3.7+
* Internet connection for AI model download (first use only)
* Modern web browser with Canvas/WebGL support (Chrome recommended)

## Setup and Installation

### Option 1: Try it on Hugging Face Spaces

**[Blur Tool](https://huggingface.co/spaces/mattrf/Blur-Tool)**

### Option 2: Local Installation

1. **Clone Repository:**
   ```bash
   git clone [repository-url]
   cd blur-tool
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application:**
   ```bash
   python app.py
   ```
   > **Note:** When you run the app locally, a public Gradio Live link will be generated (because `share=True` is set in the code). This link can be shared with others, but it is temporary and expires after one week. For a permanent public link, use Hugging Face Spaces. See [Gradio's sharing guide](https://www.gradio.app/guides/sharing-your-app) for details.

4. **Access Interface:**
   Open your browser to `http://127.0.0.1:7860` and start processing images.

## How It Works

The application workflow:
1. **Upload** your image through drag-and-drop or file picker (PNG/JPG supported)
2. **Mark Areas** using manual drawing tools or AI-generated privacy suggestions
3. **Configure** blur strength using the intensity slider (1-101 range)
4. **Apply** Gaussian blur processing to marked regions with OpenCV
5. **Download** the processed image with privacy areas blurred

**AI Privacy Detection** automatically identifies:
- People and faces
- Vehicles (cars, buses, trucks, bicycles, motorcycles, trains, boats, airplanes)
- Electronics (cell phones, laptops, TVs)
- Personal items (handbags, backpacks, suitcases)

Choose between manual precision drawing or AI-assisted detection based on your workflow needs.

## Troubleshooting

* **Image Upload Issues:** Verify file format (JPG/PNG) and try different browser. Chrome provides best compatibility.
* **Drawing Tools Not Working:** Firefox may have Canvas/WebGL rendering limitations. The app includes automatic browser compatibility warnings. Try Chrome, Safari, or Edge for full functionality.
* **AI Suggestions Unavailable:** Check internet connection for initial YOLOv11 model download. Manual drawing tools will still work offline.
* **Application Won't Start:** Ensure Python 3.7+ and run `pip install -r requirements.txt`
* **Performance Issues:** Consider resizing large images before processing
* **Blur Not Applied:** Ensure you've drawn areas or used AI suggestions before clicking "Apply Blur"

## Security & Privacy

* **No Data Collection:** Images processed locally without external transmission
* **Session-Only Processing:** No persistent storage of user images
* **Local AI Model:** YOLOv11 runs entirely on your machine after download
* **Privacy-First Design:** Complete control over your image data
* **Secure File Handling:** Automatic cleanup of temporary files
* **Input Validation:** Comprehensive security checks for uploaded files

## License

MIT License - see the [LICENSE](LICENSE) file for details. 
