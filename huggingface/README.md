---
title: Blur Tool
emoji: 👀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: mit
short_description: 'Blur images with freedom: Powered by Python, Gradio, YOLO26'
---

# Blur Tool

An intelligent image privacy application combining manual drawing tools with AI-powered object detection for precise area blurring.

## Features

* **Intelligent Drawing Interface:** Brush tools with customizable size and color for precise area marking
* **AI Privacy Suggestions:** YOLO26 finds privacy-sensitive areas for you, as whole-object boxes, exact outlines, or faces only
* **Hosted on Hugging Face Spaces:** Runs in your browser via the hosted Space
* **Real-Time Preview:** Instant blur application with adjustable strength control (1–101 intensity levels)
* **Professional Dark Theme:** Modern UI with excellent contrast and accessibility
* **Browser Compatibility Notice:** Built-in guidance for Firefox users

## Use Cases

* **Privacy Protection:** Blur faces, license plates, and personal identifiers in photos
* **Content Moderation:** Prepare sensitive images for publication or sharing
* **Social Media:** Quick privacy editing before uploading to platforms
* **Professional Photography:** Artistic background blur and focus effects
* **Document Redaction:** Hide sensitive information in screenshots and documents

## How It Works

1. **Navigate** to the Hugging Face Space URL in your browser.
2. **Upload** your image through drag-and-drop or file picker (PNG/JPG supported).
3. **Mark Areas** using manual drawing tools or AI-generated privacy suggestions.
4. **Configure** blur strength using the intensity slider (1–101 range).
5. **Apply** Gaussian blur processing to marked regions via the hosted service.
6. **Download** the processed image directly from your browser with privacy areas blurred.

**AI Privacy Suggestions** has three detection modes:
- **Whole objects:** boxes over people, vehicles (cars, buses, trucks, bicycles, motorcycles, trains, boats, airplanes), electronics (cell phones, laptops, TVs), and personal items (handbags, backpacks, suitcases)
- **Exact outlines:** traces the exact shape of the same objects, so less of the background gets blurred
- **Faces only:** finds people's faces from body keypoints (eyes, nose, ears) and marks just the head

Suggestions appear in a preview below the editor, and Apply Blur blurs them together with your brush strokes. Running Privacy Suggestions again (in any mode) adds more areas. **Remove Privacy Suggestions** deletes all the suggested areas so they won't be blurred - for example, to switch from whole objects to faces only. It doesn't affect your brush strokes; use the editor's eraser or undo for those. Faces that are very small, hidden, or turned away can be missed, and license plates and text are not detected, so check the preview and mark anything else with the brush tool.

## Troubleshooting

* **Image Upload Issues:** Verify file format (JPG/PNG) and refresh the page. Chrome generally provides the best compatibility.
* **Image Too Large:** Uploads are limited to 25 MB and 40 megapixels. Resize larger images before uploading.
* **Drawing Tools Not Working:** Some browsers (such as older Firefox or mobile browsers) may have Canvas/WebGL rendering limitations. Try Chrome, Safari, or Edge if issues arise.
* **AI Suggestions Unavailable:** The YOLO26 models (about 20 MB total) are downloaded when the Space starts. If suggestions fail, wait a moment and try again, or mark areas manually.
* **Blur Not Applied:** Ensure you have drawn areas or used AI suggestions before clicking "Apply Blur."
* **Slow Response on Large Images:** For very high-resolution uploads, try resizing locally (for example, to 1920×1080) before uploading to improve responsiveness.

## Security & Privacy

* **Where Images Go:** Images are processed on this Hugging Face Space's server and are never sent to any other third-party service.
* **No Analytics:** Gradio and Ultralytics usage analytics are turned off inside the app.
* **Temporary Storage Only:** Uploads and results are stored as temporary files on the Space. Blurred results are deleted when your session ends, and uploads are cleared from the cache after about an hour.
* **Metadata Removed:** Downloads are saved as fresh PNGs without the original photo's EXIF data (such as GPS location or camera details).
* **Upload Checks:** Only real PNG/JPG files are accepted (checked by content, not just extension), up to 25 MB and 40 megapixels.
* **Safe Errors:** Error messages never include server file paths or internal details.

To run the app on your own machine so images never leave it, see the GitHub repository.

[GitHub](https://github.com/MRFrazer25/Blur-Tool)
