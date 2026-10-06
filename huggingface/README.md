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
* **Hosted on Hugging Face Spaces:** The page opens in your browser. Blur and YOLO run on the Space server, so the photo is uploaded there
* **Real-Time Preview:** Instant blur application with adjustable strength (10–100) that scales with image size
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
4. **Configure** blur strength using the intensity slider (10–100; stronger on larger photos).
5. **Apply** Gaussian blur processing to marked regions via the hosted service.
6. **Download** the processed image directly from your browser with privacy areas blurred.

**AI Privacy Suggestions** has three detection modes:
- **Whole objects:** boxes over people, vehicles (cars, buses, trucks, bicycles, motorcycles, trains, boats, airplanes), electronics (cell phones, laptops, TVs), and personal items (handbags, backpacks, suitcases)
- **Exact outlines:** traces the exact shape of the same objects, so less of the background gets blurred
- **Faces only:** finds people's faces from body keypoints (eyes, nose, ears) and marks just the head

Suggestions appear in a preview below the editor, and Apply Blur blurs them together with your brush strokes. Running Privacy Suggestions again (for example after switching mode) replaces the previous suggestions. **Remove Privacy Suggestions** deletes the suggested areas so they won't be blurred. Neither affects your brush strokes; use the editor's eraser or undo for those. If the suggestion marks expire (about 30 minutes after they were last saved) or the session is dropped, Apply Blur refuses and asks you to run Privacy Suggestions again instead of silently leaving those areas unblurred. Faces that are very small, hidden, or turned away can be missed, and license plates and text are not detected, so check the preview and mark anything else with the brush tool.

## Troubleshooting

* **Image Upload Issues:** Verify file format (JPG/PNG) and refresh the page. Chrome generally provides the best compatibility.
* **Image Too Large:** Uploads are limited to 25 MB and 40 megapixels. Resize larger images before uploading.
* **Drawing Tools Not Working:** Some browsers (such as older Firefox or mobile browsers) may have Canvas/WebGL rendering limitations. Try Chrome, Safari, or Edge if issues arise.
* **AI Suggestions Unavailable:** Each YOLO26 model (about 20 MB total) downloads from Ultralytics the first time that suggestion mode is used. That download is the model, not your photo. If suggestions fail, wait a moment and try again, or mark areas manually.
* **Blur Not Applied:** Ensure you have drawn areas or used AI suggestions before clicking "Apply Blur."
* **Slow Response on Large Images:** For very high-resolution uploads, try resizing locally (for example, to 1920×1080) before uploading to improve responsiveness.

## Security & Privacy

* **Where Images Go:** The page opens in your browser. The photo is uploaded to this Hugging Face Space, and blur and YOLO run on the Space server. Hugging Face has the photo because Hugging Face is running the app. It is not sent to any other service.
* **No Analytics:** Gradio and Ultralytics usage analytics are turned off inside the app.
* **Session timer:** The stored image, suggestion marks, and download id use a 30-minute Gradio `time_to_live`, counted from the last time each value was saved. How Gradio deletes state and cached uploads is in [Gradio's resource cleanup guide](https://www.gradio.app/guides/resource-cleanup). Who can request a cached file is in [Gradio's file access guide](https://www.gradio.app/guides/file-access).
* **Space disk:** How long files stay on a Space's disk is [Hugging Face's disk usage guide](https://huggingface.co/docs/hub/spaces-storage). This app does not attach a Storage Bucket.
* **Metadata:** The download is a new PNG without the original photo's EXIF (such as GPS location or camera details). The session copy used for blurring is also a PNG saved without that EXIF.
* **Abuse Limits:** At most 20 queued requests and 25 stored sessions at a time. The 25 MB upload cap, session cap, and output cleanup apply when the Space imports the app, not only under `python app.py`.
* **Upload Checks:** Only real PNG/JPG files are accepted (checked by content, not just extension), up to 25 MB and 40 megapixels. The same size and layer limits are checked again when you blur or run Privacy Suggestions. Blurred outputs are written to a per-process temp folder.
* **Safe Errors:** Error messages never include server file paths or internal details.

For a photo that should never leave your computer, run the app locally. See the GitHub repository.

[GitHub](https://github.com/MRFrazer25/Blur-Tool)
