import io
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import app


def _status_text(html_obj):
    value = getattr(html_obj, "value", None)
    if value is None:
        constructor = getattr(html_obj, "constructor_args", None)
        if isinstance(constructor, dict):
            value = constructor.get("value")
        elif isinstance(constructor, (list, tuple)) and constructor:
            value = constructor[0]
    if value is None:
        value = str(html_obj)
    return str(value)


def _solid_image(size=(64, 48), color=(20, 80, 180)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 28, 28], fill=(255, 40, 40))
    return img


def _editor(img, layers=None):
    return {"background": img, "layers": layers or []}


def _brush_layer(size, box=(8, 8, 28, 28)):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle(box, fill=(255, 0, 0, 255))
    return layer


def test_privacy_copy_differs_on_space_and_locally():
    local = app.privacy_notice_html(on_space=False)
    space = app.privacy_notice_html(on_space=True)
    assert "this computer" in local.lower()
    assert "another service" in local.lower()
    assert "hugging face" in space.lower()
    assert "not sent to any other service" in space.lower()
    assert "your own machine" not in space.lower()
    assert "new png" in local.lower() and "new png" in space.lower()
    assert "hugging face" in app.processing_location_line(on_space=True).lower()
    assert "this computer" in app.processing_location_line(on_space=False).lower()
    if not os.environ.get("SPACE_ID"):
        assert "this computer" in app.privacy_notice_html().lower()


def test_blocks_construct():
    assert app.demo is not None
    assert getattr(app.demo, "title", None) == "Blur Tool"
    assert app.demo.launch is app.launch_with_limits
    assert os.path.isdir(app.OUTPUT_DIR)
    assert os.path.basename(app.OUTPUT_DIR).startswith("blur-tool-outputs-")
    assert app._sweeper_thread is not None and app._sweeper_thread.is_alive()


def test_pack_marks_roundtrip():
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[2:10, 3:12] = 1
    packed = app.pack_marks(mask)
    restored = app.unpack_marks(packed, expected_shape=(40, 30))
    assert restored is not None
    assert np.array_equal(restored, mask)
    assert app.unpack_marks(packed, expected_shape=(10, 10)) is None
    assert app.unpack_marks(None) is None


def _assert_png_has_no_exif(source):
    if isinstance(source, (bytes, bytearray)):
        handle = io.BytesIO(source)
    else:
        handle = source
    with Image.open(handle) as img:
        exif = img.getexif()
        assert dict(exif) == {}
        assert dict(exif.get_ifd(0x8825)) == {}
        assert "exif" not in img.info


def test_stored_and_download_pngs_drop_jpeg_exif():
    """GPS and camera tags on a JPEG do not survive the session PNG or the download PNG."""
    img = Image.new("RGB", (64, 48), (20, 80, 180))
    exif = img.getexif()
    exif[271] = "SecretCamera"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (40.0, 26.0, 46.0)
    gps[3] = "W"
    gps[4] = (79.0, 58.0, 56.0)

    handle, path = tempfile.mkstemp(suffix=".jpg", dir=tempfile.gettempdir())
    os.close(handle)
    temp_id = None
    try:
        img.save(path, format="JPEG", exif=exif)
        with Image.open(path) as opened:
            assert opened.getexif().get(271) == "SecretCamera"
            assert opened.getexif().get_ifd(0x8825).get(1) == "N"
            stored = app.encode_stored_image(ImageOps.exif_transpose(opened))
        _assert_png_has_no_exif(stored)

        source = app.decode_stored_image(stored)
        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        mask[4:20, 4:20] = 1
        output, _status, _download, temp_id, _preview, _flag, *_refreshed = app.handle_blur_click(
            _editor(source),
            stored,
            app.pack_marks(mask),
            True,
            None,
            app.DEFAULT_BLUR_STRENGTH,
        )
        assert output is not None
        assert dict(output.getexif()) == {}
        with app._output_files_lock:
            download_path = app._output_files[temp_id]
        _assert_png_has_no_exif(download_path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        if temp_id:
            app.remove_temp_file(temp_id)


def test_editor_preview_caps_long_side():
    img = Image.new("RGB", (2560, 1920), (1, 2, 3))
    preview = app.editor_preview(img)
    assert max(preview.size) == app.MAX_EDITOR_SIDE
    assert abs(preview.size[0] / preview.size[1] - img.size[0] / img.size[1]) < 0.02
    small = Image.new("RGB", (64, 48), (4, 5, 6))
    assert app.editor_preview(small).size == (64, 48)
    screen = app.screen_preview(img)
    assert max(screen.size) == app.MAX_SCREEN_SIDE
    assert screen.mode == "RGB"
    rgba = Image.new("RGBA", (64, 48), (4, 5, 6, 128))
    assert app.screen_preview(rgba).mode == "RGB"


def _update_field(update, name):
    if isinstance(update, dict):
        return update.get(name)
    value = getattr(update, name, None)
    if value is not None:
        return value
    args = getattr(update, "constructor_args", None)
    if isinstance(args, dict):
        return args.get(name)
    return None


def _save_png(img):
    handle, path = tempfile.mkstemp(suffix=".png", dir=tempfile.gettempdir())
    os.close(handle)
    img.save(path, "PNG")
    return path


def test_phone_user_agent_detection():
    assert app.is_phone_user_agent(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
    )
    assert app.is_phone_user_agent(
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Mobile Safari/537.36"
    )
    assert app.is_phone_user_agent(
        "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
    )
    assert not app.is_phone_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129.0.0.0"
    )
    assert not app.is_phone_user_agent("")
    assert not app.is_phone_user_agent(None)


def test_detect_device_layout_picks_phone_or_computer():
    class _Request:
        def __init__(self, ua):
            self.headers = {"user-agent": ua}

    phone_mode, editor_panel, uploaded, status = app.detect_device_layout(
        _Request("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)")
    )
    assert phone_mode == app.MODE_PHONE
    assert _update_field(editor_panel, "visible") is False
    assert "not available on phones" in _status_text(status).lower()

    computer_mode, editor_panel, uploaded, status = app.detect_device_layout(
        _Request("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0.0.0")
    )
    assert computer_mode == app.MODE_COMPUTER
    assert _update_field(editor_panel, "visible") is True
    assert "ready" in _status_text(status).lower()


def test_drawing_limitation_is_stated():
    text = app.PHONE_DRAWING_LIMITATION.lower()
    assert "not available on phones" in text
    assert "privacy suggestions" in text
    how_to = app.how_to_use_markdown()
    assert "not available on phones" in how_to.lower()
    assert app.MODE_PHONE in how_to or "Phone" in how_to


def test_phone_upload_shows_preview_and_skips_editor():
    img = Image.new("RGB", (app.MAX_EDITOR_SIDE * 2, app.MAX_EDITOR_SIDE), (20, 80, 180))
    path = _save_png(img)
    try:
        result = app.handle_file_upload(path, None, app.MODE_PHONE)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    editor_update, status, _download, _output, _temp_id, stored, marks, _preview, flag, uploaded_view, editor_panel, editor_synced = result
    assert "successfully" in _status_text(status).lower()
    assert "computer" in _status_text(status).lower() or "phone" in _status_text(status).lower()
    display = _update_field(uploaded_view, "value")
    assert max(display.size) == app.MAX_SCREEN_SIDE
    assert _update_field(uploaded_view, "visible") is True
    assert _update_field(editor_panel, "visible") is False
    original = app.decode_stored_image(stored)
    assert original.size == img.size
    assert marks is None
    assert flag is False


def test_computer_upload_loads_drawing_editor():
    img = Image.new("RGB", (app.MAX_EDITOR_SIDE * 2, app.MAX_EDITOR_SIDE), (20, 80, 180))
    path = _save_png(img)
    try:
        result = app.handle_file_upload(path, None, app.MODE_COMPUTER)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    editor_update, status, _download, _output, _temp_id, stored, marks, _preview, flag, uploaded_view, editor_panel, editor_synced = result
    assert "successfully" in _status_text(status).lower()
    assert "draw" in _status_text(status).lower()
    editor_img = _update_field(editor_update, "value")
    assert max(editor_img.size) == app.MAX_EDITOR_SIDE
    assert _update_field(editor_panel, "visible") is True
    assert _update_field(uploaded_view, "visible") is False
    assert app.decode_stored_image(stored).size == img.size
    assert marks is None
    assert flag is False


def test_switch_to_computer_loads_editor_after_phone_upload():
    img = _solid_image((80, 60))
    encoded = app.encode_stored_image(img)
    panel, editor, uploaded, status, editor_synced = app.handle_layout_change(app.MODE_COMPUTER, encoded)
    assert _update_field(panel, "visible") is True
    assert _update_field(editor, "visible") is True
    assert _update_field(editor, "value") is not None
    assert _update_field(uploaded, "visible") is False
    assert "draw" in _status_text(status).lower()

    panel, editor, uploaded, status, editor_synced = app.handle_layout_change(app.MODE_PHONE, encoded)
    assert _update_field(panel, "visible") is False
    assert _update_field(uploaded, "visible") is True
    assert "not available on phones" in _status_text(status).lower()


def test_blur_scales_preview_brush_onto_original():
    original = _solid_image((200, 100), color=(8, 16, 32))
    preview = original.resize((100, 50), Image.Resampling.NEAREST)
    encoded = app.encode_stored_image(original)
    result = app.handle_blur_click(
        _editor(preview, [_brush_layer((100, 50), (4, 4, 14, 14))]),
        encoded,
        None,
        False,
        None,
        app.DEFAULT_BLUR_STRENGTH,
    )
    output, status, _download, temp_id, *_rest = result
    assert output is not None
    assert "successfully" in _status_text(status).lower()
    assert output.size == original.size
    out_np = np.array(output.convert("RGB"))
    src_np = np.array(original)
    assert not np.array_equal(out_np[16, 16], src_np[16, 16])
    assert np.array_equal(out_np[90, 180], src_np[90, 180])
    app.remove_temp_file(temp_id)


def test_stored_image_uses_rgb_when_opaque():
    img = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    encoded = app.encode_stored_image(img)
    decoded = app.decode_stored_image(encoded)
    assert decoded.mode == "RGB"
    assert decoded.size == (16, 16)


def test_blur_changes_pixels_at_every_slider_value():
    img = _solid_image((80, 80))
    source = np.array(img.convert("RGB"))
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    for strength in range(app.MIN_BLUR_STRENGTH, app.MAX_BLUR_STRENGTH + 1, 5):
        blurred = app.apply_gaussian_blur(source.copy(), mask, strength)
        assert not np.array_equal(blurred[mask == 1], source[mask == 1]), strength
        assert np.array_equal(blurred[mask == 0], source[mask == 0]), strength


def test_blur_sigma_scales_with_image_size():
    small = app.blur_sigma_for_image(app.MAX_BLUR_STRENGTH, 100, 80)
    large = app.blur_sigma_for_image(app.MAX_BLUR_STRENGTH, 4000, 3000)
    assert small >= 2.0
    assert large > small * 10


def test_validate_requires_original_and_matching_sizes():
    img = _solid_image()
    encoded = app.encode_stored_image(img)

    source, layers, error = app.validate_editor_payload(None, None)
    assert source is None and "expired" in error.lower()

    source, layers, error = app.validate_editor_payload(None, encoded)
    assert error is None
    assert source.size == img.size
    assert layers == []

    other = img.resize((32, 32))
    source, layers, error = app.validate_editor_payload(_editor(other), encoded)
    assert source is None and "match" in error.lower()

    preview = img.resize((32, 24))
    source, layers, error = app.validate_editor_payload(
        _editor(preview, [_brush_layer((32, 24))]), encoded
    )
    assert error is None
    assert source.size == img.size

    too_many = [_brush_layer(img.size) for _ in range(app.MAX_EDITOR_LAYERS + 1)]
    source, layers, error = app.validate_editor_payload(_editor(img, too_many), encoded)
    assert source is None and "layers" in error.lower()

    source, layers, error = app.validate_editor_payload(_editor(img, [_brush_layer(img.size)]), encoded)
    assert error is None
    assert source.size == img.size
    assert len(layers) == 1


def test_layers_to_mask_ignores_mismatched_size():
    img = _solid_image((40, 30))
    good = _brush_layer((40, 30), (0, 0, 10, 10))
    bad = _brush_layer((20, 20), (0, 0, 10, 10))
    mask = app.layers_to_mask([good, bad], img.size)
    assert mask.shape == (30, 40)
    assert mask[5, 5] == 1
    assert int(mask.sum()) == 11 * 11


def test_refuse_blur_when_suggestion_marks_missing():
    img = _solid_image()
    encoded = app.encode_stored_image(img)
    result = app.handle_blur_click(
        _editor(img, [_brush_layer(img.size)]),
        encoded,
        None,
        True,
        None,
        app.DEFAULT_BLUR_STRENGTH,
    )
    output, status, _download, temp_id, preview, flag, *_refreshed = result
    assert output is None
    assert "expired" in _status_text(status).lower()
    assert temp_id is None
    assert flag is True
    assert preview is not None


def test_blur_without_editor_uses_suggestion_marks():
    img = _solid_image((64, 64), color=(8, 16, 32))
    encoded = app.encode_stored_image(img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:28, 8:28] = 1
    result = app.handle_blur_click(
        None,
        encoded,
        app.pack_marks(mask),
        True,
        None,
        app.DEFAULT_BLUR_STRENGTH,
    )
    output, status, _download, temp_id, *_rest = result
    assert output is not None
    assert "successfully" in _status_text(status).lower()
    out_np = np.array(output.convert("RGB"))
    src_np = np.array(img)
    assert not np.array_equal(out_np[mask == 1], src_np[mask == 1])
    assert np.array_equal(out_np[mask == 0], src_np[mask == 0])
    app.remove_temp_file(temp_id)


def test_blur_screen_preview_is_smaller_than_download():
    original = Image.new("RGB", (app.MAX_EDITOR_SIDE * 2, app.MAX_EDITOR_SIDE), (8, 16, 32))
    encoded = app.encode_stored_image(original)
    mask = np.zeros((original.height, original.width), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    result = app.handle_blur_click(
        None, encoded, app.pack_marks(mask), True, None, app.DEFAULT_BLUR_STRENGTH
    )
    output, status, _download, temp_id, *_rest = result
    assert "successfully" in _status_text(status).lower()
    assert max(output.size) == app.MAX_SCREEN_SIDE
    with app._output_files_lock:
        download_path = app._output_files[temp_id]
    with Image.open(download_path) as saved:
        assert saved.size == original.size
    app.remove_temp_file(temp_id)


def test_successful_blur_uses_suggestion_marks():
    img = _solid_image((64, 64), color=(8, 16, 32))
    encoded = app.encode_stored_image(img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:28, 8:28] = 1
    result = app.handle_blur_click(
        _editor(img),
        encoded,
        app.pack_marks(mask),
        True,
        None,
        app.DEFAULT_BLUR_STRENGTH,
    )
    output, status, _download, temp_id, _preview, _flag, *_refreshed = result
    assert output is not None
    assert "successfully" in _status_text(status).lower()
    assert isinstance(temp_id, str)
    out_np = np.array(output.convert("RGB"))
    src_np = np.array(img)
    assert not np.array_equal(out_np[mask == 1], src_np[mask == 1])
    assert np.array_equal(out_np[mask == 0], src_np[mask == 0])
    app.remove_temp_file(temp_id)


def test_blur_rejects_missing_original_instead_of_background_fallback():
    img = _solid_image()
    result = app.handle_blur_click(
        _editor(img, [_brush_layer(img.size)]),
        None,
        None,
        False,
        None,
        app.DEFAULT_BLUR_STRENGTH,
    )
    output, status, *_ = result
    assert output is None
    assert "expired" in _status_text(status).lower() or "upload" in _status_text(status).lower()


def test_clear_marks_clears_pending_flag():
    preview, status, marks, flag = app.handle_clear_marks()
    assert marks is None
    assert flag is False
    assert "removed" in _status_text(status).lower()
    assert preview is not None


def test_upload_stores_compact_bytes_and_clears_marks():
    img = _solid_image()
    handle, path = tempfile.mkstemp(suffix=".png", dir=tempfile.gettempdir())
    os.close(handle)
    try:
        img.save(path, "PNG")
        result = app.handle_file_upload(path, None, app.MODE_PHONE)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    _editor_update, status, _download, _output, temp_id, stored, marks, _preview, flag, _uploaded, _panel, _editor_synced = result
    assert "successfully" in _status_text(status).lower()
    assert isinstance(stored, (bytes, bytearray))
    assert app.decode_stored_image(stored).size == img.size
    assert marks is None
    assert flag is False
    assert temp_id is None


def test_exercise_upload_suggest_blur_on_synthetic_image():
    """Build Blocks and run the real upload / suggest / refuse / success handlers."""
    assert app.demo is not None
    img = _solid_image((96, 72), color=(12, 40, 90))
    handle, path = tempfile.mkstemp(suffix=".png", dir=tempfile.gettempdir())
    os.close(handle)
    try:
        img.save(path, "PNG")
        upload = app.handle_file_upload(path, None, app.MODE_PHONE)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    stored = upload[5]
    assert isinstance(stored, (bytes, bytearray))
    source = app.decode_stored_image(stored)
    editor = _editor(source)

    suggest = app.handle_suggest_click(editor, stored, app.MODE_OBJECTS)
    suggest_status = _status_text(suggest[0]).lower()
    packed_marks = suggest[2]
    pending = suggest[4]
    assert "error loading image" not in suggest_status
    assert "does not match" not in suggest_status
    assert suggest[5] is stored  # Re-saved so the session's 30-minute expiry restarts

    refused = app.handle_blur_click(
        editor, stored, None, True, None, app.DEFAULT_BLUR_STRENGTH
    )
    assert refused[0] is None
    assert "expired" in _status_text(refused[1]).lower()

    if pending is True and packed_marks is not None:
        marks = packed_marks
    else:
        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        mask[8:28, 8:28] = 1
        marks = app.pack_marks(mask)
        pending = True

    success = app.handle_blur_click(
        editor, stored, marks, pending, None, app.DEFAULT_BLUR_STRENGTH
    )
    assert success[0] is not None
    assert "successfully" in _status_text(success[1]).lower()
    app.remove_temp_file(success[3])


def _upload(img, mode):
    handle, path = tempfile.mkstemp(suffix=".png", dir=tempfile.gettempdir())
    os.close(handle)
    try:
        img.save(path, "PNG")
        return app.handle_file_upload(path, None, mode)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def test_upload_and_layout_record_whether_editor_holds_the_upload():
    img = _solid_image()
    assert _upload(img, app.MODE_COMPUTER)[11] is True
    phone = _upload(img, app.MODE_PHONE)
    assert phone[11] is False
    *_rest, synced = app.handle_layout_change(app.MODE_COMPUTER, phone[5])
    assert synced is True
    *_rest, synced = app.handle_layout_change(app.MODE_PHONE, phone[5])
    assert synced == app.gr.skip()


def test_phone_upload_ignores_strokes_left_in_hidden_editor():
    """Strokes drawn on an earlier image must not be blurred onto a new Phone upload."""
    old_image = _solid_image(color=(200, 200, 200))
    new_stored = app.encode_stored_image(_solid_image(color=(10, 10, 10)))
    stale_editor = _editor(old_image, [_brush_layer(old_image.size)])

    # Editor known to hold another image: its strokes are ignored, so nothing is marked
    ignored = app.handle_blur_click(stale_editor, new_stored, None, False, None, app.DEFAULT_BLUR_STRENGTH, False)
    assert ignored[0] is None
    assert "no areas marked" in _status_text(ignored[1]).lower()

    # Suggestion marks still blur on the Phone layout
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[30:40, 40:60] = 1
    blurred = app.handle_blur_click(stale_editor, new_stored, app.pack_marks(mask), True, None,
                                    app.DEFAULT_BLUR_STRENGTH, False)
    assert blurred[0] is not None
    app.remove_temp_file(blurred[3])

    # Unknown sync state (expired or never set) keeps using the strokes rather than dropping them
    kept = app.handle_blur_click(_editor(app.decode_stored_image(new_stored), [_brush_layer((64, 48))]),
                                 new_stored, None, False, None, app.DEFAULT_BLUR_STRENGTH, None)
    assert kept[0] is not None
    app.remove_temp_file(kept[3])


def test_successful_blur_refreshes_session_expiry():
    """Re-saving the original, marks and editor flag restarts their 30-minute time_to_live."""
    img = _solid_image()
    stored = app.encode_stored_image(img)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[8:28, 8:28] = 1
    marks = app.pack_marks(mask)
    result = app.handle_blur_click(_editor(img), stored, marks, True, None, app.DEFAULT_BLUR_STRENGTH, True)
    assert result[0] is not None
    assert result[6] is stored and result[7] is marks and result[8] is True
    app.remove_temp_file(result[3])

    # Failures leave the session data alone
    failed = app.handle_blur_click(_editor(img), None, marks, True, None, app.DEFAULT_BLUR_STRENGTH, True)
    assert failed[6] == app.gr.skip()


def test_non_numeric_blur_strength_is_rejected_cleanly():
    img = _solid_image()
    stored = app.encode_stored_image(img)
    for bad in (None, "abc", [1]):
        result = app.handle_blur_click(_editor(img, [_brush_layer(img.size)]), stored, None, False, None, bad)
        assert result[0] is None
        assert "blur strength must be a number" in _status_text(result[1]).lower()


def test_transparent_palette_and_grayscale_pngs_keep_transparency():
    palette = Image.new("P", (32, 32), 0)
    palette.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    ImageDraw.Draw(palette).rectangle([8, 8, 24, 24], fill=1)
    palette.info["transparency"] = 0
    gray_alpha = Image.new("LA", (32, 32), (128, 0))
    ImageDraw.Draw(gray_alpha).rectangle([8, 8, 24, 24], fill=(200, 255))
    for img in (palette, gray_alpha):
        stored = app.decode_stored_image(app.encode_stored_image(img))
        assert stored.mode == "RGBA"
        assert stored.getpixel((0, 0))[3] == 0      # Transparent corner stays transparent
        assert stored.getpixel((16, 16))[3] == 255  # Drawn square stays opaque


def test_session_cap_leaves_room_for_passive_visitors():
    assert app.MAX_STORED_SESSIONS >= 100
