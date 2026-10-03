import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw

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

    other = img.resize((32, 32))
    source, layers, error = app.validate_editor_payload(_editor(other), encoded)
    assert source is None and "match" in error.lower()

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
    output, status, _download, temp_id, preview, flag = result
    assert output is None
    assert "expired" in _status_text(status).lower()
    assert temp_id is None
    assert flag is True
    assert preview is not None


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
    output, status, _download, temp_id, _preview, _flag = result
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
        result = app.handle_file_upload(path, None)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    _editor_update, status, _download, _output, temp_id, stored, marks, _preview, flag = result
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
        upload = app.handle_file_upload(path, None)
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
