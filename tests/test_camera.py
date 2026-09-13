import pytest
import numpy as np
import cv2 as cv
import src.camera as camera


def test_order_corners():
    # Unordered rectangle corners
    pts = np.array([[200, 200], [0, 0], [200, 0], [0, 200]], dtype=np.float32)
    ordered = camera.order_corners(pts)

    expected = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
    np.testing.assert_allclose(ordered, expected)


def test_contour_area():
    pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    area = camera.contour_area(pts)
    assert area == pytest.approx(10000.0)


def test_transform_frame_none_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = camera.transform_frame(frame, None, None)
    np.testing.assert_array_equal(res, frame)


def test_edge_detection_canny():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv.rectangle(frame, (20, 20), (80, 80), (255, 255, 255), -1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = camera.edge_detection_canny(gray, blur_ksize=(3, 3))
    assert edges.shape == (100, 100)
    assert np.any(edges > 0)


def test_contour_stabilizer_update_and_primary():
    stabilizer = camera.ContourStabilizer(confirm_frames=2, drop_after=5)

    quad1 = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32).reshape(4, 1, 2)

    # Frame 1: track added but not confirmed yet
    confirmed1 = stabilizer.update([quad1])
    assert len(confirmed1) == 0
    assert stabilizer.primary(confirmed1) is None

    # Frame 2: track reaches confirm_frames (2)
    confirmed2 = stabilizer.update([quad1])
    assert len(confirmed2) == 1
    assert stabilizer.primary(confirmed2) is not None


def test_extract_stable_quads_invalid_method():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported edge detection method"):
        camera.extract_stable_quads(frame, max_contours=1, method="invalid_method")


def test_approximate_contours_filters_small_and_nonquad():
    small_square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32).reshape(4, 1, 2)
    big_square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape(4, 1, 2)
    triangle = np.array([[0, 0], [200, 0], [100, 200]], dtype=np.int32).reshape(3, 1, 2)

    approximated = camera.approximate_contours([small_square, big_square, triangle], max_contours=0, min_area=500)

    assert len(approximated) == 1
    assert len(approximated[0]) == 4


def test_find_screen_contour_returns_first_quad():
    triangle = np.array([[0, 0], [200, 0], [100, 200]], dtype=np.int32).reshape(3, 1, 2)
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape(4, 1, 2)

    assert camera.find_screen_contour([triangle, quad]) is quad
    assert camera.find_screen_contour([triangle]) is None


def test_find_nests_detects_overlapping_duplicates():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape(4, 1, 2)
    duplicate = square.copy()

    nests, outer_nests = camera.find_nests([square, duplicate])

    assert len(nests) == 2
    assert len(outer_nests) == 2


def test_find_nests_no_nests_for_distant_contours():
    square_a = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.int32).reshape(4, 1, 2)
    square_b = np.array([[500, 500], [550, 500], [550, 550], [500, 550]], dtype=np.int32).reshape(4, 1, 2)

    nests, outer_nests = camera.find_nests([square_a, square_b])

    assert nests == []
    assert outer_nests == []


def test_contour_stabilizer_matches_any():
    stabilizer = camera.ContourStabilizer(confirm_frames=1, match_thresh=10)
    quad = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32).reshape(4, 1, 2)

    stabilizer.update([quad])
    confirmed = stabilizer.update([quad])  # second consecutive match confirms the track
    assert stabilizer.matches_any(quad, confirmed) is True

    far_quad = np.array([[500, 500], [540, 500], [540, 540], [500, 540]], dtype=np.int32).reshape(4, 1, 2)
    assert stabilizer.matches_any(far_quad, confirmed) is False


def test_contour_stabilizer_drops_track_after_misses():
    stabilizer = camera.ContourStabilizer(confirm_frames=1, drop_after=1)
    quad = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32).reshape(4, 1, 2)

    stabilizer.update([quad])
    stabilizer.update([])  # miss 1, still within drop_after
    assert len(stabilizer._tracks) == 1

    stabilizer.update([])  # miss 2, exceeds drop_after
    assert len(stabilizer._tracks) == 0


def test_extract_stable_quads_promotes_confirmed_track():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv.rectangle(frame, (20, 20), (150, 150), (255, 255, 255), -1)
    stabilizer = camera.ContourStabilizer(confirm_frames=1, match_thresh=15)

    stable, drawn = camera.extract_stable_quads(frame, max_contours=1, stabilizer=stabilizer, method="canny")

    assert drawn.shape == frame.shape
    assert isinstance(stable, list)
