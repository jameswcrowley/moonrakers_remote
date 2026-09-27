import pytest
import numpy as np
import cv2 as cv
import src.recognition as rec


@pytest.fixture
def synthetic_image_and_library():
    # Create two synthetic images with clear features
    img = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(img, (20, 20), (180, 180), 255, -1)
    cv.circle(img, (100, 100), 30, 0, -1)

    lib_img = img.copy()

    sift = cv.SIFT_create()
    kp_lib, des_lib = sift.detectAndCompute(lib_img, None)

    orb = cv.ORB_create()
    kp_lib_orb, des_lib_orb = orb.detectAndCompute(lib_img, None)

    return img, [(kp_lib, des_lib)], [(kp_lib_orb, des_lib_orb)]


def test_match_result_init():
    res = rec.MatchResult(matches_list=[[1, 2]], scores=[10.5])
    assert res.matches_list == [[1, 2]]
    assert res.scores == [10.5]
    assert res.card_ids is None


def test_match_result_best_match_picks_lowest_score():
    res = rec.MatchResult(
        matches_list=[[0] * 5, [0] * 5, [0] * 5],
        scores=[5.0, 1.0, 3.0],
        card_ids=["a", "b", "c"],
    )
    card_id, confidence = res.best_match()
    assert card_id == "b"
    assert 0.0 < confidence <= 1.0


def test_match_result_best_match_ignores_weak_candidates():
    # too few good matches (< MIN_GOOD_MATCHES) should never win, even with a great distance score
    res = rec.MatchResult(matches_list=[[0], [0] * 5], scores=[0.0, 10.0], card_ids=["a", "b"])
    card_id, confidence = res.best_match()
    assert card_id == "b"
    assert confidence > 0.0


def test_match_result_best_match_empty_returns_none():
    res = rec.MatchResult(matches_list=[], scores=[])
    assert res.best_match() == (None, 0.0)


def test_match_result_best_match_no_card_ids_returns_none():
    res = rec.MatchResult(matches_list=[[]], scores=[1.0])
    assert res.best_match() == (None, 0.0)


def test_sift_matcher(synthetic_image_and_library):
    img, sift_lib, _ = synthetic_image_and_library
    matcher = rec.SIFTMatcher()

    result = matcher.match(img, sift_lib)

    assert isinstance(result, rec.MatchResult)
    assert len(result.matches_list) == len(sift_lib)
    assert len(result.scores) == len(sift_lib)


def test_orb_matcher(synthetic_image_and_library):
    img, _, orb_lib = synthetic_image_and_library
    matcher = rec.ORBMatcher()

    result = matcher.match(img, orb_lib)

    assert isinstance(result, rec.MatchResult)
    assert len(result.matches_list) == len(orb_lib)
    assert len(result.scores) == len(orb_lib)


def test_sift_extractor_returns_keypoints_and_descriptors():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(img, (20, 20), (180, 180), 255, -1)

    keypoints, descriptors = rec.sift_extractor(img)

    assert len(keypoints) > 0
    assert descriptors is not None


def test_orb_extractor_returns_keypoints_and_descriptors():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(img, (20, 20), (180, 180), 255, -1)
    cv.circle(img, (100, 100), 30, 0, -1)

    keypoints, descriptors = rec.orb_extractor(img)

    assert len(keypoints) > 0
    assert descriptors is not None


def test_match_against_library_populates_card_ids(synthetic_image_and_library):
    img, _, _ = synthetic_image_and_library

    class FakeCard:
        def __init__(self, keypoints, descriptors):
            self.keypoints = keypoints
            self.descriptors = descriptors

    class FakeLibrary:
        def __init__(self, cards):
            self.cards = cards

    keypoints, descriptors = rec.sift_extractor(img)
    fake_library = FakeLibrary({"card_a": FakeCard(keypoints, descriptors)})

    result = rec.match_against_library(img, fake_library, rec.SIFTMatcher())

    assert result.card_ids == ["card_a"]
    card_id, confidence = result.best_match()
    assert card_id == "card_a"
    assert 0.0 < confidence <= 1.0


def test_best_guess_stabilizer_confirms_after_consecutive_frames():
    stabilizer = rec.BestGuessStabilizer(confirm_after=3, drop_after=3, confidence_threshold=0.2)

    assert stabilizer.update([("a", 0.9)]) == (None, 0.0)
    assert stabilizer.update([("a", 0.9)]) == (None, 0.0)
    assert stabilizer.update([("a", 0.9)]) == ("a", 0.9)


def test_best_guess_stabilizer_requires_truly_consecutive_frames():
    stabilizer = rec.BestGuessStabilizer(confirm_after=3, drop_after=3, confidence_threshold=0.2)

    stabilizer.update([("a", 0.9)])
    stabilizer.update([])  # a low-confidence/miss frame should reset the streak
    stabilizer.update([("a", 0.9)])
    assert stabilizer.update([("a", 0.9)]) == (None, 0.0)
    assert stabilizer.update([("a", 0.9)]) == ("a", 0.9)


def test_best_guess_stabilizer_holds_guess_despite_disagreement():
    stabilizer = rec.BestGuessStabilizer(confirm_after=2, drop_after=3, confidence_threshold=0.2)

    stabilizer.update([("a", 0.9)])
    stabilizer.update([("a", 0.9)])
    assert stabilizer.get_best_guess() == ("a", 0.9)

    # a single disagreeing frame shouldn't immediately displace the held guess
    assert stabilizer.update([("b", 0.9)]) == ("a", 0.9)


def test_best_guess_stabilizer_drops_after_consecutive_misses():
    stabilizer = rec.BestGuessStabilizer(confirm_after=2, drop_after=2, confidence_threshold=0.2)

    stabilizer.update([("a", 0.9)])
    stabilizer.update([("a", 0.9)])
    assert stabilizer.get_best_guess() == ("a", 0.9)

    stabilizer.update([])
    assert stabilizer.update([]) == (None, 0.0)


def test_rolling_average_stabilizer_averages_over_window():
    stabilizer = rec.RollingAverageStabilizer(window_size=4)

    stabilizer.update(("a", 1.0))
    stabilizer.update(("a", 1.0))
    card_id, confidence = stabilizer.update(("a", 1.0))

    assert card_id == "a"
    assert confidence == pytest.approx(0.75)  # 3 observations averaged over a window of 4


def test_rolling_average_stabilizer_decays_after_card_leaves_window():
    stabilizer = rec.RollingAverageStabilizer(window_size=2)

    stabilizer.update(("a", 1.0))
    stabilizer.update(("a", 1.0))
    stabilizer.update((None, 0.0))
    card_id, confidence = stabilizer.update((None, 0.0))

    assert card_id is None
    assert confidence == 0.0


def test_rolling_average_stabilizer_empty_returns_none():
    stabilizer = rec.RollingAverageStabilizer(window_size=4)
    assert stabilizer.get_best_rolling_average() == (None, 0.0)