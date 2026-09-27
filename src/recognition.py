# code for taking the rectified card images and using feature matching to match them to a card from the library. 

import math

import cv2 as cv
from abc import ABC, abstractmethod

# Rough scale (in descriptor-distance units) over which confidence decays for
# each matcher's good matches; tuned to each descriptor's typical distance range.
SIFT_DISTANCE_SCALE = 150.0
ORB_DISTANCE_SCALE = 32.0
# Ratio-test matches below this count are too weak to trust as a real card match.
MIN_GOOD_MATCHES = 4


class MatchResult:
    def __init__(self, matches_list, scores, card_ids=None, distance_scale=SIFT_DISTANCE_SCALE):
        self.matches_list = matches_list
        self.scores = scores  # mean descriptor distance of each candidate's good (ratio-test) matches
        self.card_ids = card_ids
        self.distance_scale = distance_scale

    def confidences(self):
        """Return one confidence per candidate, in (0, 1].

        A raw distance-sum score saturates near 0 once real descriptor
        distances are summed over several matches, and rewards candidates
        with only a couple of matches. Instead this blends two normalized
        terms: an exponential decay of the mean match distance (tighter
        matches score higher) and a saturating term for the match count
        (more matches score higher, up to a point). Candidates with fewer
        than MIN_GOOD_MATCHES are treated as no match.
        """
        confidences = []
        for matches, score in zip(self.matches_list, self.scores):
            count = len(matches)
            if count < MIN_GOOD_MATCHES:
                confidences.append(0.0)
                continue
            distance_term = math.exp(-score / self.distance_scale)
            count_term = 1.0 - math.exp(-count / MIN_GOOD_MATCHES)
            confidences.append(distance_term * count_term)
        return confidences

    def best_match(self):
        """Return (card_id, confidence) for the most confident candidate, or (None, 0.0) if empty."""
        if not self.scores or self.card_ids is None:
            return None, 0.0
        confidences = self.confidences()
        best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
        return self.card_ids[best_idx], confidences[best_idx]

    def ranked_matches(self, n=None):
        """Return (card_id, confidence) pairs sorted best-match-first, optionally capped at n."""
        if not self.scores or self.card_ids is None:
            return []
        ranked = sorted(zip(self.card_ids, self.confidences()), key=lambda pair: pair[1], reverse=True)
        return ranked[:n] if n else ranked

class BestGuessStabilizer:
    """Holds on to a confirmed card guess across frames instead of flickering
    to whatever the top match happens to be each frame.

    A new candidate must be the top match for `confirm_after` consecutive
    frames before it can replace (or become) the held guess. Once held, that
    guess is kept even while other frames disagree, and is only dropped
    after `drop_after` consecutive frames that don't reconfirm it.
    """

    def __init__(self, confirm_after=5, drop_after=5, confidence_threshold=0.2):
        """
        Args:
            confirm_after (int): Consecutive frames a new top match must repeat before it's held.
            drop_after (int): Consecutive unconfirmed frames after which the held guess is cleared.
            confidence_threshold (float): Minimum top-match confidence to count as a real observation.
        """
        self.confirm_after = confirm_after
        self.drop_after = drop_after
        self.confidence_threshold = confidence_threshold

        self.held_guess = None
        self.held_confidence = 0.0

        self.candidate = None
        self.candidate_streak = 0
        self.misses = 0

    def update(self, matches):
        """Update state from this frame's ranked matches and return the current held guess.

        Args:
            matches (list): (card_id, confidence) pairs for this frame, best first,
                or an empty list/None if nothing was observed.

        Returns:
            tuple: (card_id, confidence) of the held guess, or (None, 0.0) if nothing is held.
        """
        top_id, top_confidence = matches[0] if matches else (None, 0.0)

        if (top_id is None) or (top_confidence < self.confidence_threshold):
            # invalid frame breaks the candidate streak so confirmation truly requires consecutive frames
            self.candidate = None
            self.candidate_streak = 0
            self._register_miss()
            return self.get_best_guess()

        if top_id == self.candidate:
            self.candidate_streak += 1
        else:
            self.candidate = top_id
            self.candidate_streak = 1

        if top_id == self.held_guess:
            self.misses = 0
            self.held_confidence = top_confidence
        elif self.candidate_streak >= self.confirm_after:
            self.held_guess = self.candidate
            self.held_confidence = top_confidence
            self.misses = 0
        else:
            self._register_miss()

        return self.get_best_guess()

    def _register_miss(self):
        """Count a frame that didn't reconfirm the held guess, dropping it past drop_after."""
        if self.held_guess is None:
            return
        self.misses += 1
        if self.misses >= self.drop_after:
            self.held_guess = None
            self.held_confidence = 0.0
            self.misses = 0

    def get_best_guess(self):
        return self.held_guess, self.held_confidence


class RollingAverageStabilizer:
    """Tracks a running average of per-card confidence over the last `window_size`
    frames, one observation pushed per frame, so a card's score fades out
    gradually instead of vanishing the instant it's no longer the top match.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.rolling_observations = [None] * window_size
        self.rolling_confidences = {}

    def update(self, observation):
        """Push this frame's single (card_id, confidence) observation into the window.

        Args:
            observation (tuple or None): (card_id, confidence) for this frame,
                or None/(None, 0.0) if nothing was observed.

        Returns:
            tuple: (card_id, average_confidence) of the best rolling average, or (None, 0.0) if empty.
        """
        card_id, confidence = observation if observation else (None, 0.0)

        removed = self.rolling_observations.pop(0)
        self.rolling_observations.append((card_id, confidence))
        if removed is not None:
            removed_card_id, removed_confidence = removed
            if removed_card_id in self.rolling_confidences:
                self.rolling_confidences[removed_card_id] -= removed_confidence

        if card_id not in self.rolling_confidences:
            self.rolling_confidences[card_id] = 0.0
        self.rolling_confidences[card_id] += confidence

        return self.get_best_rolling_average()

    def get_rolling_average(self, card_id):
        if card_id not in self.rolling_confidences:
            return 0.0
        return self.rolling_confidences[card_id] / self.window_size

    def get_best_rolling_average(self):
        # ignore None and fully-decayed (<=0) entries so a card that has aged out of the window isn't still "best"
        candidates = {cid: conf for cid, conf in self.rolling_confidences.items() if cid is not None and conf > 0.0}
        if not candidates:
            return None, 0.0
        best_card_id = max(candidates, key=lambda cid: candidates[cid])
        return best_card_id, self.get_rolling_average(best_card_id)


class CardMatcher(ABC):
    @abstractmethod
    def match(self, 
              observation,
              library) -> MatchResult:
        pass

class SIFTMatcher(CardMatcher):
    def match(self, observation, library) -> MatchResult:
        matches_list, scores = self.sift_matcher(observation, library)
        return MatchResult(matches_list, scores, distance_scale=SIFT_DISTANCE_SCALE)

    def sift_matcher(self, image, library):
        """
        Perform SIFT feature matching between the input image and a library of images.

        Parameters:
        image (numpy.ndarray): The input image.
        library: object containting card ID's and features to match against.

        Returns:
        list of cv.DMatch: A list of matches for each image in the library.
        """
        sift = cv.SIFT_create()
        matches_list = []
        scores = []
        kp1, des1 = sift.detectAndCompute(image, None)
        bf = cv.BFMatcher(cv.NORM_L2)
        for lib_features in library:
            kp2, des2 = lib_features
            matches = bf.knnMatch(des1, des2, k = 2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            scores.append(sum(m.distance for m in good_matches) / len(good_matches) if good_matches else float("inf"))
            matches_list.append(good_matches)
        return matches_list, scores


class ORBMatcher(CardMatcher):
    def match(self, observation, library) -> MatchResult:
        matches_list, scores = self.orb_matcher(observation, library)
        return MatchResult(matches_list, scores, distance_scale=ORB_DISTANCE_SCALE)

    def orb_matcher(self, image, library):
        """
        Perform ORB feature matching between the input image and a library of images.

        Parameters:
        image (numpy.ndarray): The input image.
        library: object containting card ID's and features to match against.

        Returns:
        list of cv.DMatch: A list of matches for each image in the library.
        """
        orb = cv.ORB_create()
        matches_list = []
        scores = []
        kp1, des1 = orb.detectAndCompute(image, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        for lib_features in library:
            kp2, des2 = lib_features
            matches = bf.knnMatch(des1, des2, k = 2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            scores.append(sum(m.distance for m in good_matches) / len(good_matches) if good_matches else float("inf"))
            matches_list.append(good_matches)
        return matches_list, scores


def sift_extractor(image):
    """Feature extractor suitable for CardLibrary.from_directory(feature_extractor=...)."""
    return cv.SIFT_create().detectAndCompute(image, None)


def orb_extractor(image):
    """Feature extractor suitable for CardLibrary.from_directory(feature_extractor=...)."""
    return cv.ORB_create().detectAndCompute(image, None)


def match_against_library(observation, card_library, matcher: CardMatcher) -> MatchResult:
    """
    Match an observed card image against every card in a CardLibrary.

    Args:
        observation (np.ndarray): The candidate card image (already rectified/warped).
        card_library (library.CardLibrary): Cards with precomputed keypoints/descriptors.
        matcher (CardMatcher): SIFTMatcher() or ORBMatcher(), matching how the
            library's features were extracted.

    Returns:
        MatchResult: Includes card_ids so best_match() can be used directly.
    """
    card_ids = list(card_library.cards.keys())
    features = [(card_library.cards[card_id].keypoints, card_library.cards[card_id].descriptors) for card_id in card_ids]
    result = matcher.match(observation, features)
    result.card_ids = card_ids
    return result


def manual_match(id_list, card_library) -> MatchResult:
    """
    Manually specify the match for an observed card image against a card library.

    Args:
        id_list (list of str): The manually specified list of card IDs to match.
        card_library (library.CardLibrary): Cards with precomputed keypoints/descriptors.

    Returns:
        MatchResult: Includes card_ids so best_match() can be used directly.
    """
    card_ids = list(card_library.cards.keys())
    result = MatchResult()
    result.card_ids = card_ids
    result.scores = [0.0 if card_id in id_list else float("inf") for card_id in card_ids]
    return result