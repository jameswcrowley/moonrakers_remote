# code for taking the rectified card images and using feature matching to match them to a card from the library. 

import cv2 as cv
from abc import ABC, abstractmethod

class MatchResult:
    def __init__(self, matches_list, scores, card_ids=None):
        self.matches_list = matches_list
        self.scores = scores
        self.card_ids = card_ids

    def best_match(self):
        """Return (card_id, confidence) for the lowest-distance match, or (None, 0.0) if empty.

        Confidence is derived from the raw BFMatcher distance sum (lower is
        better) via 1 / (1 + score), so it falls in (0, 1].
        """
        if not self.scores or self.card_ids is None:
            return None, 0.0
        best_idx = min(range(len(self.scores)), key=lambda i: self.scores[i])
        confidence = 1.0 / (1.0 + self.scores[best_idx])
        return self.card_ids[best_idx], confidence

class CardMatcher(ABC):
    @abstractmethod
    def match(self, 
              observation,
              library) -> MatchResult:
        pass

class SIFTMatcher(CardMatcher):
    def match(self, observation, library) -> MatchResult:
        matches_list, scores = self.sift_matcher(observation, library)
        return MatchResult(matches_list, scores)

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
            scores.append(sum([m.distance for m in good_matches]))
            matches_list.append(good_matches)
        return matches_list, scores


class ORBMatcher(CardMatcher):
    def match(self, observation, library) -> MatchResult:
        matches_list, scores = self.orb_matcher(observation, library)
        return MatchResult(matches_list, scores)

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
            scores.append(sum([m.distance for m in good_matches]))
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
