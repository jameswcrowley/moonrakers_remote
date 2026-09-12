# code for taking the rectified card images and using feature matching to match them to a card from the library. 

import cv2 as cv
from abc import ABC, abstractmethod

class MatchResult:
    def __init__(self, matches_list, scores):
        self.matches_list = matches_list
        self.scores = scores

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

    