import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import src.camera as camera
import src.library as library
import src.recognition as recognition
import os

# putting it all together: tracking the state of the current cards, including stabilization from the camera, rectification, and matching to the card library.

