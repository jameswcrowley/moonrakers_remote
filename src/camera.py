import argparse
import cv2 as cv
import numpy as np


MIN_CONTOUR_AREA = 5000

def order_corners(pts):
    """Return corners as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(diff)],
        pts[np.argmax(s)], pts[np.argmax(diff)],
    ], dtype=np.float32)


def transform_frame(frame, pts_src, pts_dst, sizex=None, sizey=None):
    """Warp a frame to the destination points when a quadrilateral is found."""
    if pts_src is None or pts_dst is None:
        return frame

    homography, _ = cv.findHomography(pts_src, pts_dst)
    if homography is None:
        return frame

    sizex = frame.shape[1] if sizex is None else sizex
    sizey = frame.shape[0] if sizey is None else sizey
    return cv.warpPerspective(frame, homography, (sizex, sizey))


def edge_detection_canny(frame, blur_ksize=(17, 17), canny_threshold1=75, canny_threshold2=200):
    """Blur a frame and return its Canny edge map."""
    blurred = cv.GaussianBlur(frame, blur_ksize, 0)
    return cv.Canny(blurred, canny_threshold1, canny_threshold2, apertureSize=5, L2gradient=True)

def edge_detection_thresholding(frame, blur_ksize=(17, 17), threshold1=100, threshold2=200):
    """Blur a frame and return its binary thresholded edge map."""
    blurred = cv.GaussianBlur(frame, blur_ksize, 0)
    return cv.threshold(blurred, threshold1, threshold2, cv.THRESH_BINARY)[1]


def find_contours(canny):
    """Find external contours, ordered from largest to smallest."""
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # changed from RETR_EXTERNAL to RETR_LIST
    return sorted(contours, key=cv.contourArea, reverse=True)


def approximate_contours(contours, max_contours=1, min_area=MIN_CONTOUR_AREA):
    """Return approximated contours to draw, with zero meaning no display limit."""
    approximated = []
    for contour in contours:
        if cv.contourArea(contour) <= min_area:
            continue

        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        area = cv.contourArea(approx)
        rect = cv.minAreaRect(approx)

        (_, _), (w, h), angle = rect

        rectangularity = area / (w * h) if w * h != 0 else 0
        aspect_ratio = w / h if h != 0 else 0

        conditions = (rectangularity > 0.5 and aspect_ratio > 0.3 and aspect_ratio < 2 and cv.isContourConvex(approx) and len(approx) == 4)

        if conditions:
            approximated.append(approx)
        if max_contours and len(approximated) >= max_contours:
            break
        
    return approximated


def find_screen_contour(contours):
    """Choose the largest approximated quadrilateral for perspective correction."""
    return next((contour for contour in contours if len(contour) == 4), None)


def contour_area(pts):
    """Compute the area of a (4, 2) array of corner points."""
    return cv.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))


def add_frame_info(frame, capture):
    """Add the current camera FPS and resolution to the frame."""
    fps = capture.get(cv.CAP_PROP_FPS)
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    text = f"FPS: {fps} Resolution: {width}x{height}"
    cv.putText(frame, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2, cv.LINE_AA)

def add_contour_info(frame, contours):
    """Add the size of the detected contours below the detected contour."""
    y_offset = 20
    for contour in contours:
        area = cv.contourArea(contour)
        location = tuple(contour[0][0])
        location = (location[0], location[1] - y_offset)
        text = f"Contour Area: {area}"
        cv.putText(frame, text, location, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2, cv.LINE_AA)


def process_frame(frame, capture, max_contours, stabilizer = None):
    """Detect, draw, and rectify contours for one camera frame."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = edge_detection_canny(gray)
    contours = approximate_contours(find_contours(canny), max_contours)

    if stabilizer is not None:
        confirmed = stabilizer.update(contours)
        screen_contour = stabilizer.primary(confirmed)
    else:
        confirmed = []
        screen_contour = find_screen_contour(contours)

    transformed = None
    if screen_contour is not None:
        pts_src = np.float32(screen_contour.reshape(4, 2))
        pts_dst = np.float32([
            [0, 0],
            [0, frame.shape[0] - 1],
            [frame.shape[1] - 1, frame.shape[0] - 1],
            [frame.shape[1] - 1, 0],
        ])
        transformed = transform_frame(frame, pts_src, pts_dst)

    is_stable = [stabilizer is not None and stabilizer.matches_any(c, confirmed) for c in contours]
    stable = [c for c, s in zip(contours, is_stable) if s]
    candidates = [c for c, s in zip(contours, is_stable) if not s]
    cv.drawContours(frame, candidates, -1, (100, 255, 255), 5)
    cv.drawContours(frame, stable, -1, (50, 255, 100), 5)
    if screen_contour is not None:
        primary_contour = np.int32(screen_contour).reshape(1, 4, 2)
        cv.drawContours(frame, primary_contour, -1, (0, 150, 0), 5)
    add_contour_info(frame, contours)
    if capture is not None:
        add_frame_info(frame, capture)

    return frame, canny, transformed

def extract_stable_quads(frame, max_contours, stabilizer = None, method = 'canny'):
    """Detect and draw all stable quadrilaterals for one camera frame.
    Args:
        frame (np.ndarray): The input camera frame.
        max_contours (int): Maximum number of contours to consider.
        stabilizer (ContourStabilizer, optional): Stabilizer for tracking stable quads.
        method (str): Edge detection method, either 'canny' or 'threshold'.
    
    Returns:
        stable (list): List of stable quadrilateral contours.
        frame (np.ndarray): The frame with drawn contours.
    """

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if method == 'canny':
        edges = edge_detection_canny(gray)
    elif method == 'threshold':
        _, edges = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
    else:
        raise ValueError(f"Unsupported edge detection method: {method}")
    
    contours = approximate_contours(find_contours(edges), max_contours)

    if stabilizer is not None:
        confirmed = stabilizer.update(contours)
    else:
        confirmed = []

    is_stable = [stabilizer is not None and stabilizer.matches_any(c, confirmed) for c in contours]
    stable = [c for c, s in zip(contours, is_stable) if s]
    candidates = [c for c, s in zip(contours, is_stable) if not s]
    cv.drawContours(frame, candidates, -1, (100, 255, 255), 5)
    cv.drawContours(frame, stable, -1, (50, 255, 100), 5)
    add_contour_info(frame, contours)

    return stable, frame


class ContourTrack:
    """A single tracked quadrilateral with its own confirmation state."""

    def __init__(self, pts):
        self.pts = pts
        self.streak = 1
        self.misses = 0
        self.confirmed = False


class ContourStabilizer:
    """
    Track and stabilize multiple detected quadrilaterals over time.

    Every raw quad found in a frame is matched (by corner proximity) to an
    existing track or starts a new one. Each track accrues its own streak
    and becomes "confirmed" independently, so several stable contours can be
    confirmed at once regardless of size.

    Arguments:
    ----------
        confirm_frames (int): Number of consecutive frames a track must be matched to be confirmed.
        drop_after (int): Number of consecutive missed frames after which a track is dropped.
        match_thresh (float): Max mean corner distance for a detection to match an existing track.
        smoothing (float): Smoothing factor applied to confirmed tracks' positions.

    """
    def __init__(self, confirm_frames=50, drop_after=20, match_thresh=10, smoothing=0.3):
        self.confirm_frames = confirm_frames
        self.drop_after = drop_after
        self.match_thresh = match_thresh
        self.smoothing = smoothing
        self._tracks = []

    def update(self, contours):
        """Match this frame's quads to tracks and return the confirmed tracks' points."""
        candidates = [order_corners(np.array(c).reshape(4, 2).astype(np.float32)) for c in contours]
        unmatched = list(range(len(candidates)))

        for track in self._tracks:
            best_idx, best_dist = None, None
            for idx in unmatched:
                dist = np.mean(np.linalg.norm(candidates[idx] - track.pts, axis=1))
                if dist < self.match_thresh and (best_dist is None or dist < best_dist):
                    best_idx, best_dist = idx, dist

            if best_idx is None:
                track.misses += 1
                continue

            unmatched.remove(best_idx)
            track.misses = 0
            track.streak += 1
            if track.confirmed:
                track.pts = (1 - self.smoothing) * track.pts + self.smoothing * candidates[best_idx]
            else:
                track.pts = candidates[best_idx]
                track.confirmed = track.streak >= self.confirm_frames

        self._tracks = [t for t in self._tracks if t.misses <= self.drop_after]
        self._tracks.extend(ContourTrack(candidates[idx]) for idx in unmatched)

        return [t.pts for t in self._tracks if t.confirmed]

    def primary(self, confirmed_pts):
        """Return the largest confirmed contour, if any."""
        if not confirmed_pts:
            return None
        return max(confirmed_pts, key=contour_area)

    def matches_any(self, contour, confirmed_pts):
        """Check whether a raw contour lines up with any confirmed track."""
        pts = order_corners(np.array(contour).reshape(4, 2).astype(np.float32))
        return any(np.mean(np.linalg.norm(pts - c, axis=1)) < self.match_thresh for c in confirmed_pts)



def parse_arguments():
    """Parse camera and contour display options."""
    parser = argparse.ArgumentParser(description="Display detected board contours from a camera.")
    parser.add_argument("--camera", type=int, default=1, help="Camera device index (default: 1).")
    parser.add_argument(
        "--display-contours",
        type=int,
        default=0,
        metavar="N",
        help="Display the N largest contours; use 0 to display all (default: 0).",
    )
    return parser.parse_args()



def main():
    """Run the camera loop until the user presses q."""
    args = parse_arguments()
    capture = cv.VideoCapture(args.camera)
    if not capture.isOpened():
        print("Cannot open camera")
        return

    # initialize the contour stabilizer:
    stabilizer = ContourStabilizer()

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            #frame, canny, transformed = process_frame(frame, capture, args.display_contours, stabilizer=stabilizer)
            stable, frame = extract_stable_quads(frame, args.display_contours, stabilizer=stabilizer, method='threshold')
            cv.imshow("frame", frame[:, ::-1])
            # cv.imshow("canny", canny)
            # if transformed is not None:
            #     cv.imshow("transformed", transformed)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()