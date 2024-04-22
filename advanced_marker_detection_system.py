import cv2
from cv2 import aruco
import numpy as np

class MarkerDetectionSystem:
    def __init__(self, marker_size_cm=18.7):
        self.marker_size_cm = marker_size_cm
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.detected_markers = {}
        self.camera_matrix = None
        self.distortion_coefficients = None

    def calibrate_camera(self, calibration_images, chessboard_size=(7, 6)):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []  
        imgpoints = []  

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_subpix)

        ret, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is not None:
            for i in range(len(ids)):
                M = cv2.moments(corners[i][0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    marker_info = {
                        'ID': ids[i],
                        'Centroid': (cX, cY),
                        'Corners': corners[i][0]
                    }
                    self.detected_markers[ids[i][0]] = marker_info
        return self.detected_markers

    def calculate_distance(self, marker_size_pixels, cap_width):
        return self.marker_size_cm * cap_width / marker_size_pixels

    def draw_markers(self, frame):
        if self.detected_markers:
            for marker_id, marker_info in self.detected_markers.items():
                cX, cY = marker_info['Centroid']
                cv2.circle(frame, (cX, cY), 5, (255, 0, 255), -1)
                marker_size_pixels = np.mean([np.linalg.norm(marker_info['Corners'][j] - marker_info['Corners'][(j + 1) % 4]) for j in range(4)])
                distance_to_marker_cm = self.calculate_distance(marker_size_pixels, frame.shape[1])
                cv2.putText(frame, "Dist. to Marker {}: {:.2f} cm".format(marker_id, distance_to_marker_cm), (cX - 120, cY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    def draw_connections(self, frame):
        marker_ids = list(self.detected_markers.keys())
        if len(marker_ids) >= 2:
            for i in range(len(marker_ids) - 1):
                for j in range(i + 1, len(marker_ids)):
                    id1 = marker_ids[i]
                    id2 = marker_ids[j]
                    dist = np.linalg.norm(np.array(self.detected_markers[id1]['Centroid']) - np.array(self.detected_markers[id2]['Centroid']))
                    cv2.putText(frame, "Dist. between {} and {}: {:.2f} cm".format(id1, id2, dist), (20, 40 + 20 * (len(self.detected_markers) + j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    def visualize_3d_coordinates(self, frame):
        if self.camera_matrix is not None and self.distortion_coefficients is not None:
            axis = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
            for marker_id, marker_info in self.detected_markers.items():
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_info['Corners'], self.marker_size_cm, self.camera_matrix, self.distortion_coefficients)
                frame = cv2.aruco.drawAxis(frame, self.camera_matrix, self.distortion_coefficients, rvec, tvec, 0.1)
        return frame

def main():
    cap = cv2.VideoCapture(0)
    mds = MarkerDetectionSystem()

    while True:
        ret, frame = cap.read()

        detected_markers = mds.detect_markers(frame)
        mds.draw_markers(frame)
        mds.draw_connections(frame)
        frame = mds.visualize_3d_coordinates(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
