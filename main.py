import cv2 as cv
import matplotlib.pyplot as plt

def instantaneous_euclidean_distance(control, test, NUM_FRAMES, NUM_POINTS):
    dist_sum = 0;
    for i in range(NUM_FRAMES):
        for n in range(NUM_POINTS):
            if control[i][n] and test[i][n]:
                x_n = control[i][n][0]
                y_n = control[i][n][1]

                X_n = test[i][n][0]
                Y_n = test[i][n][1]

                dist_sum += (x_n - X_n)**2 + (y_n - Y_n)**2
    return dist_sum


def generate_data(vidpath):
    data = []

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    cap = cv.VideoCapture(vidpath)

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frame = cv.rotate(frame, cv.ROTATE_180)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > 0.2 else None)

        data.append(points)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('OpenPose using OpenCV', frame)

    return data

def main():
    # Test Constants
    vi = "drama.mov"
    vii = "shoulders.mov"
    viii = "sinearms.mov"

    base = "shoulders.mov"

    NUM_FRAMES = 80
    NUM_POINTS = 19

    vi_data = generate_data(vi)
    vii_data = generate_data(vii)
    viii_data = generate_data(viii)

    base_data = generate_data(base)

    print(len(vi_data))
    print(len(vii_data))
    print(len(viii_data))

    vi_res = instantaneous_euclidean_distance(base_data, vi_data, NUM_FRAMES, NUM_POINTS)
    vii_res = instantaneous_euclidean_distance(base_data, vii_data, NUM_FRAMES, NUM_POINTS)
    viii_res = instantaneous_euclidean_distance(base_data, viii_data, NUM_FRAMES, NUM_POINTS)

    print("vi: " + str(vi_res))
    print("vii: " + str(vii_res))
    print("viii: " + str(viii_res))


if __name__ == '__main__':
    main()
