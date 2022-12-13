import cv2 as cv
import matplotlib.pyplot as plt
import os
from operator import itemgetter

def get_test_names(dir_path):
    #assumes folder only contains video files
    return os.listdir(dir_path)

def calc_avg_displacement(vid_data, NUM_PARTS):
    initial=vid_data[0]
    sum_disp = 0
    c=0
    for i in range(len(vid_data)):
        for n in range(NUM_PARTS):
            if initial[n] and vid_data[i][n]:
                c+=1

                x_n = initial[n][0]
                y_n = initial[n][1]

                X_n = vid_data[i][n][0]
                Y_n = vid_data[i][n][1]

                sum_disp += (x_n - X_n)**2 + (y_n - Y_n)**2

    return sum_disp/c


def instantaneous_euclidean_distance(control, test, NUM_FRAMES, NUM_PARTS):
    dist_sum = 0;
    for i in range(NUM_FRAMES):
        for n in range(NUM_PARTS):
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
            break

        #frame = cv.rotate(frame, cv.ROTATE_180) #ONLY needed if raw iPhone data

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

def show_16_vids(data_path, vid_list):
    caps = []
    for vid in vid_list:
        caps.append(cv.VideoCapture(data_path+vid['name']))

    hasFrame = True;
    while hasFrame and cv.waitKey(1) < 0:
        horiz_list=[]
        for i in range(4):
            hasFrame0, frame0 = caps[4*i].read()
            hasFrame1, frame1 = caps[4*i + 1].read()
            hasFrame2, frame2 = caps[4*i + 2].read()
            hasFrame3, frame3 = caps[4*i + 3].read()
            if not (hasFrame0 and hasFrame1 and hasFrame2 and hasFrame3):
                break;
            horiz_list.append(cv.hconcat([hasFrame0,hasFrame1,hasFrame2,hasFrame3]))
        final_grid = cv.vconcat(horiz_list)

        cv.imshow('Please Work', final_grid) #Not quite






def main():
    # Test Constants
    data_path = "/Users/emmawaters/Desktop/Dance/greenTest/"
    vid_names = get_test_names(data_path)
    vids_data = []

    NUM_FRAMES = float('inf')
    NUM_PARTS = 19

    for i in range(len(vid_names)):
        data = generate_data(data_path + vid_names[i])
        avg_r = calc_avg_displacement(data, NUM_PARTS)

        data_dict = {'name': vid_names[i], 'avg_r' : avg_r}
        vids_data.append(data_dict)

        if len(data) < NUM_FRAMES:
            NUM_FRAMES = len(data)

    sorted_by_r = sorted(vids_data, key=itemgetter('avg_r'))
    show_16_vids(data_path, sorted_by_r)

if __name__ == '__main__':
    main()
