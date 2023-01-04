import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from sklearn.decomposition import PCA

def read_video_to_movement(vidpath):
    data = []
    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    cap = cv.VideoCapture(vidpath)

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Background": 14 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"] ]

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        #frame = cv.rotate(frame, cv.ROTATE_180) #ONLY needed if raw iPhone data
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :15, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

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
            points.append((np.array([x, y])) if conf > 0.1 else np.array([None, None]))

        data.append(points)
    return data

#This is an area where more complexitity could be added
def vectorize_movement(movement, num_chunks, NUM_PARTS, NUM_FRAMES):
    move_vec = []
    chunk_size = int(NUM_FRAMES/num_chunks)
    for n in range(num_chunks):
        for joint in range(NUM_PARTS):
            x_pos_disp = 0
            x_neg_disp = 0
            y_pos_disp = 0
            y_neg_disp = 0

            for i in range(chunk_size - 1):
                prev_frame = movement[n*chunk_size + i]
                frame = movement[n*chunk_size + i + 1]

                if prev_frame[joint].all() and frame[joint].all():
                    disp = prev_frame[joint] - frame[joint]

                    if disp[0] > 0:
                        x_pos_disp += disp[0]
                    else:
                        x_neg_disp += disp[0]
                    if disp[1] > 0:
                        y_pos_disp += disp[1]
                    else:
                        y_neg_disp += disp[1]
            move_vec += [x_pos_disp, x_neg_disp, y_pos_disp, y_neg_disp]

    return move_vec

def main():
    # Test Constants
    data_path = "/Users/emmawaters/Desktop/Dance/greenTest/"
    vid_names = os.listdir(data_path)

    NUM_FRAMES = float('inf')
    NUM_PARTS = 15
    NUM_TIME_CHUNKS = 5

    movements_list = []
    data_list = []

    for i in range(len(vid_names)):
        data = read_video_to_movement(data_path + vid_names[i])
        data_list.append(data)
        if len(data) < NUM_FRAMES:
            NUM_FRAMES = len(data)
    
    for i in range(len(vid_names)):     #need to do AFTER determining min num_frames
        movements_list.append(vectorize_movement(data_list[i][:NUM_FRAMES], NUM_TIME_CHUNKS, NUM_PARTS, NUM_FRAMES))

    movements_array = np.array(movements_list)
    print(movements_array.size)

    pca = PCA(n_components=4)
    pca_fit = pca.fit(np.array(movements_array))

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()


if __name__ == '__main__':
    main()