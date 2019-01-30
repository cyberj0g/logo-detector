import cv2
import timeit
import numpy as np
import features


def main():
    video_src = -1
    cam = cv2.VideoCapture(video_src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # get train features
    img = cv2.imread('logo_train.png')
    train_features = features.getFeatures(img)
    cur_time = timeit.default_timer()
    frame_number = 0
    scan_fps = 0
    while True:
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        frame_number += 1
        if not frame_number % 100:
            scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
            cur_time = timeit.default_timer()

        region = features.detectFeatures(frame, train_features)

        cv2.putText(frame, f'FPS {scan_fps:.3f}', org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(0, 0, 255))

        if region is not None:
            box = cv2.boxPoints(region)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
