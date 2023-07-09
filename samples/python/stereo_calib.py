# %%
import argparse
from xml.dom.minidom import parse

import cv2
import matplotlib.pyplot as plt
import numpy as np


# %%
def stereo_calib(
    imagelist,
    pattern_size,
    square_size,
    display_corners=False,
    use_calibrated=True,
    show_rectified=True,
):
    if len(imagelist) % 2 != 0:
        print("Error: the image list contains odd (non-even) number of elements")
        return

    maxScale = 2
    imagePoints = [[], []]
    objectPoints = []
    image_size = None
    goodImageList = []

    nimages = int(len(imagelist) / 2)

    j = 0
    for i in range(nimages):
        for k in range(2):
            filename = imagelist[i * 2 + k]
            img = cv2.imread(filename, 0)
            if img is None:
                break
            if image_size is None:
                image_size = img.shape[::-1]
            elif img.shape[::-1] != image_size:
                print(
                    "The image "
                    + filename
                    + " has the size different from the first image size. Skipping the pair"
                )
                break
            found = False
            # corners = imagePoints[k]
            for scale in range(1, maxScale + 1):
                timg = (
                    img
                    if scale == 1
                    else cv2.resize(
                        img,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR_EXACT,
                    )
                )

                found, corners = cv2.findChessboardCorners(
                    timg,
                    pattern_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if found:
                    if scale > 1:
                        corners *= 1.0 / scale
                    break
            if display_corners:
                print(filename)
                cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(cimg, pattern_size, corners, found)
                sf = 640.0 / max(img.shape)
                cimg1 = cv2.resize(
                    cimg, None, fx=sf, fy=sf, interpolation=cv2.INTER_LINEAR_EXACT
                )
                cv2.imshow("corners", cimg1)
                c = cv2.waitKey(500)
                if c == 27 or c == ord("q") or c == ord("Q"):
                    exit(-1)
            else:
                print(".", end="")
            if not found:
                break

            cv2.cornerSubPix(
                img,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01),
            )

            imagePoints[k].append(corners)
        if k == 1:
            goodImageList.append(imagelist[i * 2])
            goodImageList.append(imagelist[i * 2 + 1])
            j += 1

    print(j, "pairs have been successfully detected.")
    nimages = j
    if nimages < 2:
        print("Error: too little pairs to run the calibration")
        return

    objectPoints = np.zeros((np.prod(pattern_size), 3), np.float32)
    objectPoints[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objectPoints *= square_size
    objectPoints = objectPoints.reshape(-1, 3)

    print("Running stereo calibration ...")

    term_crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    flags = (
        cv2.CALIB_FIX_ASPECT_RATIO
        + cv2.CALIB_ZERO_TANGENT_DIST
        + cv2.CALIB_USE_INTRINSIC_GUESS
        + cv2.CALIB_SAME_FOCAL_LENGTH
        + cv2.CALIB_RATIONAL_MODEL
        + cv2.CALIB_FIX_K3
        + cv2.CALIB_FIX_K4
        + cv2.CALIB_FIX_K5
    )

    cameraMatrix1 = cv2.initCameraMatrix2D(
        [objectPoints] * len(imagePoints[0]), imagePoints[0], image_size, 0
    )
    cameraMatrix2 = cv2.initCameraMatrix2D(
        [objectPoints] * len(imagePoints[1]), imagePoints[1], image_size, 0
    )

    cameraMatrix = [cameraMatrix1, cameraMatrix2]
    distCoeffs = np.zeros((2, 1, 14), np.float64)

    (
        rms,
        cameraMatrix[0],
        distCoeffs[0],
        cameraMatrix[1],
        distCoeffs[1],
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        [objectPoints] * len(imagePoints[0]),
        imagePoints[0],
        imagePoints[1],
        cameraMatrix[0],
        distCoeffs[0],
        cameraMatrix[1],
        distCoeffs[1],
        image_size,
        criteria=term_crit,
        flags=flags,
    )

    print("done with RMS error=", rms)

    # CALIBRATION QUALITY CHECK
    # because the output fundamental matrix implicitly
    # includes all the output information,
    # we can check the quality of calibration using the
    # epipolar geometry constraint: m2^t*F*m1=0
    err = 0
    npoints = 0
    lines = [[], []]
    for i in range(nimages):
        npt = len(imagePoints[0][i])
        for k in range(2):
            imagePoints[k][i] = cv2.undistortPoints(
                imagePoints[k][i],
                cameraMatrix[k],
                distCoeffs[k],
                R=None,
                P=cameraMatrix[k],
            )

            lines[k] = cv2.computeCorrespondEpilines(imagePoints[k][i], k + 1, F)
        for j in range(npt):
            errij = abs(
                imagePoints[0][i][j][0][0] * lines[1][j][0][0]
                + imagePoints[0][i][j][0][1] * lines[1][j][0][1]
                + lines[1][j][0][2]
            ) + abs(
                imagePoints[1][i][j][0][0] * lines[0][j][0][0]
                + imagePoints[1][i][j][0][1] * lines[0][j][0][1]
                + lines[0][j][0][2]
            )
            err += errij
        npoints += npt
    print("average epipolar err = ", err / npoints)

    # save intrinsic parameters
    fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if fs.isOpened():
        fs.write("M1", cameraMatrix[0])
        fs.write("D1", distCoeffs[0])
        fs.write("M2", cameraMatrix[1])
        fs.write("D2", distCoeffs[1])
        fs.release()
    else:
        print("Error: can not save the intrinsic parameters")

    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        cameraMatrix[0],
        distCoeffs[0],
        cameraMatrix[1],
        distCoeffs[1],
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
    )

    validRoi = [validRoi1, validRoi2]

    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if fs.isOpened():
        fs.write("R", R)
        fs.write("T", T)
        fs.write("R1", R1)
        fs.write("R2", R2)
        fs.write("P1", P1)
        fs.write("P2", P2)
        fs.write("Q", Q)
        fs.release()
    else:
        print("Error: can not save the extrinsic parameters")

    # OpenCV can handle left-right\n
    # or up-down camera arrangements\n
    isVerticalStereo = abs(P2[1, 3]) > abs(P2[0, 3])

    # COMPUTE AND DISPLAY RECTIFICATION\n
    if not show_rectified:
        return

    # rmap = [[], []]
    # IF BY CALIBRATED (BOUGUET'S METHOD)\n
    if not use_calibrated:
        # use intrinsic parameters of each camera, but\n
        # compute the rectification transformation directly\n
        # from the fundamental matrix\n
        allimgpt = [[], []]
        for k in range(2):
            for i in range(nimages):
                allimgpt[k].extend(imagePoints[k][i])
        F, mask = cv2.findFundamentalMat(
            np.array(allimgpt[0]), np.array(allimgpt[1]), cv2.FM_8POINT, 0, 0
        )
        H1, H2 = cv2.stereoRectifyUncalibrated(
            np.array(allimgpt[0]), np.array(allimgpt[1]), F, image_size, 3
        )

        R1 = np.dot(np.linalg.inv(cameraMatrix[0]), H1, cameraMatrix[0])
        R2 = np.dot(np.linalg.inv(cameraMatrix[1]), H2, cameraMatrix[1])
        P1 = cameraMatrix[0]
        P2 = cameraMatrix[1]

    # Precompute maps for cv2.remap()\n
    rmap = []
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix[0], distCoeffs[0], R1, P1, image_size, cv2.CV_16SC2
    )

    rmap.append([map1, map2])

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix[1], distCoeffs[1], R2, P2, image_size, cv2.CV_16SC2
    )

    rmap.append([map1, map2])

    sf = 300.0 / max(image_size[0], image_size[1])
    w = round(image_size[0] * sf)
    h = round(image_size[1] * sf)
    canvas = np.zeros((h * 2, w, 3), np.uint8)
    canvas_orig = np.zeros((h * 2, w, 3), np.uint8)

    if not isVerticalStereo:
        sf = 600.0 / max(image_size[0], image_size[1])
        w = round(image_size[0] * sf)
        h = round(image_size[1] * sf)
        canvas = np.zeros((h, w * 2, 3), np.uint8)
        canvas_orig = np.zeros((h, w * 2, 3), np.uint8)

    for i in range(nimages):
        for k in range(2):
            img = cv2.imread(goodImageList[i * 2 + k], 0)
            rimg = cv2.remap(img, rmap[k][0], rmap[k][1], cv2.INTER_LINEAR)
            cimg = cv2.cvtColor(rimg, cv2.COLOR_GRAY2BGR)
            gimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            canvasPart = (
                canvas[h * k : h * (k + 1), 0:w]
                if isVerticalStereo
                else canvas[0:h, w * k : w * (k + 1)]
            )
            canvasPart_orig = (
                canvas_orig[h * k : h * (k + 1), 0:w]
                if isVerticalStereo
                else canvas_orig[0:h, w * k : w * (k + 1)]
            )

            cimg = cv2.resize(
                cimg,
                (canvasPart.shape[1], canvasPart.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            gimg = cv2.resize(
                gimg,
                (canvasPart_orig.shape[1], canvasPart_orig.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            canvasPart[:] = cimg
            canvasPart_orig[:] = gimg
            if use_calibrated:
                vroi = (np.asarray(validRoi[k]) * sf).astype(np.int32)
                cv2.rectangle(
                    canvasPart, vroi[:2], vroi[:2] + vroi[2:], (255, 0, 0), 3, 8
                )
                cv2.rectangle(
                    canvasPart_orig, vroi[:2], vroi[:2] + vroi[2:], (255, 0, 0), 3, 8
                )
        if not isVerticalStereo:
            for j in range(0, canvas.shape[0], 16):
                cv2.line(canvas, (0, j), (canvas.shape[1], j), (0, 255, 0), 1, 8)
                cv2.line(
                    canvas_orig, (0, j), (canvas_orig.shape[1], j), (0, 255, 0), 1, 8
                )
        else:
            for j in range(0, canvas.shape[1], 16):
                cv2.line(canvas, (j, 0), (j, canvas.shape[0]), (0, 255, 0), 1, 8)
                cv2.line(
                    canvas_orig, (j, 0), (j, canvas_orig.shape[0]), (0, 255, 0), 1, 8
                )

        cv2.imshow("rectified", canvas)
        c = cv2.waitKey(500)
        if c == 27 or c == ord("q") or c == ord("Q"):
            exit(-1)


def read_string_list(img_xml_path):
    dom = parse(img_xml_path)
    elem = dom.documentElement
    stus = elem.getElementsByTagName("images")
    img_list = stus[0].childNodes[0].nodeValue
    img_list = img_list.split("\n")
    img_list = [i for i in img_list if i != ""]

    return img_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate the cameras and display the rectified results along with the computed disparity images."
    )
    parser.add_argument(
        "--width",
        default=9,
        type=int,
        help="Pattern size width",
    )
    parser.add_argument(
        "--height",
        default=6,
        type=int,
        help="Pattern size height",
    )
    parser.add_argument(
        "-s",
        "--size",
        default=1.0,
        type=float,
        help="Square size",
    )
    parser.add_argument(
        "--no_show_rectified",
        action="store_true",
        default=False,
        help="Do not show rectified",
    )
    parser.add_argument(
        "--no_use_calibrated",
        action="store_true",
        default=False,
        help="Use Calibrated (0) or uncalibrated (1: use stereoCalibrate(), 2: compute fundamental matrix separately) stereo.",
    )
    parser.add_argument(
        "--display_corners",
        action="store_true",
        default=False,
        help="display corners",
    )

    parser.add_argument(
        "-i",
        "--image",
        default="stereo_calib.xml",
        type=str,
        help="Image list XML/YML file",
    )

    args = parser.parse_args()

    show_rectified = not args.no_show_rectified
    use_calibrated = not args.no_use_calibrated
    display_corners = args.display_corners
    image_list_path = args.image
    width = args.width
    height = args.height
    square_size = args.size

    image_list = read_string_list(image_list_path)

    stereo_calib(
        image_list,
        (width, height),
        square_size,
        display_corners,
        use_calibrated,
        show_rectified,
    )
