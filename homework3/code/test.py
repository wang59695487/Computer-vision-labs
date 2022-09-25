#! python

# Example:
# python test.py -i image_0318.jpg -m model.color.npz -c builtin.json -o similar.png

from eigenface import *
import argparse
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def test(imgname, modelName, outputName):

    # instantiate new eigenface class
    mask = EigenFaceUtils()
    # load previous eigenvectors/mean value
    mask.loadModel(modelName)
    txtname = f"{os.path.splitext(imgname)[0]}.txt"
    if os.path.isfile(txtname):
        log.info("Found text file")
        mask.updateEyeDictEntry(txtname)
    else:
        log.warning(f"Cannot find eye text file for test: {txtname}")
    log.info(f"Loading image: {imgname}")
    src = mask.getImage(imgname)
    dst, eigen, face, ori, dbImgName = mask.reconstruct(src)

    imgs = [src, dst, eigen, face, ori]
    imgBaseName = os.path.basename(imgname)
    dbImgBaseName = os.path.basename(dbImgName)
    msgs = [f"Original Test Image\n{imgBaseName}", f"Reconstructed Test Image\n{imgBaseName}", "Most Similar Eigen Face", f"Reconstructed Most Similar\nDatabase Image\n{dbImgBaseName}", f"Original Most Similar\nDatabase Image\n{dbImgBaseName}"]

    for i in range(len(imgs)):
        img = imgs[i]
        msg = msgs[i]
        offset = 20 / 512 * mask.h
        drawOffset = offset
        scale = 1 / 512 * mask.w
        thick = int(4 / 512 * mask.w)
        splitted = msg.split("\n")
        for msg in splitted:
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(img, msg, (int((mask.w-size[0][0])/2), int(drawOffset+size[0][1])), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
            offset = size[0][1]*1.5  # 1.5 line height
            drawOffset += offset

    w = mask.w
    h = mask.h
    if mask.isColor:
        canvas = np.zeros((h, 5*w, 3), dtype="uint8")
    else:
        canvas = np.zeros((h, 5*w), dtype="uint8")
    canvas[:, 0*w:1*w] = src
    canvas[:, 1*w:2*w] = dst
    canvas[:, 2*w:3*w] = eigen
    canvas[:, 3*w:4*w] = face
    canvas[:, 4*w:5*w] = ori

    if outputName is not None:
        log.info(f"Saving output to {outputName}")
        cv2.imwrite(outputName, canvas)
    else:
        log.warning(f"You didn't specify a output file name, the result WILL NOT BE SAVED\nIt's highly recommended to save the result with -o argument since OpenCV can't even draw large window properly...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    test("./Test_DataSet/10.png", "Emodel.npz", args.output)
