import cv2
import numpy as np
import scipy as sp
import scipy.linalg as la
import logging
import coloredlogs
import os
from tqdm import tqdm
import math

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class EigenFaceException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class EigenFaceUtils:
    # desired face mask values to be used
    def __init__(self, width=512, height=512, left=np.array([188,188]), right=np.array([324,188]), isColor=True,
                 nEigenFaces=1000, targetPercentage=None, useBuiltin=True):
        # # at least one of them should be provided
        # assert nEigenFaces is not None or targetPercentage is not None
        # whether we should use cv2.PCACompute/cv2.PCACompute2 or our own PCA computation
        # note the builtin methods are much faster and memory efficient if we're only requiring
        # a small subset of all eigenvectors (don't have to allocate the covariance matrix)
        # enabling us to do large scale computation, even with colored image of 512 * 512
        self.w = width
        self.h = height
        self.l = left  # x, y
        self.r = right  # x, y
        self.face_cascade = None
        self.eye_cascade = None
        self.eyeDict = {}
        self.batch = None  # samples
        self.covar = None  # covariances
        self.eigenValues = None
        self.eigenVectors = None
        self.eigenFaces = None
        self.mean = None
        self.isColor = isColor
        self.nEigenFaces = nEigenFaces
        self.targetPercentage = targetPercentage
        self.useBuiltin = useBuiltin
        self.pathList = None
        self.faceDict = {}
        # ! cannot be pickled, rememeber to delete after loading
        try:
            self.face_cascade = cv2.CascadeClassifier("default.xml")
            self.eye_cascade = cv2.CascadeClassifier("eye.xml")
        except Exception as e:
            log.error(f"Cannot load CascadeClassifier {e}")

    def alignFace2Mask(self, face: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        faceVect = left - right
        maskVect = self.l - self.r
        log.info(f"Getting faceVect: {faceVect} and maskVect: {maskVect}")
        faceNorm = np.linalg.norm(faceVect)
        maskNorm = np.linalg.norm(maskVect)
        log.info(f"Getting faceNorm: {faceNorm} and maskNorm: {maskNorm}")
        scale = maskNorm / faceNorm
        log.info(f"Should scale the image to: {scale}")
        faceAngle = np.degrees(np.arctan2(*faceVect))
        maskAngle = np.degrees(np.arctan2(*maskVect))
        angle = maskAngle - faceAngle
        log.info(f"Should rotate the image: {maskAngle} - {faceAngle} = {angle} degrees")
        faceCenter = (left+right)/2
        maskCenter = (self.l+self.r) / 2
        log.info(f"Getting faceCenter: {faceCenter} and maskCenter: {maskCenter}")
        translation = maskCenter - faceCenter
        log.info(f"Should translate the image using: {translation}")

        # if we're scaling up, we should first translate then do the scaling
        # else the image will get cropped
        # and we'd all want to use the larger destination width*height
        if scale > 1:
            M = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]]])
            face = cv2.warpAffine(face, M, (self.w, self.h))
            M = cv2.getRotationMatrix2D(tuple(maskCenter), angle, scale)
            face = cv2.warpAffine(face, M, (self.w, self.h))

            # if we're scaling down, we should first rotate and scale then translate
            # else the image will get cropped
            # and we'd all want to use the larger destination width*height
        else:
            M = cv2.getRotationMatrix2D(tuple(faceCenter), angle, scale)
            face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
            M = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]]])
            face = cv2.warpAffine(face, M, (self.w, self.h))
        return face

    def detectFacesEyes(self, gray: np.ndarray) -> np.ndarray:
        # Detect faces and eyes on a grayscale image
        # rememeber to convert color before calling this
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = np.ndarray((0, 2))
        # assuming only one face here
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            new = self.eye_cascade.detectMultiScale(roi_gray)
            for (dx, dy, dw, dh) in new:
                eyes = np.concatenate([eyes, np.array([[dx+dw/2+x, dy+dh/2+y]])])
        order = np.argsort(eyes[:, 0])  # sort by first column, which is x
        eyes = eyes[order]
        return faces, eyes

    @property
    def grayLen(self):
        return self.w*self.h

    @property
    def colorLen(self):
        return self.grayLen*3

    @property
    def shouldLen(self):
        return self.colorLen if self.isColor else self.grayLen

    @staticmethod
    def equalizeHistColor(img):
        # perform histogram equilization on a colored image
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        log.info(f"Getting # of channels: {len(channels)}")
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        return img

    def getEyes(self, name, img=None):
        # Try getting eye position of an image
        # first check internal dictionary for already loaded txt file
        # then try recognizing the eye
        # ! this function assume you've already loaded the txt files
        # else it will just perform haar cascaded recognition
        # name = os.path.basename(name)  # get file name
        # name = os.path.splitext(name)[0]  # without ext
        if not name in self.eyeDict:
            # cannot find the already processed data in dict
            if self.isColor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            eyes = self.detectFacesEyes(gray)[1]  # we only need the eyes
            self.eyeDict[name] = eyes
            log.info(f"eyeDict updated: {self.eyeDict[name]}")
            return eyes
        else:
            return self.eyeDict[name]

    def getImage(self, name, manual_check=False) -> np.ndarray:
        # the load the image accordingly
        if self.isColor:
            img = cv2.imread(name, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

        # try getting eye position
        basename = os.path.basename(name)  # get file name
        basename = os.path.splitext(basename)[0]  # without ext
        eyes = self.getEyes(basename, img)
        log.info(f"Getting eyes: {eyes}")
        if not len(eyes) == 2:
            log.warning(f"Cannot get two eyes from this image: {name}, {len(eyes)} eyes")
            del self.eyeDict[basename]
            raise EigenFaceException("Bad image")

        # align according to eye position
        dst = self.alignFace2Mask(img, eyes[0], eyes[1])

        # hist equalization
        if self.isColor:
            dst = self.equalizeHistColor(dst)
        else:
            dst = cv2.equalizeHist(dst)

        # should we check every image before/after loading?
        if manual_check:
            cv2.imshow(name, dst)
            cv2.waitKey()
            cv2.destroyWindow(name)
        return dst

    def getImageFull(self, imgname) -> np.ndarray:
        # Load the image specified and check for corresponding txt file to get the eye position from the file
        txtname = f"{os.path.splitext(imgname)[0]}.txt"
        if os.path.isfile(txtname):
            self.updateEyeDictEntry(txtname)

        log.info(f"Loading image: {imgname}")
        return self.getImage(imgname)

    def updateBatchData(self, path="./", ext=".jpg", manual_check=False, append=False) -> np.ndarray:
        # get all image from a path with a specific extension
        # align them, update histogram and add the to self.batch
        # adjust logging level to be quite or not
        prevLevel = coloredlogs.get_level()
        if not manual_check:
            coloredlogs.set_level("WARNING")

        self.pathList = os.listdir(path)
        self.pathList = [os.path.join(path, name) for name in self.pathList if name.endswith(ext)]
        names = self.pathList
        if not append:
            if self.isColor:
                self.batch = np.ndarray((0, self.colorLen))  # assuming color
            else:
                self.batch = np.ndarray((0, self.grayLen))
        bads = []
        for index, name in tqdm(enumerate(names), desc="Processing batch"):
            try:
                dst = self.getImage(name, manual_check)
                flat = dst.flatten()
                flat = np.reshape(flat, (1, len(flat)))
                self.batch = np.concatenate([self.batch, flat])
            except EigenFaceException as e:
                log.warning(e)
                bads.append(index)
        for bad in bads[::-1]:
            del names[bad]

        coloredlogs.set_level(prevLevel)
        log.info(f"Getting {len(names)} names and {self.batch.shape[0]} batch")
        return self.batch

    def updateEyeDict(self, path="./", ext=".eye", manual_check=False) -> dict:
        # get all possible eyes position from the files with a specific extension
        # and add the to self.eyeDict
        prevLevel = coloredlogs.get_level()
        if not manual_check:
            coloredlogs.set_level("WARNING")

        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        log.info(f"Good names: {names}")
        for name in names:
            # iterate through all txt files
            self.updateEyeDictEntry(name)

        # restore the logging level
        coloredlogs.set_level(prevLevel)
        return self.eyeDict

    def updateEyeDictEntry(self, name):
        # update the dictionary but only one entry
        if not os.path.exists(name):
            log.warning(f"Cannot find file: {name} to update eye information, falling back to recognizor")
            return
        with open(name, "r") as f:
            lines = f.readlines()
            log.info(f"Processing: {name}")
            for line in lines:  # actually there should only be one line
                line = line.strip()  # get rid of starting/ending space \n
                # assuming # starting line to be comment
                if line.startswith("#"):  # get rid of comment file
                    log.info(f"Getting comment line: {line}")
                    continue
                coords = line.split()
                name = os.path.basename(name)  # get file name
                name = os.path.splitext(name)[0]  # without ext
                if len(coords) == 4:
                    self.eyeDict[name] = np.reshape(np.array(coords).astype("float64"), [2, 2])
                    order = np.argsort(self.eyeDict[name][:, 0])  # sort by first column, which is x
                    self.eyeDict[name] = self.eyeDict[name][order]
                else:
                    log.error(f"Wrong format for file: {name}, at line: {line}")
        return self.eyeDict[name]

    def updateMean(self):
        assert self.batch is not None
        # get the mean values of all the vectorized faces
        self.mean = np.reshape(np.mean(self.batch, 0), (1, -1))
        log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
        return self.mean

    def updateCovarMatrix(self) -> np.ndarray:
        assert self.batch is not None and self.mean is not None
        log.info(f"Trying to compute the covariance matrix")
        # covariance matrix of all the pixel location: width * height * color
        self.covar = np.cov(np.transpose(self.batch-self.mean))  # subtract mean
        log.info(f"Getting covar of shape: {self.covar.shape}")
        log.info(f"Getting covariance matrix:\n{self.covar}")
        return self.covar

    def updateCovarMatrixSlow(self) -> np.ndarray:
        assert self.batch is not None, "Should get sample batch before computing covariance matrix"
        nSamples = self.batch.shape[0]
        self.covar = np.zeros((nSamples, nSamples))
        for k in tqdm(range(nSamples**2), "Getting covariance matrix"):
            i = k // nSamples
            j = k % nSamples
            linei = self.batch[i]
            linej = self.batch[j]
            # naive!!!
            if self.covar[j][i] != 0:
                self.covar[i][j] = self.covar[j][i]
            else:
                self.covar[i][j] = self.getCovar(linei, linej)

    @staticmethod
    def getCovar(linei, linej) -> np.ndarray:
        # naive
        meani = np.mean(linei)
        meanj = np.mean(linej)
        unbiasedi = linei - meani
        unbiasedj = linej - meanj
        multi = np.dot(unbiasedi, unbiasedj)
        multi /= len(linei) - 1
        return multi


    def updateEigenVs(self) -> np.ndarray:
        assert self.covar is not None

        if self.targetPercentage is not None:
            log.info(f"Begin computing all eigenvalues")
            self.eigenValues = la.eigvalsh(self.covar)
            self.eigenValues = np.sort(self.eigenValues)[::-1]  # this should be sorted
            log.info(f"Getting all eigenvalues:\n{self.eigenValues}\nof shape: {self.eigenValues.shape}")
            self.updatenEigenFaces()

        log.info(f"Begin computing {self.nEigenFaces} eigenvalues/eigenvectors")
        self.eigenValues, self.eigenVectors = sp.sparse.linalg.eigen.eigsh(self.covar, k=self.nEigenFaces)
        log.info(f"Getting {self.nEigenFaces} eigenvalues and eigenvectors with shape {self.eigenVectors.shape}")

        # always needed right?
        self.eigenVectors = np.transpose(self.eigenVectors.astype("float64"))

        # ? probably not neccessary?
        # might already be sorted according to la.eigen.eigs' algorithm
        order = np.argsort(self.eigenValues)[::-1]
        self.eigenValues = self.eigenValues[order]
        self.eigenVectors = self.eigenVectors[order]

        log.info(f"Getting sorted eigenvalues:\n{self.eigenValues}\nof shape: {self.eigenValues.shape}")
        log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")

        return self.eigenValues, self.eigenVectors

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        # rubust method for reverting a flat matrix
        if len(flat.shape) == 2:
            length = flat.shape[1]
        else:
            length = flat.shape[0]
        if length == self.grayLen:
            if self.isColor:
                log.warning("You're reshaping a grayscale image when color is wanted")
            return np.reshape(flat, (self.h, self.w))
        elif length == self.colorLen:
            if not self.isColor:
                log.warning("You're reshaping a color image when grayscale is wanted")
            return np.reshape(flat, (self.h, self.w, 3))
        else:
            raise EigenFaceException(f"Unsupported flat array of length: {length}, should provide {self.grayLen} or {self.colorLen}")

    def uint8unflatten(self, flat):
        # for displaying
        img = self.unflatten(flat)
        return img.astype("uint8")

    def updatenEigenFaces(self) -> int:
        assert self.eigenValues is not None
        # get energy
        self.nEigenFaces = len(self.eigenValues)
        targetValue = np.sum(self.eigenValues) * self.targetPercentage
        accumulation = 0
        for index, value in enumerate(self.eigenValues):
            accumulation += value
            if accumulation > targetValue:
                self.nEigenFaces = index + 1
                log.info(f"For a energy percentage of {self.targetPercentage}, we need {self.nEigenFaces} vectors from {len(self.eigenValues)}")
                break  # current index should be nEigenFaces

        return self.nEigenFaces

    def loadModel(self, modelName):
        # load previous eigenvectors/mean value
        log.info(f"Loading model: {modelName}")
        data = np.load(modelName, allow_pickle=True)
        try:
            self.eigenVectors = data["arr_0"]
        except KeyError as e:
            log.error(f"Cannot load eigenvectors, {e}")
        try:
            self.mean = data["arr_1"]
        except KeyError as e:
            log.error(f"Cannot load mean face data, {e}")
        try:
            self.faceDict = data["arr_2"].item()
        except KeyError as e:
            log.error(f"Cannot load face dict, {e}")
        log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
        log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")
        log.info(f"Getting face dict of length: {len(self.faceDict)}")

    def saveModel(self, modelName):
        log.info(f"Saving model: {modelName}")
        np.savez_compressed(modelName, self.eigenVectors, self.mean, self.faceDict)
        log.info(f"Model: {modelName} saved")

    # ! unused
    def updateEigenFaces(self) -> np.ndarray:
        assert self.eigenVectors is not None
        log.info(f"Computing eigenfaces")
        self.eigenFaces = np.array([self.unflatten(vector) for vector in self.eigenVectors])
        log.info(f"Getting eigenfaces of shape {self.eigenFaces.shape}")
        return self.eigenFaces

    @staticmethod
    def normalizeImg(mean: np.ndarray) -> np.ndarray:
        return ((mean-np.min(mean)) * 255 / (np.max(mean)-np.min(mean))).astype("uint8")

    def drawEigenFaces(self, rows=None, cols=None) -> np.ndarray:
        assert self.eigenFaces is not None
        # get a canvas for previewing the eigenfaces
        faces = self.eigenFaces
        if rows is None or cols is None:
            faceCount = faces.shape[0]
            rows = int(math.sqrt(faceCount))  # truncating
            cols = faceCount // rows
            while rows*cols < faceCount:
                cols += 1
            # has to be enough
        else:
            faceCount = rows * cols
        log.info(f"Getting canvas for rows: {rows}, cols: {cols}, faceCount: {faceCount}")

        if self.isColor:
            canvas = np.zeros((rows * faces.shape[1], cols * faces.shape[2], 3), dtype="uint8")
        else:
            canvas = np.zeros((rows * faces.shape[1], cols * faces.shape[2]), dtype="uint8")

        for index in range(faceCount):
            i = index // cols
            j = index % cols
            canvas[i * faces.shape[1]:(i+1)*faces.shape[1], j * faces.shape[2]:(j+1)*faces.shape[2]] = self.normalizeImg(faces[index])
            log.info(f"Filling EigenFace of {index} at {i}, {j}")

        return canvas

    def getMeanEigen(self) -> np.ndarray:
        assert self.eigenFaces is not None
        faces = self.eigenFaces
        mean = np.mean(faces, 0)
        mean = np.squeeze(mean)
        mean = self.normalizeImg(mean)
        log.info(f"Getting mean eigenface\n{mean}\nof shape: {mean.shape}")
        return mean

    def updateFaceDict(self) -> dict:
        # compute the face dictionary
        assert self.pathList is not None and self.batch is not None and self.eigenVectors is not None and self.mean is not None
        # note that names and vectors in self.batch are linked through index
        assert len(self.pathList) == self.batch.shape[0], f"{len(self.pathList)} != {self.batch.shape[0]}"
        for index in tqdm(range(len(self.pathList)), "FaceDict"):
            name = self.pathList[index]
            flat = self.batch[index]
            flat = np.expand_dims(flat, 0)  # viewed as 1 * (width * height * color)
            flat -= self.mean
            flat = np.transpose(flat)  # (width * height * color) * 1
            # log.info(f"Shape of eigenvectors and flat: {self.eigenVectors.shape}, {flat.shape}")

            # nEigenFace *(width * height * color) matmal (width * height * color) * 1
            weights = np.matmul(self.eigenVectors, flat)  # new data, nEigenFace * 1
            self.faceDict[name] = weights

        log.info(f"Got face dict of length {len(self.faceDict)}")

        return self.faceDict

    def train(self, path, imgext, modelName="model.npz"):
        txtext = ".txt"
        self.updateEyeDict(path, txtext)
        self.updateBatchData(path, imgext)
        if self.useBuiltin:
            if self.targetPercentage is not None:
                log.info(f"Beginning builtin PCACompute2 for all eigenvalues/eigenvectors")
                # ! this is bad, we'll have to compute all eigenvalues/eigenvectors to determine energy percentage
                self.mean, self.eigenVectors, self.eigenValues = cv2.PCACompute2(self.batch, None)
                log.info(f"Getting eigenvalues/eigenvectors: {self.eigenValues}, {self.eigenVectors}")
                self.updatenEigenFaces()
                # ! dangerous, losing smaller eigenvectors (eigenvalues is small)
                self.eigenVectors = self.eigenVectors[0:self.nEigenFaces]
            else:
                log.info(f"Beginning builtin PCACompute for {self.nEigenFaces} eigenvalues/eigenvectors")
                self.mean, self.eigenVectors = cv2.PCACompute(self.batch, None, maxComponents=self.nEigenFaces)
            log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
            log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")
        else:
            self.updateMean()
            self.updateCovarMatrix()
            self.updateEigenVs()

        self.updateFaceDict()
        self.saveModel(modelName)

    def reconstruct(self, img: np.ndarray) -> np.ndarray:
        assert self.eigenVectors is not None and self.mean is not None

        dst = self.unflatten(self.mean).copy()  # mean face
        flat = img.flatten().astype("float64")  # loaded image with double type
        flat = np.expand_dims(flat, 0)  # viewed as 1 * (width * height * color)
        flat -= self.mean  # flatten subtracted with mean face
        flat = np.transpose(flat)  # (width * height * color) * 1
        log.info(f"Shape of eigenvectors and flat: {self.eigenVectors.shape}, {flat.shape}")

        # nEigenFace *(width * height * color) matmal (width * height * color) * 1
        weights = np.matmul(self.eigenVectors, flat)  # new data, nEigenFace * 1
        # getting the most similar eigenface
        eigen = self.unflatten(self.eigenVectors[np.argmax(weights)])
        # getting the most similar face in the database
        minDist = np.core.numeric.Infinity
        minName = ""
        for name in tqdm(self.faceDict.keys(), "Recognizing"):
            faceWeight = self.faceDict[name]
            dist = la.norm(weights-faceWeight)
            # log.info(f"Getting distance: {dist}, name: {name}")
            if dist < minDist:
                minDist = dist
                minName = name
        log.info(f"MOST SIMILAR FACE: {minName} WITH RESULT {minDist}")
        face = self.unflatten(self.mean).copy()  # mean face
        faceFlat = np.matmul(np.transpose(self.eigenVectors), self.faceDict[minName])  # restored
        faceFlat = np.transpose(faceFlat)
        face += self.unflatten(faceFlat)

        # Eigenvectors of real symmetric matrices are orthogonal
        # data has been lost because nEigenFaces is much smaller than the image dimension span
        # which is width * height * color
        # but because we're using PCA (principal components), most of the information will still be retained
        flat = np.matmul(np.transpose(self.eigenVectors), weights)  # restored
        log.info(f"Shape of flat: {flat.shape}")
        flat = np.transpose(flat)
        dst += self.unflatten(flat)
        if self.isColor:
            ori = np.zeros((self.h, self.w, 3))
        else:
            ori = np.zeros((self.h, self.w))
        try:
            txtname = os.path.splitext(minName)[0]
            txtname = f"{txtname}.txt"
            self.updateEyeDictEntry(txtname)
            ori = self.getImage(minName)
            log.info(f"Successfully loaded the original image: {minName}")
        except FileNotFoundError as e:
            log.error(e)
        dst = self.normalizeImg(dst)
        eigen = self.normalizeImg(eigen)
        face = self.normalizeImg(face)
        return dst, eigen, face, ori, minName


def faces(modelName, Emask=None):

    if Emask is None:
        # instantiate new eigenface class
        Emask = EigenFaceUtils()
        # load previous eigenvectors/mean value
        Emask.loadModel(modelName)
    rows = 3
    cols = 4

    Emask.updateEigenFaces()

    mean = Emask.getMeanEigen()
    cv2.imwrite("eigenmean.png", mean)

    # ! Showing only first 12 faces if more is provided
    canvas = Emask.drawEigenFaces(rows, cols)
    cv2.imwrite("eigenfaces.png", canvas)

    # # ! dangerous, the file might get extremely large
    # allfaces = mask.getCanvas()
    # cv2.imwrite("alleigenfaces.png", allfaces)
