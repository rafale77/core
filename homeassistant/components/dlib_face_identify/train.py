"""Component that will help set the Dlib face detect processing."""
import logging
import numpy as np
from easydict import EasyDict as edict
import cv2
import torch

from Retinaface import FaceDetector
from Arcface import Backbone
from joblib import dump, load
from pathlib import Path
import os
from Sklearn_PyTorch import TorchRandomForestClassifier

home = str(Path.home())+"/.homeassistant/"
modeldir = home+"model/"

conf = edict()
conf.model_path = home+'/model/'
conf.log_path = home
conf.save_path = home
conf.input_size = [112, 112]
conf.embedding_size = 512
conf.net_depth = 50
conf.drop_ratio = 0.6
conf.net_mode = 'ir_se'  # or 'ir'
conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conf.batch_size = 5  # irse net depth 50
conf.facebank_path = Path(home+'recogface/')
conf.threshold = 0.5
conf.face_limit = 10        # when inference, at maximum detect 10 faces in one image
conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
face_detector = FaceDetector(weight_path=home+'model/Resnet50_Final.pth', device=conf.device)
arcmodel = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
arcmodel.load_state_dict(torch.load(f'{conf.model_path}/model_ir_se50.pth'))
arcmodel.eval()


def faces_preprocessing(faces):

    faces = faces.permute(0, 3, 1, 2).float()
    faces = faces.div(255).to(conf.device)
    mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=conf.device)
    faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
    return faces


def locate(image):

    dlibrect = []
    (h, w) = image.shape[:2]
    ratio = 2
    image = cv2.resize(image, (int(w/ratio), int(h/ratio)))
    boxes, scores, landmarks = face_detector.detect_faces(image)
    if len(boxes) > 0:
        for box in boxes:
            face = dlib.rectangle(int(box[0]*ratio), int(box[1]*ratio), int(box[2]*ratio), int(box[3])*ratio)
            dlibrect.append(face)
    return [(max(face.top(), 0), min(face.right(), w), min(face.bottom(), h), max(face.left(), 0)) for face in dlibrect]


def main():

#    faces = torch.empty(0).to(conf.device)
#    names = torch.empty(0).to(conf.device)
    faces = []
    names = []
    dir = home+"recogface/faces/"
    train_dir = os.listdir(dir)

    for person in train_dir:
        pix = os.listdir(dir + person)

        for person_img in pix:
            pic = cv2.imread(dir + person + "/" + person_img)
            face = face_detector.detect_align(pic)[0]
            if len(face) == 1:
                with torch.no_grad():
                    face = arcmodel(faces_preprocessing(face))
                    faces.append(face)
                    names.append(person)
            else:
                print(person_img+" can't be used for training")
        # Create and train the SVC classifier
    clf = TorchRandomForestClassifier(nb_trees=100, nb_samples=3, max_depth=5, bootstrap=True)
    clf.fit(faces, names)
    dump(clf, str(Path.home())+'/model.joblibtest')

if __name__ == '__main__':
    main()
