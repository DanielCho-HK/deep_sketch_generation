import torch
import cv2
import numpy as np
import os
from PIL import Image, ImageStat
from torchvision import transforms
from torchvision.utils import save_image
import extraction
import simplification


ex_model = extraction.res_skip()
ex_model.load_state_dict(torch.load('extract.pth'))

ex_model.cuda()
ex_model.eval()

filelists = extraction.loadImages('./input')

with torch.no_grad():
    for imname in filelists:
        src = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)

        rows = int(np.ceil(src.shape[0] / 16)) * 16
        cols = int(np.ceil(src.shape[1] / 16)) * 16

        # manually construct a batch. You can change it based on your usecases.
        patch = np.ones((1, 1, rows, cols), dtype="float32")
        patch[0, 0, 0:src.shape[0], 0:src.shape[1]] = src

        tensor = torch.from_numpy(patch).cuda()
        y = ex_model(tensor)
        print(imname, torch.max(y), torch.min(y))

        yc = y.cpu().numpy()[0, 0, :, :]
        yc[yc > 255] = 255
        yc[yc < 0] = 0

        head, tail = os.path.split(imname)
        cv2.imwrite("./extract_result/" + tail.replace(".jpg", ".png"), yc[0:src.shape[0], 0:src.shape[1]])


model_gan = __import__('model_gan', fromlist=["model", "immeam", "imstd"])
simp_model = model_gan.model
immean = model_gan.immean
imstd = model_gan.imstd

use_cuda = torch.cuda.device_count() > 0

simp_model.load_state_dict(torch.load("./model_gan.pth"))
simp_model.eval()

image_path = "./extract_result/"
result_path = "./clean_result/"

filelist = os.listdir(image_path)

for index in range(len(filelist)):

    data = Image.open(image_path + filelist[index]).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    stat = ImageStat.Stat(data)

    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)

    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
    if use_cuda:
        pred = simp_model.cuda().forward(data.cuda()).float()
    else:
        pred = simp_model.forward(data)

    filename = os.path.basename(filelist[index])
    filename = filename.split('.')[0]

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    save_image(pred[0], result_path + filename + ".png")