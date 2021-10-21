import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageStat
import os


# model_mse = __import__('model_mse', fromlist=["model", "immeam", "imstd"])
model_gan = __import__('model_gan', fromlist=["model", "immeam", "imstd"])

# model = model_mse.model
# immean = model_mse.immean
# imstd = model_mse.imstd
#
# use_cuda = torch.cuda.device_count() > 0
#
# model.load_state_dict(torch.load("./model_mse.pth"))
# model.eval()
#
# image_path = "./image/"
# filelist = os.listdir(image_path)

model = model_gan.model
immean = model_gan.immean
imstd = model_gan.imstd

use_cuda = torch.cuda.device_count() > 0

#載入模型權重
model.load_state_dict(torch.load("./model_gan.pth"))
model.eval()

image_path = "./extract_result/"
result_path = "./clean_result/"

filelist = os.listdir(image_path)



for index in range(len(filelist)):

    data = Image.open(image_path + filelist[index]).convert('L')
    # data.show() #圖像數據可視化
    w, h = data.size[0], data.size[1]

    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0

    stat = ImageStat.Stat(data)

    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)

    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()

    else:
        pred = model.forward(data)

    filename = os.path.basename(filelist[index])
    filename = filename.split('.')[0]

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    save_image(pred[0], result_path + filename + ".png")
