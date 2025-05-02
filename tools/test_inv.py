import torch
from PIL import Image
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('/a800data1/wangshijie/models/Da_city_res50_inv_1/model_final.pth')
img_path = ('./a800data1/wangshijie/Domain-Adaptive-Faster-RCNN-PyTorch-master/datasets/cityscapes/images/train/frankfurt_000001_077434_leftImg8bit.png')
image = Image.open(img_path)
img_tran = T.Resize((608,1216))(image)
img_to = T.ToTensor()(img_tran)
img_to = img_to.reshape(1,3,608,1216)
img= img_to.to(device)
for name, module in model.named_children():
    print(name)
    # print(module)
    x = module(x)
    print(x.shape)
    # if name == return_layer:
    #     out.append(x.data)
    #     break
print(out[0].shape)

# outputs = model(img)
# output = T.Resize((600,1200))(outputs)
# output = torch.squeeze(output)  
# toPIL = T.ToPILImage()
# output = toPIL(output)
# output.save('../models/pic/Inv_frankfurt_000001_077434_leftImg8bit.jpg')