import torch
import torch.nn.functional as F
from torch import nn
from loss import make_inv_heads_loss_evaluator
from maskrcnn_benchmark.layers import GradientScalarLayer
from torchvision import transforms as T
import cv2
from PIL import Image
import numpy as np



class BlockConvINV(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConvINV,self).__init__()
        self.conv1_inv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1_inv1 = nn.BatchNorm2d(out_channels)
        self.conv2_inv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2_inv1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        for l in [self.conv1_inv1, self.conv2_inv1]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)
            nn.init.kaiming_uniform_(l.weight, a=1)
        

    def forward(self, x):
        x = self.conv1_inv1(x)
        x = F.relu(self.bn1_inv1(x))
        # x = self.leaky_relu(self.bn1_inv1(x))
        x = self.conv2_inv1(x)
        x = F.relu(self.bn2_inv1(x))
        return x


class InversionBackbone(nn.Module):
    def __init__(self,in_channels):
        super(InversionBackbone,self).__init__()
        print(in_channels)
        self.up_1_inv1 = nn.ConvTranspose2d(in_channels = in_channels[0], out_channels = in_channels[1],kernel_size = 2,stride =  2)
        self.right_conv_1_inv1 = BlockConvINV(in_channels[0], in_channels[1])

        self.up_2_inv1 = nn.ConvTranspose2d(in_channels = in_channels[1], out_channels = in_channels[2],kernel_size = 2,stride =  2)
        self.right_conv_2_inv1 = BlockConvINV(in_channels[1], in_channels[2])

        self.up_3_inv1 = nn.ConvTranspose2d(in_channels = in_channels[2], out_channels = in_channels[4],kernel_size = 2,stride =  2)
        self.right_conv_3_inv1 = BlockConvINV(in_channels[3], 64)

        self.up_4_inv1 = nn.ConvTranspose2d(in_channels = in_channels[4], out_channels = in_channels[5],kernel_size = 2,stride =  2)

        self.output_inv1 = nn.Conv2d(in_channels[5], 3, 1, 1, 0)

        for l in [self.output_inv1,self.up_1_inv1,self.up_2_inv1,self.up_3_inv1,self.up_4_inv1]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)
            nn.init.kaiming_uniform_(l.weight, a=1)

        

    def forward(self, feature):
        
        x6_up = self.up_1_inv1(feature[3])

        temp = torch.cat((x6_up,feature[2]), dim=1)
        x6 = self.right_conv_1_inv1(temp)
        x7_up = self.up_2_inv1(x6)

        temp = torch.cat((x7_up, feature[1]), dim=1)
        x7 = self.right_conv_2_inv1(temp)
        x8_up = self.up_3_inv1(x7)

        temp = torch.cat((x8_up,feature[0]), dim=1)
        x8 = self.right_conv_3_inv1(temp)
        x9_up = self.up_4_inv1(x8)
        
        output = self.output_inv1(x9_up)

        return output
    
class Harris(nn.Module):
    def __init__(self):
        super(Harris,self).__init__()
        self.conv1_h = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.conv2_h = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.fc1_h = nn.Linear(2500, 512)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((50, 50))


    def forward(self, feature):

        t = F.relu(self.conv1_h(feature))
        t = F.relu(self.conv2_h(t))
        t = self.adaptive_avg_pool(t)
        t = t.permute(0, 2, 3, 1)
        t = t.reshape(len(t), -1)
        t = self.fc1_h(t)
        t = t.reshape(len(t),256,2)
        return t
    

    def extractHarris(imgs):
        templist = list()
        for img in imgs:
            img = img.permute(1,2,0)
            img = img.cpu().numpy()
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            blockSize = 2 
            apertureSize = 3 
            k = 0.04 
            dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
            dst = cv2.dilate(dst, None)
            flattened = dst.flatten()
            k = min(256, len(flattened))  
            partitioned_indices = np.argpartition(-flattened, k)[:k]
            sorted_indices = partitioned_indices[np.argsort(-flattened[partitioned_indices])]
            rows, cols = np.unravel_index(sorted_indices, dst.shape)
            rows = torch.Tensor(rows)
            cols = torch.Tensor(cols)
            newtensor = torch.stack((rows,cols), dim=1)
            templist.append(newtensor)
        temptensors = torch.stack(templist, dim=0)
        temptensors = temptensors.unsqueeze(1)
        templist = list()
    
class ReverseModule(torch.nn.Module):
    def __init__(self, cfg):
        super(ReverseModule, self).__init__()
        self.cfg = cfg.clone()
        self.Harris = Harris()
        self.loss_evaluator = make_inv_heads_loss_evaluator(cfg)
        self.count = 0
        self.re_features_nums = cfg.MODEL.INVERSION.RE_FEATURES_NUMS
        out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        in_channels = [out_channels//2**(i) for i in range(self.re_features_nums+2)]
        self.InversionResnet = InversionBackbone(in_channels)
        self.inv_weight = cfg.MODEL.INVERSION.INVERSION_LOSS_WEIGHT
    
    def forward(self,imgs, feature, iteration = None):
        if not self.training:
            return None
        tag_imgs = imgs

        # VisualizationINV(x, tag_imgs)
        # x = self.Harris(feature)
        x= self.InversionResnet(feature)

        losses = {}
        # VisualizationINV(x, tag_imgs)
        if (iteration == 1):
            rev_loss = self.loss_evaluator(tag_imgs,x) 
            return rev_loss
        if self.training:
            rev_loss = self.loss_evaluator(x,tag_imgs)
            losses["loss_inv_image"] = rev_loss * 0.001
            return losses
    
    def VisualizationINV(self,x, tag_imgs):

        channel_mean = torch.tensor([102.9801, 115.9465, 122.7717])
        channel_std = torch.tensor([1., 1., 1.])

        MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
        STD = [1/std for std in channel_std]

        mean = channel_mean.type_as(x[0])
        std = channel_std.type_as(x[0])
        # x[0] = x[0]* std + mean
        # tag_imgs[0] = tag_imgs[0]* std + mean
        normalizer = T.Normalize(mean=channel_mean, std=channel_std)
        denormalizer = T.Normalize(mean=MEAN, std=STD)
        x = denormalizer(x[0])
        tag_imgs = denormalizer(tag_imgs[0])
        output = x[0]
        mean = [102.9801, 115.9465, 122.7717]
        for i in range(3):
            output[i] = output[i] + mean[i]
        output = output/255
        PIL_x= T.ToPILImage()(output) 
        tag_img = tag_imgs[0]
        for i in range(3):
            tag_img[i] = tag_img[i] + mean[i]
        tag_img = tag_img/255
        PIL_tag_imgs= T.ToPILImage()(tag_img) 
        PIL_x.save("your dir".format(count = self.count))
        PIL_tag_imgs.save("your dir".format(count = self.count))
        self.count += 1
        print ("saved")
    
def build_inv_heads(cfg):
    if cfg.MODEL.INVERSION_BACKBONE_ON:
        return ReverseModule(cfg)
    return []
