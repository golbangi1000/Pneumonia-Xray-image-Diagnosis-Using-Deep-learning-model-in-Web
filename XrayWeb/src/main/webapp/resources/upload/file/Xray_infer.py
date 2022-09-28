#%%
import torch 
from pydicom import dcmread
from PIL import Image
import torchvision.transforms as transforms
import os 
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import pydicom
import torch.nn.functional as F
from torch import nn
import torchvision as tv
from torch.autograd import Variable
import matplotlib as mpl
import warnings
import skimage 
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from pytictoc import TicToc

# TicToc object create (for running time check)
t = TicToc()

# running time check start
t.tic() 

# Warning message ignore
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

# Set hyperparameter
# random value for minimum box area threshold of the CNN model.
min_box_area = 10000
# image rescaling ratio
rescale_factor = 4 
gpu_available = True
batch_size = 1

# image min max rescaling function 
def imgMinMaxScaler(img, scale_range):
    warnings.filterwarnings("ignore")
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    img_scaled = np.rint(img_scaled).astype('uint8')
    return img_scaled

def prediction_string(predicted_boxes, confidences):  
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]

# bbox rescaling function
def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x/rescale_factor))
    y = int(round(y/rescale_factor))
    w = int(round(w/rescale_factor))
    h = int(round(h/rescale_factor))
    return [x, y, w, h]

# prediction score calculate function 
def precision(tp, fp, fn):
    return float(tp) / (tp + fp + fn + 1.e-9)

# Intersection Over Union
def IoU(pr, gt):
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU

# box mask extraction
def box_mask(box, shape=1024):
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y+h, x:x+w] = True 
    return mask

# average precision calculate function 
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    if predicted_boxes == [] and target_boxes == []:
        return np.nan
    else:
        if len(predicted_boxes)>0 and target_boxes == []:
            return 0.0
        elif len(target_boxes)>0 and predicted_boxes == []:
            return 0.0
        else:
            thresholds = np.arange(0.4, 0.8, 0.05) 
            predicted_boxes_sorted = list(reversed([b for _, b in sorted(zip(confidences, predicted_boxes), 
                                                                         key=lambda pair: pair[0])]))            
            average_precision = 0.0
            for t in thresholds: 
                tp = 0 
                fp = len(predicted_boxes)
                for box_p in predicted_boxes_sorted: 
                    box_p_msk = box_mask(box_p, shape)
                    for box_t in target_boxes: 
                        box_t_msk = box_mask(box_t, shape) 
                        iou = IoU(box_p_msk, box_t_msk) 
                        if iou>t:
                            tp += 1 
                            fp -= 1 
                            break 
                fn = len(target_boxes) 
                for box_t in target_boxes: 
                    box_t_msk = box_mask(box_t, shape) 
                    for box_p in predicted_boxes_sorted: 
                        box_p_msk = box_mask(box_p, shape) 
                        iou = IoU(box_p_msk, box_t_msk)
                        if iou>t:
                            fn -= 1
                            break
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision

def average_precision_batch(output_batch, pIds, patient_boxes_dict, rescale_factor, shape=1024, return_array=False):
    
    batch_precisions = []
    for msk, pId in zip(output_batch, pIds): 
        target_boxes = patient_boxes_dict[pId] if pId in patient_boxes_dict else []
        if len(target_boxes)>0:
            target_boxes = [[int(round(c/float(rescale_factor))) for c in box_t] for box_t in target_boxes]
        predicted_boxes, confidences = parse_boxes(msk) 
        batch_precisions.append(average_precision_image(predicted_boxes, confidences, target_boxes, shape=shape))
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions))

# model evaluation function
def evaluate(model, dataloader, loss_fn, num_steps, patient_boxes_dict, rescale_factor, shape):
    model.eval()
    losses = []
    precisions = []

    start = time.time()
    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        if i > num_steps:
            break
        input_batch = Variable(input_batch).cuda(non_blocking=True) if gpu_available else Variable(input_batch).float()
        labels_batch = Variable(labels_batch).cuda(non_blocking=True) if gpu_available else Variable(labels_batch).float()

        output_batch = model(input_batch)
        loss = loss_fn(output_batch, labels_batch)
        losses.append(loss.item())

        output_batch = output_batch.data.cpu()
        prec_batch = average_precision_batch(output_batch, pIds_batch, patient_boxes_dict, rescale_factor, shape, return_array=True)
        for p in prec_batch:
            precisions.append(p)
        print('--- Validation batch {} / {}: '.format(i, num_steps))

    metrics_mean = {'loss' : np.nanmean(losses),
                    'precision' : np.nanmean(np.asarray(precisions))}
    metrics_string = "average loss = {:05.7f} ;  ".format(metrics_mean['loss'])
    metrics_string += "average precision = {:05.7f} ;  ".format(metrics_mean['precision'])
    print("- Eval metrics : " + metrics_string)
    delta_time = time.time() - start
    print('  Evaluation run in {:.2f} seconds.'.format(delta_time))
    
    return metrics_mean

# box parsing function  
def parse_boxes(msk, threshold=0.20, connectivity=None):
    msk = msk[0]
    pos = np.zeros(msk.shape)
    pos[msk>threshold] = 1.
    lbl = skimage.measure.label(pos, connectivity=connectivity)
    
    predicted_boxes = []
    confidences = []
    
    for region in skimage.measure.regionprops(lbl):
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
    
        if w*h > min_box_area: 
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)
    
    return predicted_boxes, confidences

# Xray data elastic transform function  
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    image_warped = map_coordinates(image, indices, order=1).reshape(shape)
    
    return image_warped

# test data file path
patients_test = f'{sys.argv[1]}/{sys.argv[2]}'

# data set path
dataSet_path = './'

#Data transform pipeline  
transform = tv.transforms.Compose([tv.transforms.ToTensor()])

# convolution-batch normalization-activation 
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 batch_momentum=0.9, alpha_leaky=0.03):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=batch_momentum)
        self.activivation = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x):
        return self.activivation(self.batch_norm(self.conv(x)))

# convolution-batch normalization-activation 
class conv_t_block(nn.Module):

    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3, bias=True,
                 batch_momentum=0.9, alpha_leaky=0.03):
        super(conv_t_block, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, 
                                         bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=batch_momentum)
        self.activation = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x, output_size):
        return self.activation(self.batch_norm(self.conv_t(x, output_size=output_size)))

# Modified Unet Model definition
class Xray_diag_UNET(nn.Module):

    def __init__(self):
        super(Xray_diag_UNET, self).__init__()
        
        self.down_1 = nn.Sequential(conv_block(in_channels=1, out_channels=64), conv_block(in_channels=64, out_channels=64))
        self.down_2 = nn.Sequential(conv_block(in_channels=64, out_channels=128), conv_block(in_channels=128, out_channels=128))
        self.down_3 = nn.Sequential(conv_block(in_channels=128, out_channels=256), conv_block(in_channels=256, out_channels=256))
        self.down_4 = nn.Sequential(conv_block(in_channels=256, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.down_5 = nn.Sequential(conv_block(in_channels=512, out_channels=512), conv_block(in_channels=512, out_channels=512))
        
        self.middle = nn.Sequential(conv_block(in_channels=512, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.middle_t = conv_t_block(in_channels=512, out_channels=256)
        
        self.up_5 = nn.Sequential(conv_block(in_channels=768, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.up_5_t = conv_t_block(in_channels=512, out_channels=256)
        self.up_4 = nn.Sequential(conv_block(in_channels=768, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.up_4_t = conv_t_block(in_channels=512, out_channels=128)
        self.up_3 = nn.Sequential(conv_block(in_channels=384, out_channels=256), conv_block(in_channels=256, out_channels=256))
        self.up_3_t = conv_t_block(in_channels=256, out_channels=64)
        self.up_2 = nn.Sequential(conv_block(in_channels=192, out_channels=128), conv_block(in_channels=128, out_channels=128))
        self.up_2_t = conv_t_block(in_channels=128, out_channels=32)
        self.up_1 = nn.Sequential(conv_block(in_channels=96, out_channels=64), conv_block(in_channels=64, out_channels=1))
        
    def forward(self, x):
        down1 = self.down_1(x) 
        out = F.max_pool2d(down1, kernel_size=2, stride=2) 
        down2 = self.down_2(out) 
        out = F.max_pool2d(down2, kernel_size=2, stride=2) 
        down3 = self.down_3(out)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) 
        down4 = self.down_4(out)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) 
        down5 = self.down_5(out) 
        out = F.max_pool2d(down5, kernel_size=2, stride=2) 
        
        out = self.middle(out) 
        out = self.middle_t(out, output_size=down5.size())

        out = torch.cat([down5, out], 1) 
        out = self.up_5(out) 
        out = self.up_5_t(out, output_size=down4.size()) 
        out = torch.cat([down4, out], 1) 
        out = self.up_4(out) 
        out = self.up_4_t(out, output_size=down3.size()) 
        out = torch.cat([down3, out], 1) 
        out = self.up_3(out) 
        out = self.up_3_t(out, output_size=down2.size()) 
        out = torch.cat([down2, out], 1) 
        out = self.up_2(out) 
        out = self.up_2_t(out, output_size=down1.size()) 
        out = torch.cat([down1, out], 1) 
        out = self.up_1(out)
        return out

# Unet model load
f = torch.load('./prec.best.pth')

# CPU or GPU select
model = Xray_diag_UNET().cuda() if gpu_available else Xray_diag_UNET()

# model state_dict load
model.load_state_dict(f['state_dict'])  

# dcm data file load 
img = pydicom.dcmread(f"{patients_test}.dcm").pixel_array

# origin image size 
original_image_shape = img.shape[0]

# new image size = origin image size/rescale factor 
image_shape = original_image_shape / rescale_factor

# image_shape int type convert
image_shape = int(image_shape)    

# image resize  
img = resize(img, (image_shape, image_shape), mode='reflect')   

# image min-max scaling 
img = imgMinMaxScaler(img, (0,255))

# image dimension transform
img = np.expand_dims(img, -1)
img = transform(img)
img = img.view(1,1,256,256)


model.eval()
test_batch = img
test_batch = Variable(test_batch).cuda(non_blocking=True) if gpu_available else Variable(test_batch).float()  

# model prediction
predictions_valid = model(test_batch)
sig = nn.Sigmoid().cuda()
predictions_valid = sig(predictions_valid)
predictions_valid = predictions_valid.data.cpu().numpy()
print('Predicted {} validation images.'.format(len(predictions_valid)))
prediction = predictions_valid[0]

# visualizing feature importance and degeneration segmentation 
plt.imshow(img[0][0], cmap=mpl.cm.gist_gray)
plt.imshow(prediction[0], cmap=mpl.cm.jet, alpha=0.45)
plt.savefig(f"{patients_test}.png")
t.toc() # Unet running time

t = TicToc() # start resnet running time check
t.tic() 

# trained model weight file path
PATH = './weights/'

# image load and preprocessing function 
def load_trans_img(img_name, transform=None):
    image = dcmread(f'{img_name}.dcm')
    image = image.pixel_array
    image = image / 255.0
    image = (255*image).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image).convert('RGB')
    image = transform(image)
    return image

# data preprocessing pipe line
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor()])

#%%    
# trained resnet model load
model = torch.load(PATH + 'resmodel_0311.pt')  
# model state dict load
model.load_state_dict(torch.load(PATH + 'resmodel_0311_state_dict.pt'))  
checkpoint = torch.load(PATH + 'all.tar')   
model.load_state_dict(checkpoint['model'])

#%%

# png, test file save path
train_f = f'./{sys.argv[1]}/'

# prection file path
val_paths = os.path.join(train_f+f"{sys.argv[2]}")
columns = ['patientId', 'Target']

# image load and preprocessing 
val_dataset = load_trans_img(val_paths, transform=transform)

# CPU and GPU select
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


correct = 0
total = 0
val_dataset=val_dataset.unsqueeze(0)
images = val_dataset.to(device)

# model prediction
predictions = model(images)
_, predicted = torch.max(predictions, 1)

# prediction label print
print("predicted:" ,int(predicted))

# prediction label save 
with open(f"./{sys.argv[1]}/{sys.argv[2]}.txt", "w") as f:
    if int(predicted) == 0:
        f.write("Patient")
    else:
        f.write("Normal")
t.toc() # resnet running time