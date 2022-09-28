#%%
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable
import matplotlib as mpl
import shutil
import warnings
from matplotlib.patches import Rectangle
import skimage 
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from pytictoc import TicToc

# create TicToc class
t = TicToc()
t.tic() # start time

# ignore warning message
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

# set hyperparameter
gpu_available = True
original_image_shape = 1024
rescale_factor = 4 
batch_size = 16 

# Working directory
datapath = './'
dataSet_path = './'
# %% 
# load CSV file (bbox ground truth, label) 
data_label = pd.read_csv(datapath+'stage_1_train_labels.csv') 
data_label.groupby('Target').size().plot.bar()
data_detailed = pd.read_csv(datapath+'stage_1_detailed_class_info.csv')
data_detailed.groupby('class').size().plot.bar()

# %% 
# patientId,x,y,width,height,Target + class 
filtered_train = pd.concat([data_label, data_detailed.drop(labels=['patientId'], axis=1)], axis=1)
print(filtered_train)

#%%
# print Group size
print('■ Group size: \n\n', filtered_train.groupby(['class', 'Target']).size(),'\n')

# Check Data null value
print('■ Patient Null value: \n\n', filtered_train.loc[filtered_train['Target']==1, ['x', 'y', 'width', 'height']].isnull().any()) 
print('■ Normal Null value: \n\n', filtered_train.loc[filtered_train['Target']==0, ['x', 'y', 'width', 'height']].isnull().all())
 
patientIDs_test = [f[:-4] for f in os.listdir(datapath+'stage_1_test_images/')]
df_test = pd.DataFrame(data={'patientId' : patientIDs_test})

# box area and box describe
filtered_train['box_area'] = filtered_train['width'] * filtered_train['height']
filtered_train['box_area'].describe()

#%%
# random value for minimum box area threshold of the CNN model.
min_box_area = 10000
validation_frac = 0.10 # train set, valid set 비율

#frac은 모든 샘플을 반환, 무작위로 섞음
# frac returns all of sample, shuffle randomly
filtered_train = filtered_train.sample(frac=1, random_state=42)

# remove redundant patientId values
patients = [p_Id for p_Id in filtered_train['patientId'].unique()]

# valid set 2668, train set 24016 
patients_valid = patients[ : int(round(validation_frac*len(patients)))]
patients_train = patients[int(round(validation_frac*len(patients))) : ]

# print patients_train size, patients_valid size
print(len(patients_valid))
print(len(patients_train))

# remove redundant values in test
patients_test = df_test['patientId'].unique()

#%%
# find patient's bounding box
def get_boxes_per_patient(df, patient_id):
    boxes = df.loc[df['patientId']==patient_id][['x', 'y', 'width', 'height']].astype('int').values.tolist()
    return boxes

# n number of boxes for each patients (ex. left side, right side )
# { Patient_id : [x, y, width, height] }
patient_boxes_dict = {}
for patient_id in filtered_train.loc[(filtered_train['Target']==1)]['patientId'].unique().tolist():
    patient_boxes_dict[patient_id] = get_boxes_per_patient(filtered_train, patient_id)
#%%
# image min max rescaling function 
def imgMinMaxScaler(img, scale_range):
    warnings.filterwarnings("ignore")
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    img_scaled = np.rint(img_scaled).astype('uint8')
    return img_scaled

# function for image augmentation
# (reference: https://www.google.com/search?sa=X&source=univ&tbm=isch&q=elastic+transform&fir=Cjkk8oCx61xEJM%252CN4mfouaE6yikhM%252C_%253BhVb8XGW2R_1aQM%252Cfda1zCVYz8MjgM%252C_%253B4sTsVZ87TNbcfM%252CN4mfouaE6yikhM%252C_%253BcSt-IqYNSQHhBM%252CcuW30DyV2qoiCM%252C_%253BiI7xoTV0o3nuIM%252Cwthsz5_ESVR5hM%252C_%253B-17ha5wfjr4JlM%252C_qzKtRWi3TH9iM%252C_%253BbPAl0dBSjJsU6M%252C_GijzrLVCEnkOM%252C_%253BqBiPK31WbTc2UM%252C4vQs72f8UOf4AM%252C_%253BpugX5CgNOzGjMM%252CA4LYwabrURstsM%252C_%253BYfBhCxDXaSB4DM%252CA4LYwabrURstsM%252C_&usg=AI4_-kT6q8Qc4eHL-SgvD2flQO2N7NdYpQ&ved=2ahUKEwjO-cWr6Jf2AhUJiZQKHa42BBsQ7Al6BAg5EEU&biw=1536&bih=722&dpr=1.25#imgrc=Cjkk8oCx61xEJM)
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

# Xray Data preprocessing
class Xray_Dataset(torchDataset):
    def __init__(self, root, datatype, patient_ids, predict, boxes, rescale_factor=1, transform=None, rotation_angle=0, warping=False):
        
        # root: dataset folders
        # datatype: train or test
        # patient_ids: patient id list
        # predict: True:images and target labels return, False: only images
        # boxes: {'patient_id': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        # rescale_factor: image rescale factor in network
        # transform: transformation images, target masks
        # rotation_angle: defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        # warping: augmentation warping to image
        
        self.root = os.path.expanduser(root)
        self.datatype = datatype
        self.patient_ids = patient_ids
        self.predict = predict
        self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping
        self.data_path = self.root + 'stage_1_'+self.datatype+'_images/'
        
    def __getitem__(self, index):
        # get patientId
        patient_id = self.patient_ids[index]
        # read dcm file
        img = pydicom.dcmread(os.path.join(self.data_path, patient_id+'.dcm')).pixel_array
        # original image shape
        original_image_shape = img.shape[0]
        
        # original image shape/rescale factor calcualte this and get image rescaling shape
        image_shape = original_image_shape / self.rescale_factor

        # typecast resize size to int type
        image_shape = int(image_shape)    
        # image resize
        img = resize(img, (image_shape, image_shape), mode='reflect')   
        # image min-max scaling 
        img = imgMinMaxScaler(img, (0,255))
       
        # image augmentation 
        if self.warping:
            img = elastic_transform(img, image_shape*2., image_shape*0.1)
        
        # expand dimension
        img = np.expand_dims(img, -1)

        # image rotation
        if self.rotation_angle>0:
            angle = self.rotation_angle * (2 * np.random.random_sample() - 1)  
            img = tv.transforms.functional.to_pil_image(img)
            img = tv.transforms.functional.rotate(img, angle, resample=PIL.Image.BILINEAR)                      
        
        # apply transform if 
        if self.transform is not None:
            img = self.transform(img)
        
        # run code below is it is not prediction
        if not self.predict:
            target = np.zeros((image_shape, image_shape))
            if patient_id in self.boxes:
                for box in self.boxes[patient_id]:
                    x, y, w, h = box
                    x = int(round(x/rescale_factor))
                    y = int(round(y/rescale_factor))
                    w = int(round(w/rescale_factor))
                    h = int(round(h/rescale_factor))

                    target[y:y+h, x:x+w] = 255 
                    target[target>255] = 255 
            
            target = np.expand_dims(target, -1)   
            target = target.astype('uint8')
            
            #image transform
            if self.rotation_angle>0:
                target = tv.transforms.functional.to_pil_image(target)
                target = tv.transforms.functional.rotate(target, angle, resample=PIL.Image.BILINEAR)
            
            if self.transform is not None:
                target = self.transform(target)
            return img, target, patient_id
        else: 
            return img, patient_id

    def __len__(self):
        return len(self.patient_ids)

# rescaling box area size
min_box_area = int(round(min_box_area / float(rescale_factor**2)))

# data preprocessing
transform = tv.transforms.Compose([tv.transforms.ToTensor()])

dataset_train = Xray_Dataset(root=datapath, datatype='train', patient_ids=patients_train, predict=False, 
                                 boxes=patient_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=3, warping=True)
dataset_valid = Xray_Dataset(root=dataSet_path, datatype='train', patient_ids=patients_valid, predict=False, 
                                 boxes=patient_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=0, warping=False)
dataset_test = Xray_Dataset(root=dataSet_path, datatype='test', patient_ids=patients_test, predict=True, 
                                boxes=None, rescale_factor=rescale_factor, transform=transform,
                                rotation_angle=0, warping=False)

# DataLoader definition
DataLoader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True) 
DataLoader_valid = DataLoader(dataset=dataset_valid,batch_size=batch_size,shuffle=True) 
DataLoader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False)

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

# Modified Unet model definition 
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

# Dice entropy  Loss function 
class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, scores, targets):
        scores_flat = scores.view(-1)
        targets_flat = targets.view(-1)
        return self.loss(scores_flat, targets_flat)


# box mask extraction
def box_mask(box, shape=1024):
    # box: [x, y, w, h] box coordinates
    # shape: image shape
    # returns: mask as binary 2D array
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y+h, x:x+w] = True 
    return mask

# box parsing
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

def prediction_string(predicted_boxes, confidences):
    # predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] predicted boxes 
    # confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    # returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]   

#  Intersection-over-Union (IoU)
def IoU(pr, gt):
    # pr: prediction array 
    # gt: ground truth array 
    # IoU = intersection/union
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU

# precision calculator
def precision(tp, fp, fn):
    # tp: true positives
    # fp: false positives
    # fn: false negatives
    # precision
    return float(tp) / (tp + fp + fn + 1.e-9)

# average precision calculator
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    # predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    # confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    # target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates 
    # shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    # returns: average_precision

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
    # output_batch: cnn model output batch
    # pIds: (list) list of patient IDs contained in the output batch
    # rescale_factor: CNN image rescale factor
    # shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    # returns: average_precision
    
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

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

# pth save function 
def save_checkpoint(state, is_best, metric):
    filename = 'last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, metric+'.best.pth.tar')

# train function 
def train(model, dataloader, optimizer, loss_fn, num_steps, patient_boxes_dict, rescale_factor, shape, save_summary_steps=5):
    model.train()
    total_time = 0
    loss_avg = RunningAverage()
    loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = [], [], []
    start = time.time()        
    
    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        if i > num_steps:
            break
        
        input_batch = Variable(input_batch).cuda(non_blocking=True) if gpu_available else Variable(input_batch).float()
        labels_batch = Variable(labels_batch).cuda(non_blocking=True) if gpu_available else Variable(labels_batch).float()
            
        optimizer.zero_grad()
        output_batch = model(input_batch)

        loss = loss_fn(output_batch, labels_batch)

        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item())
        loss_t_hist_ep.append(loss.item())
        loss_avg_t_hist_ep.append(loss_avg())
    
        if i % save_summary_steps == 0:
            output_batch = output_batch.data.cpu().numpy()
            prec_batch = average_precision_batch(output_batch, pIds_batch, patient_boxes_dict, rescale_factor, shape)
            prec_t_hist_ep.append(prec_batch)
            summary_batch_string = "batch loss = {:05.7f} ;  ".format(loss.item())
            summary_batch_string += "average loss = {:05.7f} ;  ".format(loss_avg())
            summary_batch_string += "batch precision = {:05.7f} ;  ".format(prec_batch)
            print('--- Train batch {} / {}: '.format(i, num_steps) + summary_batch_string)
            delta_time = time.time() - start
            total_time += delta_time
            print('    {} batches processed in {:.2f} seconds'.format(save_summary_steps, delta_time), "total time",total_time)
            start = time.time()

    metrics_string = "average loss = {:05.7f} ;  ".format(loss_avg())
    print("- Train epoch metrics summary: " + metrics_string)
    return loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep

#%%

# evaluation function 
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

# train and evaluation start function
def train_and_evaluate(model, train_dataloader, val_dataloader, lr_init, loss_fn, num_epochs, 
                       num_steps_train, num_steps_eval, patient_boxes_dict, rescale_factor, shape, restore_file=None):

    if restore_file is not None:
        checkpoint = torch.load(restore_file)
        model.load_state_dict(checkpoint['state_dict'])
            
    best_val_loss = 1e+15
    best_val_prec = 0.0
    best_loss_model = None
    best_prec_model = None

    loss_t_history = []
    loss_v_history = []
    loss_avg_t_history = []
    prec_t_history = []
    prec_v_history = []

    # iteration train
    for epoch in range(num_epochs):
        start = time.time()
        
        lr = lr_init * 0.5**float(epoch) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print("Epoch {}/{}. Learning rate = {:05.3f}.".format(epoch + 1, num_epochs, lr))

        
        loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = train(model, train_dataloader, optimizer, loss_fn, 
                                                                   num_steps_train, patient_boxes_dict, rescale_factor, shape)
        loss_avg_t_history += loss_avg_t_hist_ep
        loss_t_history += loss_t_hist_ep
        prec_t_history += prec_t_hist_ep
        

        val_metrics = evaluate(model, val_dataloader, loss_fn, num_steps_eval, patient_boxes_dict, rescale_factor, shape)

        val_loss = val_metrics['loss']
        val_prec = val_metrics['precision']
        
        loss_v_history += len(loss_t_hist_ep) * [val_loss]
        prec_v_history += len(prec_t_hist_ep) * [val_prec]

        is_best_loss = val_loss<=best_val_loss
        is_best_prec = val_prec>=best_val_prec

        # save best model  
        if is_best_loss:
            print("- Found new best loss: {:.4f}".format(val_loss))
            best_val_loss = val_loss
            best_loss_model = model
        if is_best_prec:
            print("- Found new best precision: {:.4f}".format(val_prec))
            best_val_prec = val_prec
            best_prec_model = model
            
        
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_loss,
                         metric='loss')
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_prec,
                         metric='prec')
        
        delta_time = time.time() - start
        print('Epoch run in {:.2f} minutes'.format(delta_time/60.))

    histories = {'loss avg train' : loss_avg_t_history,
                 'loss train' : loss_t_history,
                 'precision train' : prec_t_history,
                 'loss validation' : loss_v_history, 
                 'precision validation' : prec_v_history}
    best_models = {'best loss model' : best_loss_model, # 
                   'best precision model' : best_prec_model}
    
    return histories, best_models

# prediction function
def predict(model, dataloader): 

    model.eval()
    predictions = {}
    for i, (test_batch, pIds) in enumerate(dataloader):
        print('Predicting batch {} / {}.'.format(i+1, len(dataloader)))
    
        test_batch = Variable(test_batch).cuda(non_blocking=True) if gpu_available else Variable(test_batch).float()  
        output_batch = model(test_batch)
        sig = nn.Sigmoid().cuda()
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output
        
    return predictions

# Use GPU
model = Xray_diag_UNET().cuda() if gpu_available else Xray_diag_UNET()
loss_fn = BCEWithLogitsLoss2d().cuda() if gpu_available else BCEWithLogitsLoss2d()
lr_init = 0.5

# hyperparameter settings 
num_epochs = 2
num_steps_train = len(DataLoader_train)
num_steps_eval = len(DataLoader_valid)

shape = int(round(original_image_shape / rescale_factor))

# start train and evaluation
histories, best_models = train_and_evaluate(model, DataLoader_train, DataLoader_valid, lr_init, loss_fn, 
                                            num_epochs, num_steps_train, num_steps_eval, patient_boxes_dict, rescale_factor, shape)

# model performance plotting
plt.plot(range(len(histories['loss train'])), histories['loss train'], color='k', label='loss train')
plt.plot(range(len(histories['loss avg train'])), histories['loss avg train'], color='g', ls='dashed', label='loss avg train')
plt.plot(range(len(histories['loss validation'])), histories['loss validation'], color='r', label='loss validation')
plt.legend()

plt.plot(range(len(histories['precision train'])), histories['precision train'], color='k', label='precision train')
plt.plot(range(len(histories['precision validation'])), histories['precision validation'], color='r', label='precision validation')
plt.legend()
#%%
# best_model prediction
best_model = best_models['best precision model']
dataset_valid = Xray_Dataset(root=dataSet_path, datatype='train', patient_ids=patients_valid, predict=True, 
                                 boxes=None, rescale_factor=rescale_factor, transform=transform)
DataLoader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False) 

predictions_valid = predict(best_model, DataLoader_valid)
print('Predicted {} validation images.'.format(len(predictions_valid)))
#%%

# rescaling box function
def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x/rescale_factor))
    y = int(round(y/rescale_factor))
    w = int(round(w/rescale_factor))
    h = int(round(h/rescale_factor))
    return [x, y, w, h]

# draw box function
def draw_boxes(predicted_boxes, confidences, target_boxes, ax, angle=0):
    if len(predicted_boxes)>0:
        for box, c in zip(predicted_boxes, confidences):
            x, y, w, h = box 
            patch = Rectangle((x,y), w, h, color='red', ls='dashed',
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)
            ax.text(x+w/2., y-5, '{:.2}'.format(c), color='red', size=20, va='center', ha='center')
    if len(target_boxes)>0:
        for box in target_boxes:
            x, y, w, h = box
            patch = Rectangle((x,y), w, h, color='red',  
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)
    
    return ax

# Find best threshold 
best_threshold = None
best_avg_precision_valid = 0.0
thresholds = np.arange(0.28, 0.30, 0.01)
avg_precision_valids = []

for threshold in thresholds:
    precision_valid = []
    for i in range(len(dataset_valid)):
        img, pId = dataset_valid[i]
        target_boxes = [rescale_box_coordinates(box, rescale_factor) for box in patient_boxes_dict[pId]] if pId in patient_boxes_dict else []
        prediction = predictions_valid[pId]
        predicted_boxes, confidences = parse_boxes(prediction, threshold=threshold, connectivity=None)
        avg_precision_img = average_precision_image(predicted_boxes, confidences, target_boxes, shape=img[0].shape[0])
        precision_valid.append(avg_precision_img)
    avg_precision_valid = np.nanmean(precision_valid)
    avg_precision_valids.append(avg_precision_valid)
    print('Threshold: {}, average precision validation: {:03.5f}'.format(threshold, avg_precision_valid))
    if avg_precision_valid>best_avg_precision_valid:
        print('Found new best average precision validation!')
        best_avg_precision_valid = avg_precision_valid
        best_threshold = threshold
plt.plot(thresholds, avg_precision_valids)

# visualizing bbox
for i in range(len(dataset_valid)):
    img, pId = dataset_valid[i]
    target_boxes = [rescale_box_coordinates(box, rescale_factor) for box in patient_boxes_dict[pId]] if pId in patient_boxes_dict else []
    prediction = predictions_valid[pId]
    predicted_boxes, confidences = parse_boxes(prediction, threshold=best_threshold, connectivity=None)
    avg_precision_img = average_precision_image(predicted_boxes, confidences, target_boxes, shape=img[0].shape[0])
    if i%100==0: 
        plt.imshow(img[0], cmap=mpl.cm.gist_gray)
        plt.imshow(prediction[0], cmap=mpl.cm.jet, alpha=0.5)
        draw_boxes(predicted_boxes, confidences, target_boxes, plt.gca())
        print('Prediction mask scale:', prediction[0].min(), '-', prediction[0].max())
        print('Prediction string:', prediction_string(predicted_boxes, confidences))
        print('Ground truth boxes:', target_boxes)
        print('Average precision image: {:05.5f}'.format(avg_precision_img))
        plt.show()

t.toc() # end time