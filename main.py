import random
import torch
import torch.nn as nn
import numpy as np
import os


from model_utils.model import initialize_model
from model_utils.train import train_val
from model_utils.data import getDataLoader


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)
###############################################


model_name = 'resnet18'
##########################################

num_class = 11
batchSize = 32
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
epoch = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##########################################
filepath = 'food-11_sample'
# filepath = 'food-11'
##########################

#读数据
train_loader = getDataLoader(filepath, 'train', batchSize)
val_loader = getDataLoader(filepath, 'val', batchSize)
no_label_Loader = getDataLoader(filepath,'train_unl', batchSize)


#模型和超参数
model, input_size = initialize_model(model_name, 11, use_pretrained=False)

print(input_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)

save_path = 'model_save/model.pth'

trainpara = {
            "model" : model,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'no_label_Loader': no_label_Loader,
             'optimizer': optimizer,
            'batchSize': batchSize,
             'loss': loss,
             'epoch': epoch,
             'device': device,
             'save_path': save_path,
             'save_acc': True,
             'max_acc': 0.5,
             'val_epoch' : 1,
             'acc_thres' : 0.7,
             'conf_thres' : 0.99,
             'do_semi' : True,
            "pre_path" : None
             }


if __name__ == '__main__':
    train_val(trainpara)