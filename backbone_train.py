import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import SimpleITK as sitk
import albumentations as A

import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tensorboardX import SummaryWriter
import argparse

import warnings
warnings.filterwarnings(action='ignore')

# restriction when you have multi gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# Dataset class
class BrainDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.brain_classes = ('normal', 'abnormal')
        self.transform = transform
        
        
    def reshape_image(self, img):
        img = np.squeeze(img)
        img = np.expand_dims(img, axis=2)

        return img

    def windowing(self, input_img, mode):

        if mode == 'hemorrhage':
            windowing_level = 40
            windowing_width = 160

        elif mode == 'fracture': 
            windowing_level = 600
            windowing_width = 2000

        elif mode == 'normal':
            windowing_level = 40
            windowing_width = 80

        density_low = windowing_level - windowing_width/2 # intensity = density
        density_high = density_low + windowing_width

        output_img = (input_img-density_low) / (density_high-density_low)
        output_img[output_img < 0.] = 0.           # windowing range
        output_img[output_img > 1.] = 1.

        return np.array(output_img, dtype='float32')
    
    def load_image(self, img_path):
        img = self.reshape_image(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32'))    
        img = np.concatenate([self.windowing(img, 'hemorrhage'), self.windowing(img, 'fracture'), self.windowing(img, 'normal')], axis=2)
        return img
    
    def __getitem__(self, index):
        img = self.load_image(os.path.join(self.df.loc[index]['path'],
                                           self.df.loc[index]['file_name']))
        gt = self.df.loc[index]['gt']
        
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        sample = {'img': img,
                  'label': gt}
        return sample
    
    
    def __len__(self):
        return len(self.df)
    
# collater for dataloader
def collater(data):
    imgs = [s['img'] for s in data]
    labels = [s['label'] for s in data]
    

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32)
    labels= torch.from_numpy(np.stack(labels,axis=0)).to(torch.int64)
    
    imgs = imgs.permute(0, 3, 1, 2)

    return imgs, labels


def get_args():
    parser = argparse.ArgumentParser('Backbone training')
     
    parser.add_argument('--info', type=str, default='./data/backbone/info.csv')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pth_path', type=str, default='./data/backbone/model/epoch10_valloss0.13.pth')
    
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--log_path', type=str, default="./data/backbone/log/")
    
    args = parser.parse_args()
    return args


def train(opt):    
    
    ## dataset
    info = pd.read_csv(opt.info)

    train_df = info[info['mode'] == 'train']
    valid_df = info[info['mode'] == 'valid']
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    
    
    my_transforms = A.Compose([
        A.HorizontalFlip(),
        A.augmentations.geometric.rotate.Rotate(limit=20)
    ])
    
    train_ds = BrainDataset(train_df, transform=my_transforms)
    valid_ds = BrainDataset(valid_df)
    
    train_loader = DataLoader(dataset=train_ds, batch_size=25, num_workers=4, collate_fn=collater , shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=10, num_workers=4, collate_fn=collater, shuffle=True)
    

    
    ## train argments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = opt.step
    start_epoch = opt.start_epoch
    num_epochs = opt.num_epochs
    writer = SummaryWriter(opt.log_path)
    
    
    ## model
    resnet = torchvision.models.resnet101()
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features=in_features, out_features=2)
    
    if opt.pretrained:
        param = torch.load(opt.pth_path)
        resnet.load_state_dict(param)
    
    resnet.to(device)
    

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=resnet.parameters(), lr= learning_rate, weight_decay=1e-3)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=opt.lr, weight_decay=1e-3)

    min_valid_loss = 0.40754 # default 1

    for epoch in range(start_epoch, num_epochs + start_epoch):
        epoch_loss = []
        epoch_acc = []
        tepoch = tqdm(train_loader, unit="batch")


        for x_train, y_train in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = resnet(x_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()


            acc = (outputs.argmax(dim=1).cpu() == y_train.cpu()).numpy().sum() / len(outputs)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())

            if step % 10 == 0:
                writer.add_scalar('train_acc', acc, step)
                writer.add_scalar('train_loss', loss.item(), step)


            tepoch.set_postfix(loss=loss.item(), accuracy=100. * acc)
            step += 1



        print(f"Train. Epoch: {epoch :d}, Loss: {np.mean(epoch_loss):1.5f}, acc: {np.mean(epoch_acc)*100 :1.5f}%")

        if epoch % opt.val_interval == 0:
            validation_loss = []
            validation_acc = []
            resnet.eval()

            for x_valid, y_valid in valid_loader:
                with torch.no_grad():

                    x_valid = x_valid.to(device)
                    y_valid = y_valid.to(device)

                    outputs = resnet(x_valid)
                    val_loss = criterion(outputs, y_valid)
                    val_acc = (outputs.argmax(dim=1).cpu() == y_valid.cpu()).numpy().sum() / len(outputs)
                    validation_loss.append(val_loss.item())
                    validation_acc.append(val_acc)
                    
            validation_loss = np.mean(validation_loss)
            validation_acc = np.mean(validation_acc)
            print(f"Val. Epoch: {epoch: d} , val_loss: {validation_loss:1.5f}, val_acc: {validation_acc*100}%  \n" )
            writer.add_scalar('val_acc', validation_acc, step)
            writer.add_scalar('val_loss', validation_loss, step)


            if validation_loss < min_valid_loss:
                min_valid_loss = validation_loss
                torch.save(resnet.state_dict(), f'./data/backbone/model/epoch{epoch:d}_valloss{min_valid_loss:.2f}.pth')

            resnet.train()

    writer.close()
    
    
if __name__ == '__main__':
    opt = get_args()
    train(opt)