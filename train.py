import random
import os
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE
from test import test
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
######################
# params             #
######################

source_image_root = os.path.join('.', 'dataset', 'leaf_source', 'train')
target_image_root = os.path.join('.', 'dataset', 'leaf_target', 'train')
model_root = 'model'
cuda = torch.cuda.is_available()
if cuda:
    torch.backends.cudnn.benchmark = True

lr = 1e-2
batch_size = 32
image_size = [128, 256] # height, width
n_epoch = 100
step_decay_weight = 0.95
lr_decay_step = 20000
active_domain_loss_step = 10000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################
# load data           #
#######################



# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.images = os.listdir(data_root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.images[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB if needed

        if self.transform:
            image = self.transform(image)

        return image

# Modify the transformation
img_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Use the custom dataset for source and target data
dataset_source = CustomDataset(data_root=source_image_root, transform=img_transform)
dataset_target = CustomDataset(data_root=target_image_root, transform=img_transform)

# Create data loaders
dataloader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=8)
dataloader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=8)


#####################
#  load model       #
#####################

my_net = DSN()

#####################
# setup optimizer   #
#####################

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay_weight)

loss_classification = torch.nn.CrossEntropyLoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

#############################
# training network          #
#############################

len_dataloader = min(len(dataloader_source), len(dataloader_target))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in range(n_epoch):

    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0

    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = next(data_target_iter)
        # t_img, t_label = data_target
        t_img = data_target
        my_net.zero_grad()
        loss = 0
        batch_size = len(t_img)
        # print(t_img[0].shape)
        input_img = torch.FloatTensor(batch_size, 3, image_size[0], image_size[1])
        # class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            # t_label = t_label.cuda()
            input_img = input_img.cuda()
            # class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        # class_label.resize_as_(t_label).copy_(t_label)
        target_inputv_img = input_img.clone().detach().requires_grad_(True)
        # target_classv_label = torch.tensor(class_label)
        target_domainv_label = domain_label.clone().detach()

        if current_step > active_domain_loss_step:
            p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            p = 2. / (1. + np.exp(-10 * p)) - 1

            # activate domain loss
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
            loss += target_dann
        else:
            target_dann = torch.zeros(1).float().cuda()
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
            target_privte_code, target_share_code, _, target_rec_code = result

        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
        loss += target_simse

        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = next(data_source_iter)
        # s_img, s_label = data_source
        s_img = data_source

        my_net.zero_grad()
        batch_size = len(s_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size[0], image_size[1])
        # class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            # s_label = s_label.cuda()
            input_img = input_img.cuda()
            # class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(input_img).copy_(s_img)
        # class_label.resize_as_(s_label).copy_(s_label)
        source_inputv_img = torch.tensor(input_img, requires_grad=True)
        # source_classv_label = torch.tensor(class_label)
        source_domainv_label = torch.tensor(domain_label)

        if current_step > active_domain_loss_step:

            # activate domain loss

            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
            loss += source_dann

            loss.backward()
            scheduler.step()
            optimizer.step()
        # else:
        #     source_dann = torch.zeros(1).float().cuda()
        #     result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
        #     source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

        # source_classification = loss_classification(source_class_label, source_classv_label)
        # loss += source_classification

        # source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        # loss += source_diff
        # source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
        # loss += source_mse
        # source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
        # loss += source_simse

        # loss.backward()
        scheduler.step()
        # optimizer.step()

        i += 1
        current_step += 1
    print(
          'target_dann: %f, target_diff: %f, ' \
          'target_mse: %f, target_simse: %f' \
          % (
             target_dann.detach().cpu().numpy(),
             target_diff.detach().cpu().numpy(),target_mse.detach().cpu().numpy(), target_simse.detach().cpu().numpy()))

    # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    torch.save(my_net.state_dict(), model_root + '/dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth')
    test(epoch=epoch, name='mnist')
    test(epoch=epoch, name='mnist_m')

print('done')
