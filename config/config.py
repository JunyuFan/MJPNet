from torch.utils.data import DataLoader
from model.models.model import Network
from model.datasets.data import *

from model.losses.contrastive_loss import *


# training hparam
max_epoch = 200
train_batch_size = 8
val_batch_size = 8
lr = 1e-4
accumulate_n = 1


weights_name = "uie"
weights_path = "model_weights/uie"
test_weights_name = "uie"
log_name = 'uie/{}'.format(weights_name)
monitor = 'val_psnr'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None

test_dataset = 'images'

#  define the network
net = Network()

# define the loss
contrast_loss = ContrastiveLoss()


# define the dataloader

train_dataset = get_training_set(dataset_path='data/train', data='images', label='labels', patch_size=256, data_augmentation=True, image_size=512)

val_dataset = get_eval_set(dataset_path='data/test', data='images', label='labels', image_size=256)

test_dataset = get_test_set(dataset_path='data/test', data=test_dataset)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
tone = max_epoch // 10
for i in range(1, max_epoch):
    if i % tone == 0:
        milestones.append(i)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5)