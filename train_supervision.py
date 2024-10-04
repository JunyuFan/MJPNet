import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import torch
from torch import nn
import argparse
from pathlib import Path
from tools.utils import Adder
from pytorch_lightning.loggers import CSVLogger

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model.losses.pytorch_msssim import SSIM, MS_SSIM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default=r'config/config.py')
    return parser.parse_args()

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.contrastive_loss = config.contrast_loss


        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_metrics = MS_SSIM(data_range=1)

        self.ssim_epoch_adder_train = Adder()
        self.ssim_epoch_adder_val = Adder()

        self.psnr_epoch_adder_train = Adder()
        self.psnr_epoch_adder_val = Adder()

        self.contrastive_loss_epoch_adder_train = Adder()
        self.contrastive_loss_epoch_adder_val = Adder()



    def forward(self, x):
        # only net is used in the prediction/inference
        pred = self.net(x)

        return pred

    def training_step(self, batch, batch_idx):
        img, label, name = batch[0], batch[1], batch[2]
        output = self.net(img)
        
        pred = output

        l1_loss = self.l1_loss(pred, label)

        ssim = self.ssim_metrics(pred, label)
        ssim_loss = 1 - ssim
        contrastive_loss = self.contrastive_loss(pred, label, img)


        pred_clip = torch.clamp(pred, 0, 1)
        pred = pred_clip.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        psnr = peak_signal_noise_ratio(pred, label)
        self.psnr_epoch_adder_train(psnr.item())
        self.ssim_epoch_adder_train(ssim.item())


        loss = l1_loss*0.2 + ssim_loss*0.8 + contrastive_loss*0.1

        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return {'loss': loss, 'l1_loss': l1_loss.detach(), 'ssim_loss': ssim_loss.detach(), 'contrastive_loss': contrastive_loss.detach()}


    def training_epoch_end(self, outputs):
        ssim = self.ssim_epoch_adder_train.average()
        psnr = self.psnr_epoch_adder_train.average()
        print('\n')


        loss = torch.stack([x['loss'] for x in outputs]).mean()
        contrastive_loss = torch.stack([x['contrastive_loss'] for x in outputs]).mean()
        log_dict = {'train_loss': loss, 'contrastive_loss': contrastive_loss, 'train_ssim': ssim, 'train_psnr': psnr}
        self.log_dict(log_dict, prog_bar=True)

        self.ssim_epoch_adder_train.reset()
        self.psnr_epoch_adder_train.reset()

    def validation_step(self, batch, batch_idx):
        img, label, name = batch[0], batch[1], batch[2]
        results = self.forward(img)

        if type(results) is not torch.Tensor:
            pred = results[0]
        else:
            pred = results

        ssim = self.ssim_metrics(pred, label)

        pred_clip = torch.clamp(pred, 0, 1)
        pred = pred_clip.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        psnr = peak_signal_noise_ratio(pred, label)

        self.ssim_epoch_adder_val(ssim.item())
        self.psnr_epoch_adder_val(psnr.item())


        return {}

    def validation_epoch_end(self, outputs):
        ssim = self.ssim_epoch_adder_val.average()
        psnr = self.psnr_epoch_adder_val.average()

        log_dict = {'val_ssim': ssim, 'val_psnr': psnr}
        self.log_dict(log_dict, prog_bar=True)

        self.ssim_epoch_adder_val.reset()
        self.psnr_epoch_adder_val.reset()

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader



# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    rand_seed = 123
    seed_everything(rand_seed)
    print('seed: ', str(rand_seed))

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)
    trainer.fit(model=model)



if __name__ == "__main__":
   main()

