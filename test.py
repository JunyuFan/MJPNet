import ttach as tta
import time
import argparse
import torch
import torchvision
import torch.nn.functional as F
import os

from pathlib import Path
from train_supervision import *
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default=r'config/config.py', help="Path to config")
    arg("-o", "--output_path", type=Path, default=r'fig_results', help="Path where to save results.")
    arg("-r", "--resize", type=bool, default=True)

    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(123)

    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda(config.gpus[0])
    model.eval()

    test_dataset = config.test_dataset

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    dummy_input = torch.rand(1, 3, 240, 320).to(config.gpus[0])

    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    torch.cuda.synchronize()

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        count = 0
        avg_time = 0
        for input, filename in test_loader:
            if args.resize:
                input = F.interpolate(input, size=(240, 320), mode='bilinear', align_corners=False)
            t0 = time.time()
            pred = model(input.cuda(config.gpus[0]))
            t1 = time.time()
            img_inf_time = t1 - t0
            avg_time += img_inf_time
            count += 1
            torchvision.utils.save_image(pred, os.path.join(args.output_path, filename[0]))
            print('inference time: {} s'.format(img_inf_time))
        print('average inference time: {} s'.format(avg_time / count))


if __name__ == "__main__":
    main()
