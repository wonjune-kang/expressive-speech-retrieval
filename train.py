import argparse

import torch
from omegaconf import OmegaConf

from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help="yaml file for configuration"
    )
    parser.add_argument(
        '-g', '--gpu_idx', type=int, default=0,
        help="index of GPU device"
    )
    parser.add_argument(
        '-n', '--run_name', type=str, required=True,
        help='name of the run'
    )
    parser.add_argument(
        '-p', '--checkpoint_path', type=str,
        help='path to load weights from if resuming from checkpoint'
    )
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nUsing device: {device}\n")

    config = OmegaConf.load(args.config)
    trainer = Trainer(args, config, device)

    trainer.train()
