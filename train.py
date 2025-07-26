import sys
import argparse
from pprint import pprint
from dataclasses import fields
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.logger import Logger
from src.utils import set_seed
from src.data import Loader, PoseDataset
from src.model import get_model


def main(**args_override):
    sys.stdout = open(f"logs/{args_override['device'][-1]}.log", "w")
    sys.stderr = sys.stdout

    # use argparse so that we can override config.yaml in command line
    # the priproity is command_line_args > config_path > default_value
    parser = argparse.ArgumentParser()
    # you can specify to use a config file instead of default value
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to the config file"
    )
    # you can specify to use command line args instead of default value
    for field in fields(Config):
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,
            help=f"update {field.name} in config",
        )
    args = parser.parse_args()

    # use default value if config_path is not specified
    config = (
        Config() if args.config_path is None else Config.from_yaml(args.config_path)
    )

    # override default value with command line args
    for field in fields(Config):
        value = getattr(args, field.name)
        if value is not None:
            setattr(config, field.name, value)

    # override default value with main args
    for field in fields(Config):
        value = args_override.get(field.name, None)
        if value is not None:
            setattr(config, field.name, value)    


    print("[ INFO ] before wandb initialization...")
    pprint(config)

    # set seed for reproducibility
    set_seed(config.seed)

    # setup logger
    logger = Logger(config)

    # set device
    assert (
        config.device == "cpu" or torch.cuda.is_available()
    ), f"CUDA is not available, device={config.device}"
    device = torch.device(config.device)

    # loading datasets
    train_dataset = PoseDataset(config, mode="train", scale=100)
    val_dataset = PoseDataset(config, mode="val", scale=100)
    # This loader will load data infinitely
    train_loader = Loader(
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            drop_last=True,
            num_workers=config.num_workers,
            shuffle=True,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
        )
    )
    val_loader = Loader(
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            drop_last=False,
            num_workers=config.num_workers,
            shuffle=True,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
        )
    )

    # model, optimizer, and scheduler
    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, config.max_iter, eta_min=config.learning_rate_min
    )

    # load checkpoint if exists
    if config.checkpoint is not None:
        checkpoint = torch.load(config.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        cur_iter = checkpoint["iter"]
        for _ in range(cur_iter):
            scheduler.step()

        # move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"loaded checkpoint from {config.checkpoint}")
    else:
        cur_iter = 0

    # init training
    model.to(device)
    model.train()

    # start training loop here
    for it in trange(cur_iter, config.max_iter):
        optimizer.zero_grad()
        data = train_loader.get()
        data = {k: v.to(device) for k, v in data.items()}
        # calls model.forward here
        loss, result_dict = model(**data)
        loss.backward()
        print("loss", loss.item())
        optimizer.step()
        scheduler.step()

        # log train metrics
        if it == 0 or (it+1) % config.log_interval == 0:
            logger.log(result_dict, "train", it)

        # save checkpoint and optimizer states
        if (it + 1) % config.save_interval == 0:
            logger.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iter=it + 1,
                ),
                it + 1,
            )

        # run validation for multiple batchs to get stable result
        if it == 0 or (it+1) % config.val_interval == 0:
            with torch.no_grad():
                model.eval()
                result_dicts = []
                for _ in range(config.val_num):
                    data = val_loader.get()
                    data = {k: v.to(device) for k, v in data.items()}
                    loss, result_dict = model(**data)
                    result_dicts.append(result_dict)
                logger.log(
                    {
                        k: np.array([dic[k].cpu() for dic in result_dicts]).mean()
                        for k in result_dicts[0].keys()
                    },
                    "val",
                    it,
                )
                model.train()


if __name__ == "__main__":
    import time
    import multiprocessing as mp
    import trimesh


    #################################
    #   Part 1: Train EstPoseNet    #
    #################################
    learning_rate = [1e-3, 1e-4]
    batch_size = [16, 32]

    processes = []
    for i in range(1):
        bsz = 16 # batch_size[int(i & 2 == 2)]
        mse_rot = True
        max_iter = 8000
        lr = 1e-3 # learning_rate[int(i & 1 == 1)]
        args = dict(
            model_type = "est_pose",
            batch_size = bsz,
            use_mse_loss_on_rot = mse_rot,
            learning_rate = lr,
            #checkpoint = "/home/bowenxiao/research/Assignment2/exps/est_pose_bsz_16_lr_0.001_mse_rot_True_20250417_234428/checkpoint/checkpoint_5000.pth",
            max_iter = max_iter,
            device = f"cuda:{i}",
            #exp_name = f"est_pose_bsz_{bsz}_lr_{lr}_mse_rot_{mse_rot}_{time.strftime('%Y%m%d_%H%M%S')}",
            exp_name = "debug"
        )

        p = mp.Process(target=main, kwargs=args)
        p.start()
        p.join()


    ##################################
    #   Part 2: Train EstCoordNet    #
    ##################################
    """ Comment the previous part and uncomment the following part to train EstCoordNet.
    
    learning_rate = [1e-3, 1e-4]
    batch_size = [16]

    processes = []
    for i in range(1):
        bsz = 16 # batch_size[int(i & 2 == 2)]
        lr = 1e-3 # learning_rate[int(i & 1 == 1)]
        mse_rot = False
        args = dict(
            model_type = "est_coord",
            learning_rate = lr,
            batch_size = bsz,
            use_mse_loss_on_rot = mse_rot,
            device = f"cuda:{i}",
            exp_name = f"est_coord_bsz_{bsz}_lr_{lr}_mse_rot_{mse_rot}_{time.strftime('%Y%m%d_%H%M%S')}",
        )

        p = mp.Process(target=main, kwargs=args)
        p.start()
        p.join()
    """