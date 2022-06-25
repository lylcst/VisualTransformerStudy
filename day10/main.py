import numpy as np
import paddle 
import paddle.nn as nn 
import paddle.distributed as dist 
from paddle.io import Dataset 
from paddle.io import DataLoader 
from paddle.io import DistributedBatchSampler


def build_model():
    model = nn.Sequential(*[
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 10)
    ])
    return model


def main_worker(*args):
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    model = build_model()
    model = paddle.DataParallel(model)
    print(f"hello, PPViT, I am {local_rank}: I built a model for myself")


def main():
    dist.spawn(main_worker, nprocs=8)


if __name__ == '__main__':
    main()