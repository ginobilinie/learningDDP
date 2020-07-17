import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler,Sampler
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import numpy as np
import torch.distrubuted as dist

'''
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 learn_ddp.py
dong nie
'''

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input shape", input.shape)
        return output

class RandomDataset(Dataset):
    def __init__(self, size, length, num_classes):
        self.len = length
        self.data = torch.randn(length, size).to('cuda')
        self.label = torch.randint(num_classes-1,(length,))
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(
        self, labels, mode= "downsampling"
    ):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the dataset
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self):
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        """
        Returns:
             length of result sample
        """
        return self.length

class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas = None,
        rank = None,
        shuffle = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



input_size = 5
output_size = 2
batch_size = 30
data_size = 900
num_classes = 11
base_lr = 1e-2
world_size = 2

# step.1 Initialization
torch.distributed.init_process_group(backend="nccl")

# step.2 config gpu(s) for each processor
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# step.3 use DistributedSampler
dataset = RandomDataset(input_size, data_size,num_classes)
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=DistributedSampler(dataset))
custom_sampler = None

custom_sampler =BalanceClassSampler(dataset)


if custom_sampler is not None:
    sampler =DistributedSamplerWrapper(sampler=custom_sampler)
else:
    sampler = DistributedSampler(dataset)


# step.4 Wrap the model to ddp
model = Model(input_size, output_size)
model.to(device) # or model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
optimizer = torch.optim.SGD(model.parameters(),lr=base_lr*world_size,momentum=0.9,weight_decay=1e-4)

ce = nn.CrossEntropyLoss(ignore_index=-1)

# step.5 run the model
optimizer.zero_grad()

for idx,batch in enumerate(rand_loader):
    data = batch[0]
    label = batch[1]
    data = data.cuda() # or data.to(device)
    label = label.cuda() # or label.to(device)

    output = model(data)

    loss_ce = ce(output, label)

    dist.all_reduce(loss_ce,dist.ReduceOp.SUM)
    loss_ce = loss_ce/world_size

    optimizer.zero_grad()
    loss_ce.backward()
    optimizer.step()

    print('iter:',iter,'loss_ce:%.4f'%loss_ce)
