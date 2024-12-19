import math

import torch
from torch.utils.data import IterableDataset, DataLoader

import data_utilities as du


class MyIterableDataset1(IterableDataset):
    def __init__(self, start: int, end: int, batch_size: int, transpose):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__init__() called.')
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.transpose = transpose

    def __iter__(self):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__iter__() called.')
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            worker_id = -1

        else:  # in a worker process
            worker_id = worker_info.id

            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        logger.info(f'Worker ID: {worker_id} iter_start: {iter_start} iter_end: {iter_end}.')

        samples = []
        for sample in range(iter_start, iter_end):
            samples.append(sample)
            if len(samples) == self.batch_size:
                yield samples
                samples = []
        
        if len(samples):
            return samples
        
        #return iter(range(iter_start, iter_end))


class MyIterableDataset2(IterableDataset):
    def __init__(self, start: int, end: int, transpose):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__init__() called.')
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.transpose = transpose

    def __iter__(self):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__iter__() called.')
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            worker_id = -1

        else:  # in a worker process
            worker_id = worker_info.id

            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        logger.info(f'Worker ID: {worker_id} iter_start: {iter_start} iter_end: {iter_end}.')

        for sample in range(iter_start, iter_end):
            yield self.transpose(sample)


class MyIterableDataset3(IterableDataset):
    def __init__(self, start: int, end: int, transpose):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__init__() called.')
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.transpose = transpose

    def __iter__(self):
        logger = du.create_logger()
        logger.info('MyIterableDataset.__iter__() called.')
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            worker_id = -1

        else:  # in a worker process
            worker_id = worker_info.id

            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        logger.info(f'Worker ID: {worker_id} iter_start: {iter_start} iter_end: {iter_end}.')

        samples = []
        for sample in range(iter_start, iter_end):
            samples.append(sample)

        return map(self.transpose, samples)


def my_transpose(x):
    return torch.tensor(x)


def main(start: int, end: int) -> None:
    batch_size = 2
    #ds = MyIterableDataset1(start=0, end=100, batch_size=batch_size, transpose=my_transpose)
    #ds = MyIterableDataset2(start=0, end=100, transpose=my_transpose)
    ds = MyIterableDataset3(start=0, end=10, transpose=my_transpose)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)

    for sample in dl:
        print(sample)


if __name__ == '__main__':
    main(0, 11)