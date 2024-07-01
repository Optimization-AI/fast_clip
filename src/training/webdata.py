import logging
import random
from multiprocessing import Value
from typing import Dict, Callable, Optional

from torch.utils.data import get_worker_info
import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def json_parse_key(json_dict: Dict) -> int:
    return int(json_dict["key"])


class WebDataset(wds.DataPipeline):
    r"""
        An image-text dataset that is stored in webdataset format. For more information on webdataset format,
        refer to https://github.com/webdataset/webdataset.

        Args:
            input_shards (str): Path to the dataset shards.
            is_train (bool): Whether the dataset is for training or evaluation.
            batch_size (int): Batch size per worker.
            preprocess_img (Callable): Function to preprocess the image.
            seed (int): Seed for shuffling the dataset.
            epoch (int): Start epoch number.
            tokenize (Optional[Callable]): Tokenizer function for the text data.
            return_index (bool): Whether to return the index of the data.
    """
    def __init__(self,
                 input_shards: str,
                 is_train: bool,
                 batch_size: int,
                 preprocess_img: Callable,
                 seed: int = 0,
                 epoch: int = 0,
                 tokenize: Optional[Callable] = None,
                 return_index: bool = False,
                ):
        self.shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
        pipeline = [wds.SimpleShardList(input_shards)]

        # at this point we have an iterator over all the shards
        if is_train:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=seed,
                    epoch=self.shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
            pipeline.extend([
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ])
        else:
            pipeline.extend([
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ])

        # here we also load the key of data
        if return_index:
            rename = wds.rename(image="jpg;png;jpeg;webp", text="txt", key="json")
            if tokenize is not None:
                map_dict = wds.map_dict(image=preprocess_img, text=tokenize, key=json_parse_key)
            else:
                map_dict = wds.map_dict(image=preprocess_img, key=json_parse_key)
            to_tuple = wds.to_tuple("image", "text", "key", "key")
        else:
            rename = wds.rename(image="jpg;png;jpeg;webp", text="txt")
            if tokenize is not None:
                map_dict = wds.map_dict(image=preprocess_img, text=tokenize)
            else:
                map_dict = wds.map_dict(image=preprocess_img)
            to_tuple = wds.to_tuple("image", "text")
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            rename, map_dict, to_tuple,
            wds.batched(batch_size, partial=not is_train)
        ])

        super().__init__(*pipeline)
