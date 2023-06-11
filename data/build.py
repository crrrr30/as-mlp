from functools import partial
import random
from typing import Callable
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data.mixup import *
from timm.data.random_erasing import RandomErasing
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp

from .samplers import SubsetRandomSampler

from datasets import concatenate_datasets, Dataset


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


def fast_collate(batch):
    batch_size = len(batch)
    targets = torch.tensor([b["label"] for b in batch], dtype=torch.int64)
    assert len(targets) == batch_size
    tensor = torch.zeros((batch_size, *batch[0]["image"].shape), dtype=torch.float32)
    for i in range(batch_size):
        tensor[i].copy_(batch[i]["image"])
    return tensor, targets


class FastCollateMixup(Mixup):
    """ Fast Collate w/ Mixup/Cutmix that applies different params to each element or whole batch

    A Mixup impl that's performed while collating the batches.
    """

    def _mix_elem_collate(self, output, batch, half=False):
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        if half:
            lam_batch = np.concatenate((lam_batch, np.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output, batch):
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed_i = batch[i][0]
            mixed_j = batch[j][0]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    mixed_temp = mixed_i.astype(np.float32) * lam + mixed_j.astype(np.float32) * (1 - lam)
                    mixed_j = mixed_j.astype(np.float32) * lam + mixed_i.astype(np.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    np.rint(mixed_j, out=mixed_j)
                    np.rint(mixed_i, out=mixed_i)
            output[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            output[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam

    def __call__(self, batch, _=None):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        if isinstance(batch[0], dict):
            batch = [(b["image"], b["label"]) for b in batch]
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        if self.mode == 'elem' or self.mode == 'half':
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == 'pair':
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)
        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')
        target = target[:batch_size]
        return output, target


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x
            
            
def _worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))
            

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    elif config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        collate_fn=FastCollateMixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES),
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        worker_init_fn=partial(_worker_init, worker_seeding='all'),
        persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=fast_collate,
        worker_init_fn=partial(_worker_init, worker_seeding='all'),
        persistent_workers=True
    )
    data_loader_train = PrefetchLoader(
            data_loader_train,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            fp16=False,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT
        )

    # setup mixup / cutmix
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        dataset = concatenate_datasets([Dataset.from_file(f"../imagenet-1k/imagenet-1k-train-{i:05d}-of-00257.arrow") for i in range(257)]).shuffle() if is_train else \
            concatenate_datasets([Dataset.from_file(f"../imagenet-1k/imagenet-1k-validation-{i:05d}-of-00013.arrow",) for i in range(13)])
        def transforms(examples):
            examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
            return examples
        dataset.set_transform(transforms)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            use_prefetcher=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=str_to_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
