import logging
import os
import os.path as osp
import pickle
import random
import math
import shutil
import tempfile
from functools import partial

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.image import tensor2imgs
from mmcv.parallel import collate
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from mmcv.runner import get_dist_info
from mmcv.utils import get_logger, build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DistributedSampler

classes = ('background', 'bead')
palette = [[255, 255, 255], [255, 49, 49]]


@DATASETS.register_module()
class WeldingDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert os.path.exists(self.img_dir) and self.split is not None


class Configuration:
    """
    mmcv 에서 미리 정의된 설정정보를 기반으로 용접비드면 탐지위한 설정값을 덮어쓴다.

    """
    def __init__(self, cfg_file):
        """
        :param cfg_file: 설정 파일 경로
       """
        self.cfg = Config.fromfile(cfg_file)

    def define_configuration(self, args):
        self.cfg.data.samples_per_gpu = args.batch_size
        self.cfg.data.workers_per_gpu = args.num_worker

        # Set up working dir to save files and logs.
        self.cfg.work_dir = args.work_dir

        self.cfg.runner.max_iters = args.max_iters
        self.cfg.log_config.interval = args.log_interval
        self.cfg.evaluation.interval = args.evaluation_interval
        self.cfg.checkpoint_config.interval = args.checkpoint_interval

        # Set seed to facitate reproducing the result
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.cfg.seed = args.seed
        self.cfg.gpu_ids = list(range(1))

        return self.cfg


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.4,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    image_dir = os.path.join(os.getcwd(), 'test_image')

    if hasattr(model, 'module'):
        model = model.module

    if fig_size is None:
        dpi = matplotlib.rcParams['figure.dpi']
        height, width, depth = img.shape
        fig_size = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=fig_size, frameon=False)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.axis('off')

    img = model.show_result(img, result, palette=[[255, 255, 255], [0, 0, 255]], show=False, opacity=opacity)
    plt.imshow(img)
    plt.savefig(os.path.join(image_dir, title) + '.jpg', bbox_inches='tight', pad_inches=0, transparent=True,
                format="jpg")

    return os.path.join(image_dir, title) + '.jpg'


def show_test_result_pyplot(model,
                            img,
                            ann,
                            result,
                            palette=None,
                            fig_size=(15, 10),
                            opacity=0.5,
                            title='',
                            image_dir='test_image',
                            block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    image_dir = os.path.join(os.getcwd(), image_dir)

    if hasattr(model, 'module'):
        model = model.module

    # annotation file
    _, mask = cv2.threshold(ann, 50, 255, cv2.THRESH_BINARY_INV)
    white = 100 * np.ones_like(mask)
    mask = cv2.add(mask, white)

    plt.figure(figsize=fig_size)
    img = model.show_result(img, result, palette=[[255, 255, 255], [0, 0, 255]], show=False, opacity=opacity)
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, title))
    plt.close()
    plt.clf()

    plt.figure(figsize=fig_size)
    # img = mmcv.bgr2rgb(img)
    # res_fig = cv2.add(img, mask)
    res_fig = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    plt.imshow(res_fig)
    # # create a patch (proxy artist) for every color
    annot_palette = [[255, 0, 0], [255, 255, 100]]
    annot_classes = ['predicted', 'true']
    patches = [mpatches.Patch(color=np.array(annot_palette[i]) / 255., label=annot_classes[i]) for i in range(2)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, borderaxespad=0., fontsize='large')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, title + '_true_overlap'))
    plt.close()
    plt.clf()


def get_root_logger(log_file=None, log_level=logging.INFO):
    # generate logger
    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
        """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def single_gpu_test(self, model,
                        data_loader,
                        show=False,
                        out_dir=None,
                        efficient_test=False,
                        opacity=0.5):
        """Test with single GPU.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (utils.data.Dataloader): Pytorch data loader.
            show (bool): Whether show results during inference. Default: False.
            out_dir (str, optional): If specified, the results will be dumped into
                the directory to save output results.
            efficient_test (bool): Whether save the results as local numpy files to
                save CPU memory during evaluation. Default: False.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            list: The prediction results.
        """

        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, **data)

            if show or out_dir:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for img, img_meta in zip(imgs, img_metas):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        return results

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        results = self.single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def collect_results_cpu(self, result_part, size, tmpdir=None):
        """Collect results with CPU."""
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN,),
                                    32,
                                    dtype=torch.uint8,
                                    device='cuda')
            if rank == 0:
                tmpdir = tempfile.mkdtemp()
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
                dir_tensor[:len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            mmcv.mkdir_or_exist(tmpdir)
        # dump the part result to the dir
        mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
                part_list.append(mmcv.load(part_file))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)
            return ordered_results

    def collect_results_gpu(self, result_part, size):
        """Collect results with GPU."""
        rank, world_size = get_dist_info()
        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            return ordered_results

    def multi_gpu_test(self, model,
                       data_loader,
                       tmpdir=None,
                       gpu_collect=False,
                       efficient_test=False):
        """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
        it encodes results to gpu tensors and use gpu communication for results
        collection. On cpu mode it saves the results on different gpus to 'tmpdir'
        and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (utils.data.Dataloader): Pytorch data loader.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode.
            gpu_collect (bool): Option to use either gpu or cpu to collect results.
            efficient_test (bool): Whether save the results as local numpy files to
                save CPU memory during evaluation. Default: False.

        Returns:
            list: The prediction results.
        """

        model.eval()
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)

            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        if gpu_collect:
            results = self.collect_results_gpu(results, len(dataset))
        else:
            results = self.collect_results_cpu(results, len(dataset), tmpdir)
        return results

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


def intersect_and_union(pred_label, ann, num_classes=2):
    pred_label = torch.from_numpy((pred_label))
    label = np.zeros(ann.shape[:2])
    label[ann[:, :, 0] == 49] = 1
    label = torch.from_numpy(label)
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def divide_img(img, crop_size):
    img_size = img.shape
    nh = math.ceil(img_size[0] / crop_size[0])
    nw = math.ceil(img_size[1] / crop_size[1])
    imgs = []
    coor = []
    for i, dh in enumerate(np.linspace(0, crop_size[0] * nh - img_size[0], nh)):
        for j, dw in enumerate(np.linspace(0, crop_size[1] * nw - img_size[1], nw)):
            h = int((i + 1) * crop_size[0] - dh)
            w = int((j + 1) * crop_size[1] - dw)
            imgs.append(img[h - crop_size[0]:h, w - crop_size[1]:w])
            coor.append((h, w))
    return imgs, coor, img_size


def aggregate_img(result, coor, org_size, crop_size):
    result_agg = np.zeros(org_size[:2])
    for i, (h, w) in enumerate(coor):
        result_agg[h - crop_size[0]:h, w - crop_size[1]:w] = \
            np.where(result_agg[h - crop_size[0]:h, w - crop_size[1]:w] > result[i],
                     result_agg[h - crop_size[0]:h, w - crop_size[1]:w], result[i])

    return [result_agg]
