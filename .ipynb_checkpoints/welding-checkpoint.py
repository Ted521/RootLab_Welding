import os
import warnings

import mmcv
import torch
from dotmap import DotMap
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import collate, scatter
from mmcv.runner import build_optimizer, build_runner, load_checkpoint
from mmseg.datasets.custom import Compose
from mmseg.models.builder import build_segmentor

from util import Configuration, build_dataset, build_dataloader, LoadImage, show_test_result_pyplot, get_root_logger, \
    DistEvalHook, EvalHook, show_result_pyplot, intersect_and_union, divide_img, aggregate_img


def get_args():
    args = DotMap()

    args.max_iters = 2000  # max iteration
    args.log_interval = 500  # 20 iteration에 한 번씩 로깅
    args.evaluation_interval = 500  # 200 iteration에 한 번씩 evaluation
    args.checkpoint_interval = 500  # 200 iteration에 한 번씩 checkpoint 저장
    args.batch_size = 2
    args.crop_size = [512, 512]  # 이미지 분할 크기 지정
    args.num_worker = 2
    args.seed = 0
    # 학습 및 테스트 모델 환경설정 파일 경로
    # args.cfg_file = 'configs/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    args.cfg_file = 'configs/configs.py'

    # 비드면 탐지 웹 서비스(Django Framework) 프로젝트 디렉토리 data/WeldingDetection를 생성하고,
    # 학습 및 테스트 이미지, checkpoint 저장
    data = 'data'

    # 학습 결과 checkpoint 파일 저장 및 테스트 결과 이미지 저장 경로
    args.work_dir = os.path.join(data, 'work')
    args.test_dir = 'test_image'

    return args


class WeldingSegmentor:

    def __init__(self, args, device):
        """

        :param args:  전달인자
        :param device: 사용할 장치
        """
        self.args = args
        self.device = device

    def train_segmentor(self, model,
                        dataset,
                        cfg,
                        distributed=False,
                        validate=False,
                        timestamp=None,
                        meta=None):
        """Launch segmentor training."""
        logger = get_root_logger(cfg.log_level)

        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                drop_last=True) for ds in dataset
        ]

        # put model on gpus
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                # model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if cfg.get('runner') is None:
            cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

        # register hooks
        runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                       cfg.checkpoint_config, cfg.log_config,
                                       cfg.get('momentum_config', None))

        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = timestamp

        # register eval hooks
        if validate:
            print(cfg.data.val)
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow)

    def train(self):

        self.cfg = Configuration(self.args.cfg_file).define_configuration(self.args)

        # Build the dataset
        datasets = [build_dataset(self.cfg.data.train)]

        # Build the detector
        self.model = build_segmentor(
            self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        # Add an attribute for visualization convenience
        self.model.CLASSES = datasets[0].CLASSES

        # Create work_dir
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        self.train_segmentor(self.model, datasets, self.cfg, distributed=False, validate=False, meta=dict())

        with open(os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.split), 'r') as f:
            test_image_list = f.read().splitlines()

        self.model.eval()

        image_dir = os.path.join(os.getcwd(), self.args.work_dir)
        os.makedirs(image_dir, exist_ok=True)

        total_area_intersect = torch.zeros((2,), dtype=torch.float64)
        total_area_union = torch.zeros((2,), dtype=torch.float64)
        total_area_pred_label = torch.zeros((2,), dtype=torch.float64)
        total_area_label = torch.zeros((2,), dtype=torch.float64)

        mmcv.mkdir_or_exist(os.path.abspath(self.args.test_dir))
        for idx, img_name in enumerate(test_image_list):
            print(f"Testing {idx}/{len(test_image_list)} - {img_name}.jpg")

            img = mmcv.imread(os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.img_dir) + '/' + img_name + '.jpg')
            ann = mmcv.imread(os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.ann_dir) + '/' + img_name + '.png')

            self.model.cfg = self.cfg

            divided_imgs, divided_coor, org_size = divide_img(img, self.model.cfg.crop_size)
            result = self.inference_segmentor(self.model, divided_imgs)
            result = aggregate_img(result, divided_coor, org_size, self.model.cfg.crop_size)

            area_intersect, area_union, area_pred_label, area_label = intersect_and_union(result[0], ann)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
            show_test_result_pyplot(self.model, img, ann, result, title=img_name, image_dir=self.args.test_dir)

        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        print(f'IoU| Background: {iou[0]}, Bead: {iou[1]}')
        print(f'ACC| Background: {acc[0]}, Bead: {acc[1]}')


    def inference_segmentor(self, model, img):
        """Inference image with the segmentor.

        Args:
            model (nn.Module): The loaded segmentor.
            img (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            (list[Tensor]): The segmentation result.
        """
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        if isinstance(img, list):
            data = []
            for img_patch in img:
                data.append(test_pipeline(dict(img=img_patch)))
            data = collate(data, samples_per_gpu=1)
        else:
            data = dict(img=img)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        return result

    def inference(self, img_file):
        img = mmcv.imread(img_file)
        self.cfg = Configuration(self.args.cfg_file).define_configuration(self.args)
        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None
        self.model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        checkpoint_file_path = os.path.join(self.args.work_dir, 'latest.pth')

        if checkpoint_file_path is not None:
            chkpt = load_checkpoint(self.model, checkpoint_file_path, map_location='cpu')
            self.model.CLASSES = chkpt['meta']['CLASSES']

        self.model.cfg = self.cfg
        self.model.to(self.device)
        self.model.eval()

        divided_imgs, divided_coor, org_size = divide_img(img, self.model.cfg.crop_size)
        result = self.inference_segmentor(self.model, divided_imgs)
        result = aggregate_img(result, divided_coor, org_size, self.model.cfg.crop_size)

        title = os.path.splitext(os.path.basename(img_file))[0] + '_result'
        show_result_pyplot(self.model, img, result, title=title)

    def test(self, checkpoints='latest.pth'):
        self.cfg = Configuration(self.args.cfg_file).define_configuration(self.args)
        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None
        self.model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        checkpoint_file_path = os.path.join(self.args.work_dir, checkpoints)
        if checkpoint_file_path is not None:
            chkpt = load_checkpoint(self.model, checkpoint_file_path, map_location='cpu')
            self.model.CLASSES = chkpt['meta']['CLASSES']

        self.model.cfg = self.cfg
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.split), 'r') as f:
            test_image_list = f.read().splitlines()

        image_dir = os.path.join(os.getcwd(), self.args.work_dir)
        os.makedirs(image_dir, exist_ok=True)

        total_area_intersect = torch.zeros((2,), dtype=torch.float64)
        total_area_union = torch.zeros((2,), dtype=torch.float64)
        total_area_pred_label = torch.zeros((2,), dtype=torch.float64)
        total_area_label = torch.zeros((2,), dtype=torch.float64)

        mmcv.mkdir_or_exist(os.path.abspath(self.args.test_dir))
        for idx, img_name in enumerate(test_image_list):
            print(f"Testing {idx}/{len(test_image_list)} - {img_name}.jpg")

            img = mmcv.imread(
                os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.img_dir) + '/' + img_name + '.jpg')
            ann = mmcv.imread(
                os.path.join(self.cfg.data.val.data_root, self.cfg.data.val.ann_dir) + '/' + img_name + '.png')

            self.model.cfg = self.cfg

            divided_imgs, divided_coor, org_size = divide_img(img, self.model.cfg.crop_size)
            result = self.inference_segmentor(self.model, divided_imgs)
            result = aggregate_img(result, divided_coor, org_size, self.model.cfg.crop_size)

            area_intersect, area_union, area_pred_label, area_label = intersect_and_union(result[0], ann)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
            show_test_result_pyplot(self.model, img, ann, result, title=img_name, image_dir=self.args.test_dir)

        precision = total_area_intersect / total_area_pred_label
        recall = total_area_intersect / total_area_label
        f1 = 2 * ( precision * recall ) / (precision + recall)
        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        print(f'IoU| Background: {iou[0]}, Bead: {iou[1]}')
        print(f'ACC| Background: {acc[0]}, Bead: {acc[1]}')
        print(f'Precision| Background: {precision[0]}, Bead: {precision[1]}')
        print(f'Recall| Background: {recall[0]}, Bead: {recall[1]}')
        print(f'F1| Background: {f1[0]}, Bead: {f1[1]}')



if __name__ == '__main__':
    segmentor = WeldingSegmentor(get_args(), device='cuda:0')

    # train & validation
    # segmentor.train()
    segmentor.test('iter_100000.pth')

    # inference 1 image and save result to test_images directory
    # img_file = '506.jpg'
    # segmentor.inference(img_file)
