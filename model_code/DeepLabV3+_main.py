from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
import tensorflow as tf

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from datasets.heartxray import HeartXRayDataset
from datasets.chestxray import ChestXRayDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]
        )
        print("Success")
    except RuntimeError as e:
        print("Fail", e)



# def get_argparser():
#     parser = argparse.ArgumentParser()
#
#     # Datset Options
#     parser.add_argument("--data_root", type=str, default='datasets/Chest-X-Ray',
#                         help="path to Dataset")
#     parser.add_argument("--dataset", type=str, default='voc',
#                         choices=['voc', 'cityscapes', 'chestxray', 'qdata', 'heartxray'], help='Name of dataset')
#     parser.add_argument("--num_classes", type=int, default=None,
#                         help="num classes (default: None)")
#
#     # Deeplab Options
#     available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
#                               not (name.startswith("__") or name.startswith('_')) and callable(
#                               network.modeling.__dict__[name])
#                               )
#     parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
#                         choices=available_models, help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
#     parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
#
#     # Train Options
#     parser.add_argument("--test_only", action='store_true', default=False)
#     parser.add_argument("--save_val_results", action='store_true', default=False,
#                         help="save segmentation results to \"./results\"")
#     parser.add_argument("--total_itrs", type=int, default=30e3,
#                         help="epoch number (default: 30k)")
#     parser.add_argument("--total_epochs", type=int, default=20, help="number of training epochs")  # epoch 기준
#     parser.add_argument("--lr", type=float, default=0.01,
#                         help="learning rate (default: 0.01)")
#     parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
#                         help="learning rate scheduler policy")
#     parser.add_argument("--step_size", type=int, default=10000)
#     parser.add_argument("--crop_val", action='store_true', default=False,
#                         help='crop validation (default: False)')
#     parser.add_argument("--batch_size", type=int, default=8,
#                         help='batch size (default: 8)')
#     parser.add_argument("--val_batch_size", type=int, default=8,
#                         help='batch size for validation (default: 8)')
#     parser.add_argument("--crop_size", type=int, default=513)
#
#     parser.add_argument("--ckpt", default=None, type=str,
#                         help="restore from checkpoint")
#     parser.add_argument("--continue_training", action='store_true', default=False)
#
#     parser.add_argument("--loss_type", type=str, default='cross_entropy',
#                         choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
#     parser.add_argument("--gpu_id", type=int, default=0,
#                         help="GPU ID")
#     parser.add_argument("--weight_decay", type=float, default=1e-4,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument("--random_seed", type=int, default=42,
#                         help="random seed (default: 42)")
#     parser.add_argument("--print_interval", type=int, default=10,
#                         help="print interval of loss (default: 10)")
#     parser.add_argument("--val_interval", type=int, default=100,
#                         help="epoch interval for eval (default: 100)")
#     parser.add_argument("--download", action='store_true', default=False,
#                         help="download datasets")
#
#     # PASCAL VOC Options
#     parser.add_argument("--year", type=str, default='2012',
#                         choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
#
#     # Visdom options
#     parser.add_argument("--enable_vis", action='store_true', default=False,
#                         help="use visdom for visualization")
#     parser.add_argument("--vis_port", type=str, default='13570',
#                         help='port for visdom')
#     parser.add_argument("--vis_env", type=str, default='main',
#                         help='env for visdom')
#     parser.add_argument("--vis_num_samples", type=int, default=8,
#                         help='number of samples for visualization (default: 8)')
#     return parser


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='datasets/Chest-X-ray', help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='heartxray',
                        choices=['chestxray', 'heartxray'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes (BCE uses 1 output channel)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and callable(network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet', choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False)
    parser.add_argument("--total_epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'])
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--loss_type", type=str, default='bce', choices=['bce', 'focal_loss'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--enable_vis", action='store_true', default=False, help="enable visualization")
    parser.add_argument("--total_itrs", type=int, default=10000, help="total iterations for scheduler")

    return parser

def get_dataset(opts):

    if opts.dataset == 'chestxray':
        transform = et.ExtCompose([
            et.ExtResize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ✅ RGB 기준 정규화
        ])
         ###################여기 val값 ??
        train_dst = ChestXRayDataset(root=opts.data_root, mode='train', transforms=transform)
        val_dst = ChestXRayDataset(root=opts.data_root, mode='val', transforms=transform)

        return train_dst, val_dst


    elif opts.dataset == 'heartxray':
        train_transform = et.ExtCompose([
            et.ExtResize(size=(128, 128), interpolation=InterpolationMode.BILINEAR),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtResize(size=(128, 128), interpolation=InterpolationMode.BILINEAR),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = HeartXRayDataset(root=opts.data_root, transforms=train_transform, mode='train')
        val_dst = HeartXRayDataset(root=opts.data_root, transforms=val_transform, mode='val')

        return train_dst, val_dst



# def get_dataset(opts):
#     """ Dataset And Augmentation
#     """
#     if opts.dataset == 'voc':
#         train_transform = et.ExtCompose([
#             # et.ExtResize(size=opts.crop_size),
#             et.ExtRandomScale((0.5, 2.0)),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         if opts.crop_val:
#             val_transform = et.ExtCompose([
#                 et.ExtResize(opts.crop_size),
#                 et.ExtCenterCrop(opts.crop_size),
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         else:
#             val_transform = et.ExtCompose([
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                     image_set='train', download=opts.download, transform=train_transform)
#         val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                   image_set='val', download=False, transform=val_transform)
#
#     elif opts.dataset == 'cityscapes':
#         train_transform = et.ExtCompose([
#             # et.ExtResize( 512 ),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
#             et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#
#         val_transform = et.ExtCompose([
#             # et.ExtResize( 512 ),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#
#         train_dst = Cityscapes(root=opts.data_root,
#                                split='train', transform=train_transform)
#         val_dst = Cityscapes(root=opts.data_root,
#                              split='val', transform=val_transform)
#
#     elif opts.dataset == 'chestxray':
#         train_transform = et.ExtCompose([
#             et.ExtResize(size=(128, 128), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         val_transform = et.ExtCompose([
#             et.ExtResize(size=(128, 128), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#
#         train_dst = ChestXRayDataset(root=os.path.join(opts.data_root, 'train'), transforms=train_transform)
#         val_dst = ChestXRayDataset(root=os.path.join(opts.data_root, 'val'), transforms=val_transform)
#
#         return train_dst, val_dst
#
#     elif opts.dataset == 'qdata':
#         train_transform = et.ExtCompose([
#             et.ExtResize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         val_transform = et.ExtCompose([
#             et.ExtResize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#
#         train_dst = QDataDataset(
#             x_path=os.path.join(opts.data_root, "X_train.npy"),
#             y_path=os.path.join(opts.data_root, "y_train.npy"),
#             transforms=train_transform
#         )
#         val_dst = QDataDataset(
#             x_path=os.path.join(opts.data_root, "X_val.npy"),
#             y_path=os.path.join(opts.data_root, "y_val.npy"),
#             transforms=val_transform
#         )
#         return train_dst, val_dst
#
#     # ✅ main.py 내부 get_dataset 수정
#     elif opts.dataset == 'heartxray':
#         train_transform = et.ExtCompose([
#             et.ExtResize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         val_transform = et.ExtCompose([
#             et.ExtResize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#
#         train_dst = ChestXRayDataset(root=opts.data_root, transforms=train_transform, mode='train', random_seed=42)
#         val_dst = ChestXRayDataset(root=opts.data_root, transforms=val_transform, mode='val', random_seed=42)
#
#         return train_dst, val_dst



def dice_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)



def validate(opts, model, loader, device, metrics=None, ret_samples_ids=None):
    """Do validation and return specified samples"""
    if metrics is not None:
        metrics.reset()

    ret_samples = []
    dice_scores = []
    iou_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    accuracy_list = []

    total_loss = 0.0
    num_batches = 0

    # # ✅ 손실 함수 정의 (옵션에 따라 선택)
    # if opts.loss_type == 'focal_loss':
    #     loss_fn = utils.FocalLoss(ignore_index=255, size_average=True)
    # else:
    #     loss_fn = nn.BCEWithLogitsLoss()

    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'bce':
        pos_weight = torch.tensor([0.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if opts.save_val_results:
        os.makedirs("results", exist_ok=True)
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)

            # 🎯 마스크 차원 정리 (shape: B x 1 x H x W)
            if labels.ndim == 5:
                labels = labels.squeeze(1)
            elif labels.ndim == 3:
                labels = labels.unsqueeze(1)

            labels = labels.to(device, dtype=torch.float32)

            # 모델 추론
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            # 🎯 시그모이드 + threshold → 0 or 1 예측 결과
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().squeeze(1).cpu().numpy()  # shape: (B, H, W)
            targets = labels.squeeze(1).cpu().numpy().astype(int)  # shape: (B, H, W)

            for pred, target in zip(preds, targets):
                # 🎯 shape 불일치 예외 처리
                if pred.shape != target.shape:
                    print(f"[!] Skipped due to shape mismatch: pred {pred.shape}, target {target.shape}")
                    continue

                pred_bin = pred.flatten()
                target_bin = target.flatten()

                # 🎯 길이 불일치 예외 처리
                if pred_bin.shape[0] != target_bin.shape[0]:
                    print(f"[!] Skipped due to length mismatch: pred {len(pred_bin)}, target {len(target_bin)}")
                    continue

                # 🎯 메트릭 계산
                dice_scores.append(dice_score(pred_bin, target_bin))

                cm = confusion_matrix(target_bin, pred_bin, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0

                eps = 1e-7
                iou = (tp + eps) / (tp + fp + fn + eps)
                precision = (tp + eps) / (tp + fp + eps)
                recall = (tp + eps) / (tp + fn + eps)
                specificity = (tn + eps) / (tn + fp + eps)
                accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)
                specificity_list.append(specificity)
                accuracy_list.append(accuracy)

            # 🎯 샘플 저장용
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

    # 🎯 최종 메트릭 정리
    score = {}
    if metrics is not None:
        score = metrics.get_results()
    score.update({
        "Dice": np.mean(dice_scores),
        "IoU": np.mean(iou_list),
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "Specificity": np.mean(specificity_list),
        "Accuracy": np.mean(accuracy_list),
        "loss": total_loss / num_batches
    })

    print("✅ Validation complete")
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 1  # ✅ Background, Lung
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)
    gpu_id = int(opts.gpu_id)  # ⭐️ 꼭 추가!
    if torch.cuda.is_available():
        # torch.cuda.set_device(gpu_id)  # 수정된 부분
        device = torch.device(f"cuda:{gpu_id}")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Using CPU.")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        pretrained_backbone=True
    )


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')

    # while True:
    #     model.train()
    #     cur_epochs += 1
    #     for (images, labels) in train_loader:
    #         cur_itrs += 1
    #
    #         images = images.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)
    #
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         np_loss = loss.detach().cpu().numpy()
    #         interval_loss += np_loss
    #         writer.add_scalar("Train/Loss", np_loss, cur_itrs)
    #
    #         if (cur_itrs) % 10 == 0:
    #             interval_loss = interval_loss / 10
    #             print("Epoch %d, Itrs %d/%d, Loss=%f" %
    #                   (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
    #             interval_loss = 0.0
    #
    #         if (cur_itrs) % opts.val_interval == 0:
    #             save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
    #                       (opts.model, opts.dataset, opts.output_stride))
    #             print("validation...")
    #             model.eval()
    #             val_score, ret_samples = validate(
    #                 opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
    #                 ret_samples_ids=vis_sample_id)
    #             val_loss = 1 - val_score['Mean IoU']
    #             writer.add_scalar("Val/MeanIoU", val_score['Mean IoU'], cur_itrs)
    #             writer.add_scalar("Val/Dice", val_score['Dice'], cur_itrs)
    #
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 patience_counter = 0
    #                 best_score = val_score['Mean IoU']
    #                 save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth')
    #                 print(f"💾 Best model updated. Dice Score: {val_score['Dice']:.4f}")
    #             else:
    #                 patience_counter += 1
    #
    #             if patience_counter >= patience:
    #                 print("Early stopping triggered.")
    #                 return
    #
    #             if vis is not None:
    #                 vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
    #                 vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
    #                 vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
    #                 for k, (img, target, lbl) in enumerate(ret_samples):
    #                     img = (denorm(img) * 255).astype(np.uint8)
    #                     target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
    #                     lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
    #                     concat_img = np.concatenate((img, target, lbl), axis=2)
    #                     vis.vis_image(f'Sample {k}', concat_img)
    #                     # 추가된 시각화
    #                     plt.figure(figsize=(12, 4))
    #                     plt.subplot(1, 3, 1)
    #                     plt.imshow(np.transpose(img, (1, 2, 0)))
    #                     plt.title("Input")
    #                     plt.subplot(1, 3, 2)
    #                     plt.imshow(np.transpose(target, (1, 2, 0)))
    #                     plt.title("Ground Truth")
    #                     plt.subplot(1, 3, 3)
    #                     plt.imshow(np.transpose(lbl, (1, 2, 0)))
    #                     plt.title("Prediction")
    #                     plt.tight_layout()
    #                     plt.savefig(f'visualization/sample_{cur_itrs}_{k}.png')
    #                     plt.close()
    #             model.train()
    #         scheduler.step()
    #         if cur_itrs >= opts.total_itrs:
    #             return

# epoch 20 - early stop

    train_losses = []
    val_losses = []

    for cur_epochs in range(1, opts.total_epochs + 1):
        model.train()
        interval_loss = 0.0
        for images, labels in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            if labels.ndim == 5:
                labels = labels.squeeze(1)
            elif labels.ndim == 3:
                labels = labels.unsqueeze(1)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            writer.add_scalar("Train/Loss", np_loss, cur_itrs)

        train_losses.append(interval_loss / len(train_loader))  # ✅ 한 epoch이 끝나고 append

        print("Epoch %d/%d, Loss=%.4f" % (cur_epochs, opts.total_epochs, interval_loss / len(train_loader)))

        # 검증 주기
        if cur_epochs % opts.val_interval == 0:
            print("validation...")
            model.eval()
            val_loss = 0.0
            num_batches = 0

            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=None,
                ret_samples_ids=vis_sample_id)

            # ✔️ 실제 loss가 존재한다면 val_loss로 처리
            val_loss = val_score['loss']
            val_losses.append(val_loss)

            writer.add_scalar("Val/Dice", val_score['Dice'], cur_epochs)
            writer.add_scalar("Val/Loss", val_loss, cur_epochs)

            # ✅ best 모델 갱신 시에만 저장 및 시각화
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # ✅ 모델 저장
                save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth')
                print(f"💾 Best model updated. Dice Score: {val_score['Dice']:.4f}")

                # ✅ 평가지표 저장
                os.makedirs("results", exist_ok=True)
                with open('results/best_model_scores.txt', 'w') as f:
                    f.write(f"Dice: {val_score['Dice']:.4f}\n")
                    f.write(f"Loss: {val_score['loss']:.4f}\n")
                    f.write(f"Accuracy: {val_score['Accuracy']:.4f}\n")
                print("📊 Best model scores saved to results/best_model_scores.txt")

                # ✅ 시각화 이미지 저장 (모든 샘플 한 장에 세로로)
                os.makedirs("visualization", exist_ok=True)
                num_samples = len(ret_samples)
                if num_samples > 0:
                    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

                    for i, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(1, 2, 0)
                        lbl = train_dst.decode_target(lbl).transpose(1, 2, 0)

                        axes[i, 0].imshow(img)
                        axes[i, 0].set_title("Input")
                        axes[i, 0].axis('off')

                        axes[i, 1].imshow(target, cmap='gray')
                        axes[i, 1].set_title("Ground Truth")
                        axes[i, 1].axis('off')

                        axes[i, 2].imshow(lbl, cmap='gray')
                        axes[i, 2].set_title("Prediction")
                        axes[i, 2].axis('off')

                    plt.tight_layout()
                    plt.savefig("visualization/best_model_all_samples.png")
                    plt.close()
                    print("🖼️ Best model visualization image saved to visualization/best_model_all_samples.png")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                return

        # 손실함수 시각화
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        os.makedirs("results", exist_ok=True)
        plt.savefig('results/loss_curve.png')
        plt.close()
        print("📈 Loss curve saved to results/loss_curve.png")


if __name__ == '__main__':
    main()


