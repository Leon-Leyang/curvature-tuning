"""
This script evaluates the generalization improvement achieved by CT from classification to semantic segmentation.

The code is adapted from the semseg repository: https://github.com/hszhao/semseg.
"""
import argparse
import os
import time
import cv2
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.semseg.util import dataset, transform, config
from utils.semseg.util.util import AverageMeter, intersectionAndUnion, intersectionAndUnionGPU, check_makedirs, \
    colorize, poly_learning_rate, find_free_port, main_process, check
from utils.semseg.model.pspnet import PSPNet
from utils.curvature_tuning import replace_module, CT
from utils.utils import get_file_name, set_logger, fix_seed
from loguru import logger
import wandb

cv2.ocl.setUseOpenCL(False)


def main_train(beta):
    args.save_path = f'diff_task_results/seed{args.manual_seed}/models/{beta:.2f}'
    os.makedirs(args.save_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        fix_seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args, beta))
    else:
        train_worker(args.train_gpu, args.ngpus_per_node, args, beta)


def train_worker(gpu, ngpus_per_node, argss, beta):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.cls, model.aux]
    if beta != 1:
        modules_ori = replace_module(nn.ModuleList(modules_ori), nn.ReLU, CT, beta=beta, coeff=0.5)

    params_list = []
    for module in modules_ori:  # Freeze the feature extractor
        for param in module.parameters():
            param.requires_grad = False
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_process(args):
        logger.debug(args)
        logger.debug("=> creating model ...")
        logger.debug("Classes: {}".format(args.classes))
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    # Check for existing checkpoints
    latest_checkpoint = None
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(args.save_path) if f.startswith('train_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        epochs = [int(f.replace('train_epoch_', '').replace('.pth', '')) for f in checkpoint_files]
        max_epoch = max(epochs)
        latest_checkpoint = os.path.join(args.save_path, f'train_epoch_{max_epoch}.pth')
        start_epoch = max_epoch
        logger.debug(f"Found checkpoint '{latest_checkpoint}'. Resuming training from epoch {start_epoch}.")

    # Load checkpoint if available
    if latest_checkpoint and os.path.isfile(latest_checkpoint):
        if main_process(args):
            logger.debug(f"=> loading checkpoint '{latest_checkpoint}'")
        checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage.cuda(gpu[0])) # gpu[0] is a hack to be able to run the single GPU case
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if main_process(args):
            logger.debug(f"=> loaded checkpoint '{latest_checkpoint}' (epoch {start_epoch})")
    else:
        if main_process(args):
            logger.debug("No checkpoint found. Starting training from scratch.")
        start_epoch = 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=[x * 255 for x in mean], ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=[x * 255 for x in mean], ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    args.start_epoch = start_epoch
    if main_process(args):
        wandb.init(project='semseg', entity='leyang_hu')
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, epoch)

        if (epoch_log % args.save_freq == 0) and main_process(args):
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.debug('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
    if main_process(args):
        wandb.finish()

    # Destroy the process group
    if args.distributed:
        dist.destroy_process_group()


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output, main_loss, aux_loss = model(input, target)
        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process(args):
            logger.debug('Epoch: [{}/{}][{}/{}] '
                         'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                         'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         'Remain {remain_time} '
                         'MainLoss {main_loss_meter.val:.4f} '
                         'AuxLoss {aux_loss_meter.val:.4f} '
                         'Loss {loss_meter.val:.4f} '
                         'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                           batch_time=batch_time,
                                                           data_time=data_time,
                                                           remain_time=remain_time,
                                                           main_loss_meter=main_loss_meter,
                                                           aux_loss_meter=aux_loss_meter,
                                                           loss_meter=loss_meter,
                                                           accuracy=accuracy))
            wandb.log({'epoch': epoch, 'main_loss': main_loss_meter.val, 'aux_loss': aux_loss_meter.val, 'loss': loss_meter.val, 'accuracy': accuracy})

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process(args):
        logger.debug('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def main_test(beta):
    args.save_folder = f'diff_task_results/seed{args.manual_seed}/results/{beta:.2f}'
    args.save_path = f'diff_task_results/seed{args.manual_seed}/models/{beta:.2f}'
    os.makedirs(args.save_folder, exist_ok=True)
    args.model_path = args.save_path + f'/train_epoch_{args.epochs}.pth'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.debug(args)
    logger.debug("=> creating model ...")
    logger.debug("Classes: {}".format(args.classes))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_transform = transform.Compose([transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    if not args.has_prediction:
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        if beta != 1:
            replace_module(nn.ModuleList(modules_ori), nn.ReLU, CT, beta=beta, coeff=0.5)
            logger.debug(f"Replaced ReLU with CT, beta={beta: .2f}")

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.debug("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.debug("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        test(test_loader, test_data.data_list, model, args.classes, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors)
    if args.split != 'test':
        mIoU, mAcc, allAcc = cal_acc(test_data.data_list, gray_folder, args.classes, names)
        return mIoU, mAcc, allAcc
    else:
        return None, None, None


def net_process(model, image, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=0)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.debug('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.debug('Test: [{}/{}] '
                         'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                         'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                     data_time=data_time,
                                                                                     batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.debug('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.debug('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.debug('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.debug('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

    return mIoU, mAcc, allAcc


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='voc2012_pspnet50.yaml', help='config file')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for CT')
    parser.add_argument('opts', help='see voc2012_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    cfg.beta = args.beta
    return cfg


def check_log_file(log_file_path, beta):
    if not os.path.exists(log_file_path):
        return False

    beta_pattern = re.compile(rf'Beta={beta:.2f}, mIoU: (\d+\.\d+)')
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if beta_pattern.search(line):
                return True
    return False


def find_best_miou(log_file_path):
    if not os.path.exists(log_file_path):
        return None, None

    beta_miou_pattern = re.compile(r'Beta=(\d+\.\d+), mIoU: (\d+\.\d+)')
    best_beta = None
    best_miou = -float('inf')

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = beta_miou_pattern.search(line)
            if match:
                beta = float(match.group(1))
                miou = float(match.group(2))
                if miou > best_miou:
                    best_beta = beta
                    best_miou = miou

    return best_beta, best_miou


if __name__ == '__main__':
    args = get_parser()
    check(args)
    f_name = get_file_name(__file__)
    log_file_path = set_logger(name=f'{f_name}_seed{args.manual_seed}')
    logger.info(f'Log file: {log_file_path}')

    beta = args.beta

    if check_log_file(log_file_path, beta):
        logger.info(f"Results for Beta={beta:.2f} already exist in the log file. Exiting.")
        exit(0)

    main_train(beta)
    mIoU, _, _ = main_test(beta)
    logger.info(f'Beta={beta:.2f}, mIoU: {mIoU:.4f}')

    # Find the best mIoU achieved
    if beta == 1.0:
        best_beta, best_miou = find_best_miou(log_file_path)
        if best_beta is not None:
            logger.info(f'Best mIoU: {best_miou:.4f} achieved at Beta={best_beta:.2f}')
        else:
            logger.info('No mIoU results found in the log file.')
