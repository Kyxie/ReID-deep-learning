# This code is for Training, Metric Learning
# Engineer: Kunyang Xie
# Last Update: 19/3/2021

from __future__ import print_function, absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
# Uncomment if gpu is used
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from TriHard import TriHardLoss
from sample import RandomIdentitySampler
from torch.optim import lr_scheduler
import data_manager
from dataset_loader import ImageDataset
import torchvision.transforms as T
import transform
import models
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate

parser = argparse.ArgumentParser(description='Train image model with ring loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int, help="number of data loading workers (default: 4)") # String
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128, help="width of an image (default: 128)")

# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int, help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int, help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")

# Triplte Hard Loss
parser.add_argument('--margin', type=float, default=0.3, help='margin for triplet loss')
parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False, help="if this is True, only htri loss is used in training")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1, help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='logs')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help="gpu devices")

args = parser.parse_args()

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):

    losses = AverageMeter()
    xent_losses = AverageMeter()
    triplet_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        # Uncomment if gpu is used
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure elapsed time
        data_time.update(time.time() - end)

        outputs, features = model(imgs)
        xent_loss = criterion_class(outputs, pids)  # Repr loss
        triplet_loss = criterion_metric(features, pids) # Metric loss
        loss = xent_loss + triplet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        xent_losses.update(xent_loss.item(), pids.size(0))
        triplet_losses.update(triplet_loss.item(), pids.size(0))
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("--------------------------------------------------------------------------------------------------------------")
            print("  Epoch: {0} | Batch: [{1}/{2}] | Batch Time: {batch_time.val:.3f} s | Loss: {loss.avg:.4f} | Class Loss: {xent_loss.avg:.4f} | Metric Loss: {triplet_loss.avg:.4f}"
                  .format(epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time, loss=losses, xent_loss=xent_losses, triplet_loss=triplet_losses))
            print("--------------------------------------------------------------------------------------------------------------")

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]

def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        pin_memory = True
    else:
        pin_memory = False

    if not args.evaluate:   # If not test model
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))   # Log file is saved in log_train.txt
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Uncomment when gpu is used
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    dataset = data_manager.init_img_dataset(root=args.root, name=args.dataset)

    # 3 dataloader: train, query, gallery
    # Train needs augmentation
    transform_train = T.Compose([
        transform.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax', 'metric'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_class = nn.CrossEntropyLoss()
    criterion_metric = TriHardLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Decay
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # Uncomment when gpu is used
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only!")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Strat training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0:
            scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
            epoch + 1) == args.max_epoch:
            torch.save(model, 'logs/model_metric.pkl')
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

if __name__ == '__main__':
    main()