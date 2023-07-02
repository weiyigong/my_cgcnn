import time
import torch, shutil, argparse
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from data import MyDataset
from model import MyModel


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer, epoch):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accus],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        output = model(data)
        target = data.y.unsqueeze(-1)

        loss = loss_criterion(output, target)
        accu = accuracy_criterion(output, target)

        losses.update(loss.item(), target.size(0))
        accus.update(accu.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

    return losses.avg, accus.avg


def validate(model, device, test_loader, loss_criterion, accuracy_criterion):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, accus],
        prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):

            data = data.to(device, non_blocking=True)
            output = model(data)
            target = data.y.unsqueeze(-1)

            loss = loss_criterion(output, target)
            accu = accuracy_criterion(output, target)

            losses.update(loss.item(), target.size(0))
            accus.update(accu.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    return accus.avg


def main():
    parser = argparse.ArgumentParser(description='CGCNN')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='trainning ratio')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='Adam',
                        help='choose an optimizer, SGD or Adam, (default:Adam)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--drop-last', default=False, type=bool)
    parser.add_argument('--pin-memory', default=True, type=bool)

    args = parser.parse_args()

    best_accu = 1e6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dataset = MyDataset('cifs')

    n_data = len(dataset)
    train_split = int(n_data * args.train_ratio)
    dataset_train, dataset_val = random_split(
        dataset,
        [train_split, len(dataset) - train_split],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    data0 = dataset[0]
    model = MyModel(data0.x.size(-1), data0.edge_attr.size(-1), num_layers=5, h_dim=256)
    model.to(device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer must be SGD or Adam.')
    loss_criterion = torch.nn.MSELoss()
    accuracy_criterion = torch.nn.L1Loss()

    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_accu = train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer,
                                       epoch)
        val_accu = validate(model, device, val_loader, loss_criterion, accuracy_criterion)
        # scheduler.step()

        is_best = val_accu < best_accu
        best_accu = min(val_accu, best_accu)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }, is_best)

    # ckpt = torch.load('model_best.pth.tar')
    # print(ckpt['best_accu'])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
