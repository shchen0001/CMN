import torch, argparse, os
import random, dataset, utils, losses
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.CMN import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
import logging
from timm.utils.log import setup_default_logging
import time
logging.getLogger().setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from tqdm import *

parser = argparse.ArgumentParser()
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='/CMN/logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop,air'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Number of the concepts.'
)
parser.add_argument('--concept-dim', default = 2048, type = int,
    dest = 'concept_dim',
    help = 'dim of concept vector.'
)
parser.add_argument('--batch-size', default = 250, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 40, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 1, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--optimizer', default = 'adamw',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 7e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--weight-decay', default = 5e-4, type =float,
    help = 'Weight decay setting'
)
parser.add_argument('--lr-decay-step', default = 5, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--alpha', default = 32, type = float,
    help = 'Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 8, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 1, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)
parser.add_argument('--lr-backbone', default = 1.0, type = float,
    help = 'lr-backbone setting'
)
parser.add_argument('--lr-concept', default = 1.0, type = float,
    help = 'lr-concept setting'
)
parser.add_argument('--lr-mlp', default = 1.0, type = float,
    help = 'lr-mlp setting'
)
parser.add_argument('--lr-proxy', default = 100.0, type = float,
    help = 'lr-proxy setting'
)
parser.add_argument('--save-model', default = False, type = bool,
    help = 'save-model'
)
parser.add_argument('--a1', default = 0.1, type = float,
    help = 'a1'
)

args = parser.parse_args()

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

time_str = time.strftime("%m-%d-%H-%M", time.localtime())
LOG_DIR = args.LOG_DIR + '/logs_{}/{}'.format(args.dataset, time_str)

os.chdir('/CMN/data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = False
            ))
else:
    trn_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = False
            ))

nb_classes = trn_dataset.nb_classes()

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = False
            ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )
    
else:
    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = False
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = False
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

# Backbone Model
model = CMN(embedding_size=args.sz_embedding, concept_dim=args.concept_dim, 
            pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding, 
                                mrg = args.mrg, alpha = args.alpha).cuda()

param_groups = [
    {'params': model.model.parameters(), 'lr':float(args.lr) * args.lr_backbone},
    {'params': model.MLP1.parameters(), 'lr':float(args.lr) * args.lr_mlp},
    {'params': model.MLP2.parameters(), 'lr':float(args.lr) * args.lr_mlp},
    {'params': model.MLP3.parameters(), 'lr':float(args.lr) * args.lr_mlp},
    {'params': model.concept_v, 'lr':float(args.lr) * args.lr_concept},
]
param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr) * args.lr_proxy})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

if not os.path.exists('{}'.format(LOG_DIR)):
    os.makedirs('{}'.format(LOG_DIR))

setup_default_logging(log_path=LOG_DIR+'/logs.log')
logger.info("=====================begin===========================")
logger.info("Training parameters: {}".format(vars(args)))

best_recall=[0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        unfreeze_model_param = list(filter(lambda p:id(p) not in list(map(id,model.model.parameters())),model.parameters())) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        m = model.forward(x.squeeze().cuda())

        loss1 = criterion(m, y.squeeze().cuda())
        loss2 = model.Concept_Separation_Loss()
        loss = loss1 + args.a1 * loss2

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss_all: {:.6f}, Loss1: {:.6f}, Loss2: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item(), loss1.item(), loss2.item()))

    logger.info('Train Epoch: {}  Loss_all: {:.6f}, Loss1: {:.6f}, Loss2: {:.6f}'.format(
                epoch, loss.item(), loss1.item(), loss2.item()))
    
    scheduler.step()
    
    if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset == 'Inshop':
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                for k,l in enumerate([1, 10, 20, 30, 40, 50]):
                    logger.info("R@{} : {:.3f}".format(l, 100 * Recalls[k]))
            elif args.dataset != 'SOP':
                Recalls = utils.evaluate_cos(model, dl_ev)
                for k,l in enumerate([1, 2, 4, 8, 16, 32]):
                    logger.info("R@{} : {:.3f}".format(l, 100 * Recalls[k]))
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_ev)
                for k,l in enumerate([1, 10, 100, 1000]):
                    logger.info("R@{} : {:.3f}".format(l, 100 * Recalls[k]))
        
        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if args.save_model and best_recall[0]>0.0:
                torch.save(model.state_dict(), '{}/{}_best.pth'.format(LOG_DIR, args.dataset))
            with open('{}/{}_best_results.txt'.format(LOG_DIR, args.dataset), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.dataset == 'Inshop':
                    for i, K in enumerate([1,10,20,30,40,50]):    
                        f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                elif args.dataset != 'SOP':
                    for i in range(6):
                        f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))
                else:
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))