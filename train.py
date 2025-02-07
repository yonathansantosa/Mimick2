import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, gradcheck
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import numpy as np
import math

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse

from tqdm import trange, tqdm
import os

from distutils.dir_util import copy_tree

def cosine_similarity(tensor1, tensor2, neighbor=5):
    '''
    Calculating cosine similarity for each vector elements of
    tensor 1 with each vector elements of tensor 2

    Input:

    tensor1 = (torch.FloatTensor) with size N x D
    tensor2 = (torch.FloatTensor) with size M x D
    neighbor = (int) number of closest vector to be returned

    Output:

    (distance, neighbor)
    '''
    tensor1_norm = torch.norm(tensor1, 2, 1)
    tensor2_norm = torch.norm(tensor2, 2, 1)
    tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

    divisor = [t * tensor2_norm for t in tensor1_norm]

    divisor = torch.stack(divisor)

    # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
    result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()
    d, n = torch.sort(result, descending=True)
    n = n[:, :neighbor]
    d = d[:, :neighbor]
    return d, n

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)

def pairwise_distances(x, y=None, multiplier=1., loss=False, neighbor=5):
    '''
    Input: 
    
    x is a Nxd matrix
    y is an optional Mxd matirx

    Output: 
    
    dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    if y is not given then use 'y=x'.

    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y *= multiplier
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    if loss:
        result = F.pairwise_distance(x, y)
        return result
    else:
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)) 
        d, n = torch.sort(dist, descending=False)
        n = n[:, :neighbor]
        d = d[:, :neighbor]
        return d, n

def decaying_alpha_beta(epoch=0, loss_fn='cosine'):
    # decay = math.exp(-float(epoch)/200)
    if loss_fn == 'cosine':
        alpha = 1
        beta = 0.5
    else:
        alpha = 0.5
        beta = 1
    return alpha, beta

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=1, help='starting epoch (default=1)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')
parser.add_argument('--model', default='lstm', help='choose which mimick model')
parser.add_argument('--lr', default=0.1, help='learning rate')
parser.add_argument('--charlen', default=20, help='maximum length')
parser.add_argument('--charembdim', default=300)
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--dropout', default=0)
parser.add_argument('--bsize', default=64)
parser.add_argument('--epoch', default=0)
parser.add_argument('--asc', default=False, action='store_true')
parser.add_argument('--quiet', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--nesterov', default=False, action='store_true')
parser.add_argument('--loss_reduction', default=False, action='store_true')
parser.add_argument('--num_feature', default=50)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--momentum', default=0)
parser.add_argument('--multiplier', default=1)
parser.add_argument('--classif', default=200)
parser.add_argument('--neighbor', default=5)
parser.add_argument('--seed', default=64)

args = parser.parse_args()

cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.embedding)
logger_dir = '%s/logs/run%s/' % (saved_model_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_model_path, args.run)

if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
run = int(args.run)
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
random_seed = int(args.seed)
shuffle_dataset = args.shuffle
validation_split = .8
neighbor = int(args.neighbor)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# *Hyperparameter
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
weight_decay = float(args.weight_decay)
momentum = float(args.momentum)
multiplier = float(args.multiplier)
classif = int(args.classif)

char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
emb_dim = dataset.emb_dim

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))


# if shuffle_dataset:
np.random.shuffle(indices)

#* Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)

#* Initializing model
if args.model == 'lstm':
    model = mimick(char_embed.embed, char_embed.char_emb_dim, dataset.emb_dim, int(args.num_feature))
elif args.model == 'cnn2':
    model = mimick_cnn2(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn':
    model = mimick_cnn(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn3':
    model = mimick_cnn3(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        mtp=multiplier, 
        random=False, asc=args.asc)
elif args.model == 'cnn4':
    model = mimick_cnn4(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        classif=classif,
        random=False, asc=args.asc)
else:
    model = None

model.to(device)

if args.loss_fn == 'mse': 
    if args.loss_reduction:
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.MSELoss()
else:
    criterion = nn.CosineSimilarity()

if run < 1:
    args.run = '1'

if run != 1:
    args.load = True

if args.load:
    model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))
    
elif not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
        
word_embedding = dataset.embedding_vectors.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=args.nesterov)

if args.init_weight: model.apply(init_weights)

step = 0
# *Training

for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
    # conv2weight = model.conv2.weight.data.clone()
    # mlpweight = model.mlp[2].weight.data.clone()
    sum_loss = 0.
    for it, (X, y) in enumerate(train_loader):
        model.zero_grad()
        words = dataset.idxs2words(X)
        idxs = char_embed.char_split(words).to(device)
        if args.model != 'lstm': idxs = idxs.unsqueeze(1)
        inputs = Variable(idxs) # (length x batch x char_emb_dim)
        target = Variable(y*multiplier).squeeze().to(device) # (batch x word_emb_dim)

        output = model.forward(inputs) # (batch x word_emb_dim)
        loss = criterion(output, target) if args.loss_fn == 'mse' else (1-criterion(output, target)).mean()

        # ##################
        # Tensorboard
        # ################## 
        sum_loss += loss.item() if not args.loss_reduction else loss.mean().item()
                
        if not args.loss_reduction:
            loss.backward()
        else:
            loss = loss.mean(0)
            for i in range(len(loss)-1):
                loss[i].backward(retain_graph=True)

            loss[len(loss)-1].backward()

        optimizer.step()
        optimizer.zero_grad()

        if not args.quiet:
            if it % int(dataset_size/(batch_size*5)) == 0:
                tqdm.write('loss = %.4f' % loss.mean())
                model.eval()
                random_input = np.random.randint(len(X))
                
                words = dataset.idx2word(X[random_input]) # list of words  

                distance, nearest_neighbor = cosine_similarity(output[random_input].detach().unsqueeze(0), word_embedding, neighbor=neighbor)
                
                loss_dist = torch.dist(output[random_input], target[random_input]*multiplier)
                tqdm.write('%d %.4f | ' % (step, loss_dist.item()) + words + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[0]))
                model.train()
                tqdm.write('')
    writer.add_scalar('Loss/train', sum_loss, epoch)
    model.eval()
  
    ############################
    # SAVING TRAINED MODEL
    ############################

    if not args.local:
        copy_tree(logger_dir, cloud_dir+logger_dir)
        
    torch.save(model.state_dict(), f'{saved_model_path}/{args.model}.pth')
    # torch.save(model.embedding.state_dict(), '%s/charembed.pth' % saved_model_path)

    val_loss = 0.
    for it, (X, target) in enumerate(validation_loader):
        words = dataset.idxs2words(X)
        idxs = char_embed.char_split(words).to(device)
        if args.model != 'lstm': idxs = idxs.unsqueeze(1)
        inputs = Variable(idxs) # (length x batch x char_emb_dim)
        target = target.to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        
        loss = criterion(output, target) if args.loss_fn == 'mse' else (1-criterion(output, target)).mean()
        val_loss += loss.item() if not args.loss_reduction else loss.mean().item()
        
        if not args.quiet:
            if it < 1:
                # distance, nearest_neighbor = pairwise_distances(output.cpu(), word_embedding.cpu())
                distance, nearest_neighbor = cosine_similarity(output, word_embedding, neighbor=neighbor)
                
                for i, word in enumerate(X):
                    if i >= 1: break
                    loss_dist = torch.dist(output[i], target[i])
                    
                    tqdm.write('%.4f | %s \t=> %s' % (loss_dist.item(), dataset.idx2word(word), dataset.idxs2sentence(nearest_neighbor[i])))
                # *SANITY CHECK
                # dist_str = 'dist: '
                # for j in dist[i]:
                #     dist_str += '%.4f ' % j
                # tqdm.write(dist_str)
    # total_val_loss = alpha*cosine_dist + beta*mse_loss
    # total_val_loss = mse_loss
    writer.add_scalar('Loss/Val', val_loss, epoch)

    if not args.quiet: print('total loss = %.8f' % val_loss)