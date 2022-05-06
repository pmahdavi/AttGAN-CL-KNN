from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from datasets_n import TextDataset_n
from trainer import condGANTrainer as trainer
from model import RNN_ENCODER, CNN_ENCODER
import torch.nn as nn
from tqdm import tqdm
import pickle
from datasets import prepare_data
from datasets_n import prepare_data_n

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--optim', type=str, default='adam', help='define the optimizer') #PM
    args = parser.parse_args()
    return args

def gen_n(dir): #pm 
    imsize = 64 * (2 ** (3 - 1))
    image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])
    dataset = TextDataset_n(dir, split_dir,
                      base_size=64,
                      transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1000,
    drop_last=True, shuffle=True, num_workers=1)
    data_iter = iter(dataloader)
    
    imgs_t = [] 
    keys_t = []
    for i in range(len(dataloader)):
        data = data_iter.next()
        imgs, _,_,_,_, keys,_,_,_,_,_ = prepare_data_n(data)
        imgs_t.append(imgs[1])
        keys_t = keys_t + keys
    imgs_t = torch.cat(imgs_t) 
    
    image_encoder = CNN_ENCODER(256)
    img_encoder_path = '../DAMSMencoders/bird/image_encoder200.pth'
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()
    image_encoder = image_encoder.cuda()
    
    keys_n = {}
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print("Start generating neighbours...")
    for j in range(imgs_t.shape[0]):
        _ , cnn1 = image_encoder(imgs_t[j:j+1])
        max = -3 
        id_max = -5
        for i in range(imgs_t.shape[0]):
            _ , cnn2 = image_encoder(imgs_t[i:i+1])
            output = cos(cnn1, cnn2)
            if output > max and output!= 1:
                id_max = i
                max = output
            keys_n[keys_t[j]] = keys_t[id_max]
    print("Neigbours are generated")      
    f = open("NN_key.pkl","wb")
    pickle.dump(keys_n,f)
    # close file
    f.close()
    return 

def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.TRAIN.FLAG:
    	dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          	base_size=cfg.TREE.BASE_SIZE,
                          	transform=image_transform)
    else: 
    	dataset = TextDataset_n(cfg.DATA_DIR, split_dir,
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
      
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset,args.optim)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
      	#gen_n(cfg.DATA_DIR)
        algo.train()
        
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
