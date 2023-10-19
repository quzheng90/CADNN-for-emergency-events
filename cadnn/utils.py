from gettext import npgettext
import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import param_da
import numpy as np
import pandas as pd
import pickle
from torch.autograd import Variable, Function
from random import sample

def one_hot(df):
    label=pd.get_dummies(df['label'],prefix='label')
    df=df.drop('label',axis=1)
    df=pd.concat([df,label],axis=1)
    return df

def split_train_test(df):
    np.random.seed(42)
    split_ratio=0.7
    mask = np.random.rand(len(df)) < split_ratio
    train_df = df[mask]
    test_df = df[~mask]
    train_df=train_df.reset_index(drop=True)
    test_df=test_df.reset_index(drop=True)
    return train_df,test_df


def read_data(file_path_dataset):
    return pd.read_csv(file_path_dataset, delimiter=',',encoding='ansi')


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(param_da.model_root):
        os.makedirs(param_da.model_root)
    torch.save(net.state_dict(),
               os.path.join(param_da.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(param_da.model_root,
                                                             filename)))


def get_data_loader(sequences, labels, maxlen=None):
    # dataset and data loader
    text_dataset = TextDataset(sequences, labels, maxlen)

    text_data_loader = DataLoader(
        dataset=text_dataset,
        batch_size=param_da.batch_size,
        shuffle=True)

    return text_data_loader


def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file)
    file.close()

def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded

def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * l + [mask_idx] * (max_len - l)) for l in lens]
    return mask

def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

def truncate_sents(sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

def get_loss_weight(labels, label_order):
    nums = [np.sum(labels == lo) for lo in label_order]
    loss_weight = torch.tensor([n / len(labels) for n in nums])
    return loss_weight

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()

def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is "+str(len(train[i])))
        print(i)
        #print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp

def make_weights_for_balanced_classes(event, nclasses = 15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight

def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is "+ str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is "+ str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation




class TextDataset(Dataset):
    def __init__(self, sequences, labels, maxlen):

        seqlen = max([len(sequence) for sequence in sequences])

        if maxlen is None or maxlen > seqlen:
            maxlen = seqlen

        seq_data = list()
        for sequence in sequences:
            sequence.insert(0, 101) # insert [CLS] token
            seqlen = len(sequence)
            if seqlen < maxlen:
                sequence.extend([0] * (maxlen-seqlen))
            else:
                sequence = sequence[:maxlen]
            seq_data.append(sequence)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.data = torch.LongTensor(seq_data).cuda()
            self.labels = torch.LongTensor(labels).cuda()
            self.dataset_size = len(self.data)
        else:
            self.data = torch.LongTensor(seq_data)
            self.labels = torch.LongTensor(labels)
            self.dataset_size = len(self.data)

    def __getitem__(self, index):
        review, label = self.data[index], self.labels[index]
        return review, label

    def __len__(self):
        return self.dataset_size
