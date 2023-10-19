"""Main script for CADNN."""

from importlib.resources import read_binary
import torch
import param_da
from core import eval_src, eval_tgt, train_src, train_tgt
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import read_data, get_data_loader, init_model, init_random_seed,split_train_test,one_hot
from transformers import BertTokenizer
import argparse
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('always') 

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="accident", choices=["accident", "natural haze", "public health event", "social security incident"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="natural haze", choices=["accident", "natural haze", "public health event", "social security incident"],
                        help="Specify tgt dataset")

    parser.add_argument('--enc_train', default=False, action='store_true',
                        help='Train source encoder')

    parser.add_argument('--seqlen', type=int, default=200,
                        help="Specify maximum sequence length")

    parser.add_argument('--patience', type=int, default=5,
                        help="Specify patience of early stopping for pretrain")

    parser.add_argument('--num_epochs_pre', type=int, default=200,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--eval_step_pre', type=int, default=10,
                        help="Specify eval step size for pretrain")

    parser.add_argument('--save_step_pre', type=int, default=100,
                        help="Specify save step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for adaptation")

    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("enc_train: " + str(args.enc_train))
    print("seqlen: " + str(args.seqlen))
    print("patience: " + str(args.patience))
    print("num_epochs_pre: " + str(args.num_epochs_pre))
    print("log_step_pre: " + str(args.log_step_pre))
    print("eval_step_pre: " + str(args.eval_step_pre))
    print("save_step_pre: " + str(args.save_step_pre))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("save_step: " + str(args.save_step))

    # init random seed
    init_random_seed(param_da.manual_seed)

    # preprocess data
    print("=== Processing datasets ===")


    #train test split
    src_=read_data('./CADNN/data/processed/' + args.src + '/' + args.src + '.csv')
    tgt_=read_data('./CADNN/data/processed/' + args.tgt + '/' + args.tgt + '.csv')
    # src_=one_hot(src_)
    # tgt_=one_hot(tgt_)
    label_map={'informative':0,'not_informative':1}
    src_['label']=src_['label'].map(label_map)
    tgt_['label']=tgt_['label'].map(label_map)
    
    src_train, src_test = split_train_test(src_)
    tgt_train, tgt_test = split_train_test(tgt_)


    tokenizer = BertTokenizer.from_pretrained('E:/quz/hashtag_hijack/pre_train_model')

    src_train_sequences = []
    src_test_sequences = []
    tgt_train_sequences = []
    tgt_test_sequences = []

    for i in range(len(src_train.text)):
        tokenized_text = tokenizer.tokenize(src_train.text[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_train_sequences.append(indexed_tokens)

    for i in range(len(src_test.text)):
        tokenized_text = tokenizer.tokenize(src_test.text[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_test_sequences.append(indexed_tokens)

    for i in range(len(tgt_train.text)):
        tokenized_text = tokenizer.tokenize(tgt_train.text[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_train_sequences.append(indexed_tokens)

    for i in range(len(tgt_test.text)):
        tokenized_text = tokenizer.tokenize(tgt_test.text[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_test_sequences.append(indexed_tokens)


    # load dataset
    src_data_loader = get_data_loader(src_train_sequences, src_train.label, args.seqlen)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label, args.seqlen)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label, args.seqlen)

    print("=== Datasets successfully loaded ===")

    # load models
    src_encoder = init_model(BERTEncoder(),
                             restore=param_da.src_encoder_restore)
    src_classifier = init_model(BERTClassifier(),
                                restore=param_da.src_classifier_restore)
    tgt_encoder = init_model(BERTEncoder(),
                             restore=param_da.tgt_encoder_restore)
    critic = init_model(Discriminator(),
                        restore=param_da.d_model_restore)

    # freeze encoder params
    if not args.enc_train:
        for param in src_encoder.parameters():
            param.requires_grad = False

    # train source model
    print("=== Training classifier for source domain ===")
    # if not (src_encoder.restored and src_classifier.restored and
    #         param.src_model_trained):
    src_encoder, src_classifier = train_src(
        args, src_encoder, src_classifier, src_data_loader, src_data_loader_eval)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if not (tgt_encoder.restored and critic.restored and
            param.tgt_model_trained):
        tgt_encoder = train_tgt(args, src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
