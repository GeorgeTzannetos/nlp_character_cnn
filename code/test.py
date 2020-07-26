import os
import argparse
from model import CharCNN
from data_loader import Novels
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Character level CNN text classifier testing', formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default='/home/george/Desktop/charCNN/models_CharCNN/CharCNN_epoch_9.pth.tar',
                    help='Path to pre-trained model')
parser.add_argument('--dropout', type=float, default=0.0, help='the probability for dropout [default: 0.5]')
parser.add_argument('--leng', type=int, default=1014, help='maximum length of input sequence to CNNs [default: 1014]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='5,6,7', help='comma-separated kernel size to use for convolution')
# data
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='/home/george/Desktop/charCNN/data/mydata/xtest_obfuscated.txt')
                                                          
parser.add_argument('--batch-size', type=int, default=100, help='batch size for training [default: 128]')
parser.add_argument('--alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
# device
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
# logging options
parser.add_argument('--save-folder', default='Results/', help='Location to save epoch models')
args = parser.parse_args()


if __name__ == '__main__':


    # load testing data
    print("\nLoading testing data...")
    test_dataset = Novels(label_data_path=args.test_path, alphabet_path=args.alphabet_path)
    print("Transferring testing data to iterator...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    _, num_class_test = test_dataset.get_class_weight()
    print('\nNumber of testing samples: '+str(test_dataset.__len__()))

    args.num_features = len(test_dataset.alphabet)
    model = CharCNN(args)
    print("=> loading weights from '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # using GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    size = 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, data in enumerate(test_loader):
        #Here we care only for the inputs.There is no target.It is just a
        # dummy target for simplicity
            inputs, target = data
            size += len(target)
            target = np.asarray(target)
            target = target.astype(int)
            target = torch.LongTensor(target)

            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            inputs = Variable(inputs, volatile=True)
            target = Variable(target)
            logit = model(inputs)
            predicates = torch.max(logit, 1)[1].view(target.size()).data
            predicates_all += predicates.cpu().numpy().tolist()

    predictions = pd.DataFrame(predicates_all)
    predictions.to_csv("/home/george/Desktop/predictions.txt",index=False,header= False)

