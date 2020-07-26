import csv
import torch
import json
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

class Novels(Dataset):
    def __init__(self, label_data_path, alphabet_path, leng=1014):
        """Create dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv. I combined
            sentences and labels in a csv file.
            leng: max length of a sample. The max length of sentence can be computed to 452 characters.
            So for leng a number bigger than 452 can be used.
            alphabet_path: The path of alphabet json file. Here I have used the default english alphabet
            with all the special characters. Although in this case the sentences do not contain any
            special character, only the 26 letters.
        """
        self.label_data_path = label_data_path
        # read alphabet
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        self.alphabet = alphabet
        self.leng = leng
        self.load()
        self.y = self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y


    def load(self, lowercase=True):
        self.label = []
        self.data = []
        with open(self.label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for index, row in enumerate(rdr):
                self.label.append((row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

    def oneHotEncode(self, idx):
        X = torch.zeros(len(self.alphabet), self.leng)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def get_class_weight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class

if __name__ == '__main__':
    
    label_data_path = '/home/george/Desktop/charCNN/data/mydata/label_data.csv'
    alphabet_path = '/home/george/Desktop/charCNN/alphabet.json'

    train_dataset = Novels(label_data_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)
   
    for i_batch, sample_batched in enumerate(train_loader):
        inputs = sample_batched['data']
        print(inputs.size())

