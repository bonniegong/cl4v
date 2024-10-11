import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from model import MyNet, MyDataset
import torch
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as pkl

np.random.seed(123)
torch.manual_seed(123)

# remove:P =  0.5907335907335908
# R =  0.765
# F1 =  0.6666666666666666

#invert:P =  0.6006914433880726
# R =  0.695
# F1 =  0.6444135373203523

# s1:P =  0.59247889485802
# R =  0.772
# F1 =  0.6704298740772905

# s2:P =  0.5961995249406176
# R =  0.753
# F1 =  0.6654882898806893

if __name__ == "__main__":
    transformer = SentenceTransformer('sentencebert-pretrain-model')
    encoder = LabelEncoder()
    # data = pd.read_csv('/Users/project/CL/pruneanttest/train_qemu_M2_remove.csv')
    data = pd.read_csv('/Users/project/CL/pruneanttest/train_qemu_M2_invert.csv')
    # data = pd.read_csv('/Users/project/CL/pruneanttest/train_qemu_s1.csv')
    # data = pd.read_csv('/Users/project/CL/pruneanttest/train_qemu_s2.csv')
    
    data = data[['code', 'label']]
    train_texts, train_labels = data["code"].values, data["label"].values
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    train_texts = transformer.encode(train_texts)

    train_dataset = MyDataset(train_texts, train_labels, data=train_texts, targets=train_labels)
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,drop_last=False)


    data = pd.read_parquet('/Users/project/CL/test_qemu.parquet')
    data = data[['code', 'label']]
    test_texts, test_labels = data["code"].values, data["label"].values
    encoder.fit(test_labels)
    test_labels = encoder.transform(test_labels)
    test_texts = transformer.encode(test_texts)
    test_dataset = MyDataset(train_texts, train_labels, data=test_texts, targets=test_labels)
    test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,drop_last=False)

    model = MyNet(256,2)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    lossfunction = torch.nn.BCELoss()

    model.train()

    for epoch in range(1, 1001):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target).long()
            data = data.squeeze(0)
            target = target.squeeze(0)
            target = F.one_hot(target,num_classes=2).float()
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunction(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("TrainEpoch: {} \tLoss: {:.6f}".format(epoch,loss.item(),),)

    model.eval()

    outputs = []
    labels = []
    for data, label in test_loader:
        with torch.no_grad():
            data = Variable(data)
            output = model(data)
        outputs.append(output)
        labels.append(label)

    outputs = torch.cat(outputs, dim=0).argmax(axis=1)
    out = outputs.cpu().numpy()
    labels = torch.cat(labels, dim=0)

    print('P = ', precision_score(labels, out))
    print('R = ', recall_score(labels, out))
    print('F1 = ', f1_score(labels, out))







