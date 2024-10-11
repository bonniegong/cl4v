import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator

import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from cleanlab.classification import CleanLearning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from torch.autograd import Variable

from model import MyNet, MyDataset

lossfunction = torch.nn.BCELoss()


np.random.seed(1234)
torch.manual_seed(1234)


def get_dataset():
    dataset = MyDataset(train_texts, train_labels)
    return dataset, len(train_labels), 0


class MyEstimator(BaseEstimator):
    def __init__(
        self,
        batch_size=64,
        epochs=1000,
        log_interval=200,  # Set to None to not print
        lr=0.01,
        momentum=0.5,
        no_cuda=None,
        seed=1,
        test_batch_size=64,
        loader=None,
        dataset_emb = None,
        dataset_label = None
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        if self.cuda:  # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.model = MyNet(256,2)
        if self.cuda:  # pragma: no cover
            self.model.cuda()

        self.loader_kwargs = {"num_workers": 1, "pin_memory": True} if self.cuda else {}
        self.loader = loader
        self.test_batch_size = test_batch_size
        self.dataset, self.train_size, self.test_size = get_dataset()

  
    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "log_interval": self.log_interval,
            "lr": self.lr,
            "momentum": self.momentum,
            "no_cuda": self.no_cuda,
            "test_batch_size": self.test_batch_size,
            # "dataset": self.dataset,
        }

    def set_params(self, **parameters):  # pragma: no cover
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.dataset, self.train_size, self.test_size = get_dataset()
        return self

    def fit(self, train_idx, train_labels=None, sample_weight=None, loader="train"):
        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError("Check that train_idx and train_labels are the same length.")

        if sample_weight is not None:  # pragma: no cover
            if len(sample_weight) != len(train_labels):
                raise ValueError(
                    "Check that train_labels and sample_weight " "are the same length."
                )
            class_weight = sample_weight[np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None

        train_dataset = self.dataset

        # Use provided labels if not None o.w. use MNIST dataset training labels
        if train_labels is not None:
            # Create sparse tensor of train_labels with (-1)s for labels not
            # in train_idx. We avoid train_data[idx] because train_data may
            # very large, i.e. ImageNet
            sparse_labels = (
                np.zeros(self.train_size if loader == "train" else self.test_size, dtype=int) - 1
            )
            sparse_labels[train_idx] = train_labels
            train_dataset.targets = sparse_labels
            train_dataset.data = train_dataset.emb

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            # sampler=SubsetRandomSampler(train_idx if train_idx is not None
            # else range(self.train_size)),
            sampler=BatchSampler(train_idx, 
            batch_size=self.batch_size, 
            drop_last=False,
            **self.loader_kwargs)
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):
            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:  # pragma: no cover
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target).long()
                data = data.squeeze(0)
                target = target.squeeze(0)
                target = F.one_hot(target).float()
                optimizer.zero_grad()
                output = self.model(data)
                loss = lossfunction(output, target)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and batch_idx % self.log_interval == 0:
                    print(
                        "TrainEpoch: {} \tLoss: {:.6f}".format(
                            epoch,
                            loss.item(),
                        ),
                    )

    def predict(self, idx=None, loader=None):
        """Get predicted labels from trained model."""
        # get the index of the max probability
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None, loader=None):
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = (
                idx is not None
                and len(idx) == self.test_size
                and np.all(np.array(idx) == np.arange(self.test_size))
            )
            loader = "test" if is_test_idx else "train"
        dataset = self.dataset
        # Filter by idx
        if idx is not None:
            if (loader == "train" and len(idx) != self.train_size) or (
                loader == "test" and len(idx) != self.test_size
            ):
                dataset.data = dataset.data[idx]
                dataset.targets = dataset.targets[idx]

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if loader == "train" else self.test_batch_size,
            **self.loader_kwargs,
        )

        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()

        # Run forward pass on model to compute outputs
        outputs = []
        for data, _ in loader:
            if self.cuda:  # pragma: no cover
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = self.model(data)
            outputs.append(output)

        # Outputs are log_softmax (log probabilities)
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        # pred = np.exp(out)
        return out




if __name__ == "__main__":
    global train_texts, train_labels

    data = pd.read_parquet('/Users/project/CL/train_qemu.parquet')
    data = shuffle(data)
    dataindex = list(data['newidx'])
    data = data[['code', 'label']]


    raw_texts, raw_labels = data["code"].values, data["label"].values

    encoder = LabelEncoder()
    encoder.fit(raw_labels)

    train_labels = encoder.transform(raw_labels)

    transformer = SentenceTransformer('sentencebert-pretrain-model')

    train_texts = transformer.encode(raw_texts)

    model = MyEstimator()

    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better

    cl = CleanLearning(model, cv_n_folds=cv_n_folds)

    label_issues = cl.find_label_issues(X=np.array(list(range(len(train_labels)))), labels=train_labels)
    label_issues['newidx'] = dataindex

    label_issues.to_csv('/Users/project/CL/clmethods/label_issues.csv')
