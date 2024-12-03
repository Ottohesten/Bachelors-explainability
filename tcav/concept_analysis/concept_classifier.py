from captum.concept import Classifier
import torch
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np


class TCAVClassifier(Classifier):
    def __init__(self):
        self.lm = linear_model.SGDClassifier(alpha=0.01, max_iter=10000, tol=1e-3)

    def train_and_eval(self, dataloader):
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        # normalize inputs (mean 0, std 1)
        inputs = [(input - input.mean()) / input.std() for input in inputs]
        x_train, x_test, y_train, y_test = train_test_split(torch.cat(inputs), torch.cat(labels), random_state=42)
        self.lm.fit(x_train.detach().numpy(), y_train.detach().numpy())
        preds = torch.tensor(self.lm.predict(x_test.detach().numpy()))
        return {'accs': (preds == y_test).float().mean()}

    def weights(self):
        if len(self.lm.coef_) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self):
        return self.lm.classes_