import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OCRModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)))
        
        self.lstm = nn.LSTM(1024, 100, bidirectional=True)
        self.fc = nn.Linear(100*2, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)

def zero_pad(x):
    max_w = max(xi.shape[2] for xi in x)
    shape = (1, x[0].shape[1], max_w)

    out = []

    for xi in x:
        o = torch.zeros(shape)
        o[:, :, :xi.shape[2]] = xi
        out.append(o)
   
    return out

def ctc_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    
    input_lengths = [xi.size(2) for xi in x]
    x = zero_pad(x)
    target_lengths = [len(yi) for yi in y]
    labels = torch.IntTensor(np.hstack(y))
    x = torch.stack(x)

    return (x, input_lengths), (labels, target_lengths)
