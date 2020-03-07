import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from utils import FixedHeightResize
from model import OCRModel, ctc_collate
from dataset import OCRDataset
from test import test

IMAGE_HEIGHT = 32

fin_transforms = transforms.Compose([
    transforms.Lambda(lambda x: ImageOps.invert(x)),
    FixedHeightResize(IMAGE_HEIGHT),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ocr_dataset = OCRDataset('../ocr/ocr_data/fin-13k/train/', transform=fin_transforms)

BATCH_SIZE = 32
N_EPOCH = 20
TEST_SIZE = int(len(ocr_dataset) * 0.3)
TRAIN_SIZE = len(ocr_dataset) - TEST_SIZE

X_train, X_test = torch.utils.data.random_split(ocr_dataset, 
                                                (TRAIN_SIZE, TEST_SIZE))

train_loader = torch.utils.data.DataLoader(X_train, batch_size = BATCH_SIZE, 
                                           shuffle = True, 
                                           collate_fn = ctc_collate)

test_loader = torch.utils.data.DataLoader(X_test, batch_size = BATCH_SIZE, 
                                          shuffle = False, 
                                          collate_fn = ctc_collate)

model = OCRModel(num_classes=ocr_dataset.get_num_classes())
model.to(device)

BLANK = ocr_dataset.get_num_classes()-1

criterion = torch.nn.CTCLoss(blank=BLANK, zero_infinity=False)

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(N_EPOCH):
    total_loss = 0

    for i, ((x, input_lengths),(y,target_lengths)) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        #torchvision.utils.save_image(x, f"out{i}.png", nrow=1)
        outputs = model(x)

        loss = criterion(outputs, y, input_lengths=input_lengths,       # outputs = (T, N, C) T=input len, N=batch size, C = nbr of classes
                         target_lengths=target_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))


test(model, ocr_dataset)