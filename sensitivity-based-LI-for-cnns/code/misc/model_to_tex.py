from pytorch2tikz import Architecture
import torch
import sys
from os.path import dirname, abspath, join


sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')

from nets import build_vgg_baseline, extend_VGG
from data_preparation import prep_cifar10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = prep_cifar10()


model = build_vgg_baseline(BN=True)
model.eval()
arch  = Architecture(model)

with torch.inference_mode():
    for image, _ in trainloader:
        image = image.to(device, non_blocking=True)
        output = model(image)
print('yay')
#print('Write result to ___.tex')
#arch.save('tikz_models/baseline1.tex')