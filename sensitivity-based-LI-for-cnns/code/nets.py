import torch.nn as nn

def init_vgg(model_baseline):
    # no specififc initalization for now
    return model_baseline

def build_vgg_baseline(BN=False, small = False, m=None):
    if small:
        model = VGG_small(m) 
        model = init_vgg(model)

    else:
        if BN:
            if m is not None:
                print('Error: m must be None for BN=True')
                return 0
            model= VGG_BN()
            model = init_vgg(model)
        else: 
            model = VGG(m)
            model = init_vgg(model)
    return model

class VGG(nn.Module):
    """
    Baseline VGG implementation for the  CIFAR10 DATASET
    """
    def __init__(self,m=[64,128,256]):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, m[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # conv2 16x16-8x8
            nn.Conv2d(m[0], m[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv3 8x8-4x4
            nn.Conv2d(m[1], m[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(m[2] * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    


def get_fullext_model(old_model):
    old_child=None # from last iteration
    list_fullyext_children=[]
    for i,child in enumerate(old_model.children()):
        for j,subchild in enumerate(child.children()):
            setattr(subchild, 'freezed', False)
            list_fullyext_children.append(subchild)
            # if correct add new conv + relu
            if isinstance(subchild,nn.ReLU) and isinstance(old_child,nn.Conv2d): # old child must be conv, subchild must be relu
                # get no of channels m for new kernel
                m=old_child.out_channels # channel out dim of old child
                # freeze new conv and relu via
                new_conv_child = nn.Conv2d(m, m, 3, padding=1)
                setattr(new_conv_child, 'freezed', True)
                new_relu_child = nn.ReLU()
                setattr(new_relu_child, 'freezed', True)
                list_fullyext_children.append(new_conv_child)
                list_fullyext_children.append(new_relu_child)
            # update old child
            old_child = subchild
            
        
        if i==0: 
            Flatten_with_attr = nn.Flatten()
            setattr(Flatten_with_attr, 'freezed', False)
            list_fullyext_children.append(Flatten_with_attr)
                
        
    net = nn.Sequential(*list_fullyext_children)
    return net


def extend_VGG(position, BN=False): # only relevant for manual layer insertion
    '''
    generates model which is extended by one layer.

    Args:
        position: either 0,1,2 dep on the position in the vgg baseline
        BN: indicates whether batch normalization is used in the architecture
    '''
    if position not in [0,1,2]:
        print(F'Error: {position} is not feasible!')
        return 0
    
    class VGG_ext(nn.Module):
        def __init__(self,BN=False, position=0):
            super().__init__()

            modules = []

            modules.append(nn.Conv2d(3, 64, 3, padding=1))
            if position==0:
                if BN:
                    modules.append(nn.BatchNorm2d(64))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(64,64,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(64))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))


            modules.append(nn.Conv2d(64, 128, 3, padding=1))
            if position==1:
                if BN:
                    modules.append(nn.BatchNorm2d(128))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(128,128,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(128))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))

            modules.append(nn.Conv2d(128, 256, 3, padding=1))
            if position==2:
                if BN:
                    modules.append(nn.BatchNorm2d(256))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(256,256,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(256))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))

            self.features = nn.Sequential(*modules)

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500,10,bias=False)
            )

            
        def forward(self, x):
            for layer in self.features:
                if isinstance(layer, nn.MaxPool2d):
                    x, location = layer(x)
                else:
                    x = layer(x)
            
            x = x.view(x.size()[0], -1)
            x = self.classifier(x)
            return x
    
    model = VGG_ext(BN, position)
    print(f'Attention! new weight has no identity layer initialization but random!')
    return model

def extend_VGG_new(old_model, position, BN=False):
    # if type(position) is not integer return error
    if not isinstance(position, int) or position < 0:
        print(F'Error: {position} is not feasible!')
        return old_model

    old_child=None # from last iteration
    curr_pos = -1
    list_fullyext_children_class=[]
    list_fullyext_children_features=[]
    new_layer_created=False
    for i,child in enumerate(old_model.children()):
        for j,subchild in enumerate(child.children()):
            if i==0:
                list_fullyext_children_features.append(subchild)
                if isinstance(subchild,nn.ReLU) and isinstance(old_child,nn.Conv2d):
                    curr_pos+=1
                # add new conv + relu on correct position
                if curr_pos == position and new_layer_created==False:
                    # get no of channels m for new kernel
                    m=old_child.out_channels
                    list_fullyext_children_features.append(nn.Conv2d(m, m, 3, padding=1))
                    list_fullyext_children_features.append(nn.ReLU())
                    new_layer_created=True
                old_child = subchild
            if i==1:
                list_fullyext_children_class.append(subchild)

    class VGG_ext(nn.Module):        
        def __init__(self,BN=False, position=0):
            super().__init__()

            self.features = nn.Sequential(*list_fullyext_children_features)

            self.classifier = nn.Sequential(*list_fullyext_children_class)

        def forward(self, x):
            for layer in self.features:
                if isinstance(layer, nn.MaxPool2d):
                    x, location = layer(x)
                else:
                    x = layer(x)
            
            x = x.view(x.size()[0], -1)
            x = self.classifier(x)
            return x
    
    model = VGG_ext(BN, position)
    #print(f'Attention! new weight has no identity layer iniitaliaztion but random!')
    return model
            



















############################ not relevant #################################################


def get_fullext_model_old(BN=True): #relevant
    if BN:
        net = nn.Sequential(
            nn.Conv2d(3,64,3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4096,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500, 10, bias=False)
        )
    else:
        net = nn.Sequential(
            nn.Conv2d(3,64,3, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(64,64,3, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4096,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500, 10, bias=False)
        )

    # set freezed attribute of layers:
    if BN:
        freezed_list = [3,4,5,10,11,12,17,18,19]
    else:
        freezed_list = [2,3,7,8,12,13]
        
    for i, layer in enumerate(net.children()):
        if i not in freezed_list:
            setattr(layer, 'freezed', False)
        else:
            setattr(layer, 'freezed', True)
    return net


class VGG_BN(nn.Module):
    """
    Baseline VGG implementation for the  CIFAR10 DATASET with BN.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # conv2 16x16-8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv3 8x8-4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

class VGG_small(nn.Module):
    """
    Baseline VGGsmall implementation for the  MNIST DATASET.
    """
    def __init__(self, m):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(1, m[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # conv2 16x16-8x8
            nn.Conv2d(m[0], m[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv3 8x8-4x4
            nn.Conv2d(m[1], m[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(9 * m[2], 10),
            nn.ReLU(),
            nn.Linear(10,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x




def get_fullext_model_small(m):

    net = nn.Sequential(
            nn.Conv2d(1,m[0],3, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(m[0],m[0],3, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(m[0], m[1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(m[1],m[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(m[1], m[2], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(m[2], m[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9*m[2],10),
            nn.ReLU(),
            nn.Linear(10, 10, bias=False)
        )
    
    freezed_list = [2,3,7,8,12,13]

    for i, layer in enumerate(net.children()):
        if i not in freezed_list:
            setattr(layer, 'freezed', False)
        else:
            setattr(layer, 'freezed', True)

    return net












































#model = build_vgg_baseline(BN=False)
#print(model)
#print(model.state_dict())
# for child in model.children():
#     print(child)
#     print(child.freezed)
#     if hasattr(child, 'weight'):
#         print(child.weight.shape)
#         print(child.__dict__)
#     print('-------------------')