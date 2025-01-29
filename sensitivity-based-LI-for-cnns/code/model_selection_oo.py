from typing import List, Tuple, Callable
from itertools import chain
import copy

import torch
import numpy as np

import utils
import nets
import backpack


class ExtendableModel:
    """
    Base class for extendable models.
    """

    def __init__(self, device=None, use_kfac = False) -> None:
        """
        Initialize the ExtendableModel.

        Args:
            device (torch.device, optional): The device to use for computations. Defaults to None.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = None
        self.freezed = []
        self.selection_phase = False
        self.use_kfac = use_kfac

    def fully_extend(self) -> None:
        """
        Fully extend the model.
        """
        self.fully_ext_model = self._build_fullyext() 
        print(f'Fully extended model: {self.fully_ext_model}')
        self._init_fully_extended() 
        self.selection_phase = True
        self.fully_ext_model.to(self.device)

    def _build_fullyext(self):
        """
        Build the fully extended model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.

        Returns:
            The fully extended model.
        """
        raise NotImplementedError()

    def find_insertion_points(self):
        """
        Find the insertion points in the model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _get_layer_iterator(self):
        """
        Get an iterator over the layers of the original model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _get_new_layer_iterator(self):
        """
        Get an iterator over the layers of the fully extended model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _get_all_layer_iterator(self):
        """
        Get an iterator over all layers of the model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _copy_layer(self):
        """
        Copy the parameters of a layer from the original model to the fully extended model.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _layer_is_relevant(self, layer):
        """
        Check if a layer is relevant for selection.

        Args:
            layer: The layer to check.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _layer_metric(self, layer):
        """
        Calculate the metric for a layer.

        Args:
            layer: The layer to calculate the metric for.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _layer_predecessor(self, layer):
        """
        Get the predecessor layers of a given layer.

        Args:
            layer: The layer to get the predecessor layers for.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _layer_successor(self, layer):
        """
        Get the successor layers of a given layer.

        Args:
            layer: The layer to get the successor layers for.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _handle_new_layer_list(self, new_layer_list):
        """
        Handle the new layer list after selecting a layer.

        Args:
            new_layer_list: The new layer list.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    def _select_layer(self, rule: str = None, sens_norm: str = None):
        """
        Select a layer based on a given rule.

        Args:
            rule (str): The rule to use for layer selection. Includes 'abs max', 'abs min', 'threshold_abs', 'threshold_rel', 'pos0', 'pos1', 'pos2'. Defaults to 'abs max'.
            sens_norm (str): The sensitivity norm to use. Defaults to 'conv_2'. Includes 'fro_squared', 'fro_squared_scaled', 'conv_1', 'conv_2'.

        Returns:
            The selected layer.
        """
        if rule is None:
            rule = 'abs max'
        if sens_norm is None:
            sens_norm = 'fro_squared_scaled'

        sens = []
        g = self.full_grad_sq_scaled()


        
        # start position finding...
        
        if rule == 'abs max' or rule == 'abs min':
            if rule == 'abs max':
                comp = lambda x, y: x > y
                curr_metric = -1
            else:
                comp = lambda x, y: x < y
                curr_metric = 1e5

            layer_to_insert = None
            
            pos = -1
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    sens.append(self._layer_metric(layer, sens_norm))
                    if layer_to_insert is None or comp(self._layer_metric(layer, sens_norm), curr_metric):
                        pos+=1
                        layer_to_insert = layer
                        curr_metric = self._layer_metric(layer, sens_norm)
            print(f'Best sens {curr_metric}')
            if rule == 'abs max':
                pos = sens.index(max(sens))
            if rule == 'abs min':
                pos = sens.index(min(sens))
            # save all norms of sens in attribute sensitivities_all
            if sens_norm=='all':
                norms_layers = []
                for layer in self._get_new_layer_iterator():
                    if layer.freezed and self._layer_is_relevant(layer):
                        norms_layers.append(layer.sens_norms_all)
                self.sensitivities_all = norms_layers
            return layer_to_insert, pos, sens
        
        if rule == 'threshold_abs':
            threshold = 1e-4
            comp = lambda x, y: x > y
            layer_to_insert = None
            curr_metric = -1
            pos = -1
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    sens.append(self._layer_metric(layer, sens_norm))
                    if layer_to_insert is None or comp(self._layer_metric(layer, sens_norm), curr_metric):
                        pos+=1
                        layer_to_insert = layer
                        curr_metric = self._layer_metric(layer, sens_norm)
            print(f'best sens {curr_metric} and threshold {threshold}')
            pos = sens.index(max(sens))
            if curr_metric <= threshold:
                layer_to_insert = None
                pos = None
            return layer_to_insert, pos, sens
        
        if rule == 'threshold_rel':
            threshold = 1.
            denom = self.denominator_threshold(sens_norm)
            comp = lambda x, y: x > y
            layer_to_insert = None
            curr_metric = -1
            pos = -1
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    pos+=1
                    sens.append(self._layer_metric(layer, sens_norm))
                    if layer_to_insert is None or comp(self._layer_metric(layer, sens_norm), curr_metric):
                        layer_to_insert = layer
                        curr_metric = self._layer_metric(layer, sens_norm)
            print(f'best sens {curr_metric} and denom {denom} and threshold {threshold}')
            pos = sens.index(max(sens))
            if curr_metric/denom <= threshold:
                layer_to_insert = None
                pos = None
            return layer_to_insert, pos, sens

        if rule == 'pos0':
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    sens.append(self._layer_metric(layer, sens_norm))
            for layer in self._get_new_layer_iterator():        
                if layer.freezed and self._layer_is_relevant(layer):
                    return layer, 0, sens

        if rule == 'pos2':
            res = None
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    sens.append(self._layer_metric(layer, sens_norm))
                if layer.freezed and self._layer_is_relevant(layer):
                    res = layer
            return res, 2, sens
        
        if rule == 'pos1': 
            p=0
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    sens.append(self._layer_metric(layer, sens_norm))
            for layer in self._get_new_layer_iterator():
                if layer.freezed and self._layer_is_relevant(layer):
                    if p==1:
                        return layer, 1, sens
                    p+=1
                    


    @torch.no_grad()
    def _init_fully_extended(self):
        """
        Initialize the fully extended model by copying the parameters from the original model.
        """
        # features and classifier part in one iterator for old and fe model
        old_child_iterator = self._get_layer_iterator() # baseline chained with flatten children

        for new_layer in self._get_new_layer_iterator(): # fully extended model children
            if new_layer.freezed:
                self._to_id(new_layer)
            else:
                child_old = next(old_child_iterator)
                self._copy_layer(child_old, new_layer)

    def _number_of_frozen_parameters_collected(self) -> int:
        """
        Count the total number of frozen parameters in the model.

        Returns:
            The total number of frozen parameters.
        """
        no = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if utils.is_freezed(p, self.freezed):
                    no += 1
        return no

    def number_of_free_parameters_collected(self) -> int:
        """
        Count the total number of not-frozen parameters in the model.

        Returns:
            The total number of not-frozen parameters.
        """
        no = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if not utils.is_freezed(p, self.freezed):
                    no += 1
        return no

    def calculate_shadow_prices_mb(self, traindataloader, loss_fn=None):
        """
        Calculate the shadow prices for the model.

        Args:
            traindataloader: The training dataloader.
            loss_fn: The loss function to use. Defaults to None.
        """
        assert self.selection_phase, 'Model is not fully extended'

        grad_norms_paramwise = []
        for p in self.fully_ext_model.parameters():
            grad_norms_paramwise.append([]) # one list for each parameter

        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        if self.use_kfac:
            print('extending loss function with backpack...')
            backpack.extend(self.fully_ext_model)
            backpack.extend(loss_fn)

        for layer in self._get_new_layer_iterator():
            if hasattr(layer, 'weight'):
                setattr(layer, 'sensitivities_w', torch.zeros(layer.weight.shape))
                if layer.bias is not None:
                    setattr(layer, 'sensitivities_b', torch.zeros(layer.bias.shape))
            
            setattr(layer, 'sensitivity_w', 0)
            if hasattr(layer,'bias'):
                setattr(layer, 'sensitivity_b', 0)
            

        print('Starting sensitivity calculation epoch...')
        for X, y in traindataloader:
            X, y = X.to(self.device), y.to(self.device)

            self.fully_ext_model.zero_grad()
            loss = loss_fn(self.fully_ext_model(X), y)
            if self.use_kfac:
                with backpack.backpack(backpack.extensions.KFAC(),backpack.extensions.SumGradSquared()):
                    loss.backward()
            else:
                loss.backward()


            with torch.no_grad():
                layer = 0
                for p in self.fully_ext_model.parameters():
                    grad_norms_paramwise[layer].append(
                        torch.square(p.grad).cpu().sum().item())
                    layer += 1

            with torch.no_grad():
                for layer in self._get_new_layer_iterator():
                    if hasattr(layer, 'weight'): 
                        layer.sensitivities_w += layer.weight.grad.cpu()
                        if layer.bias is not None:
                            layer.sensitivities_b += layer.bias.grad.cpu()

                        

        rel_layers = [layer for layer in self._get_new_layer_iterator() if isinstance(layer, torch.nn.Conv2d) and layer.freezed==True]

        print('Sensitivity calculation epoch done!')
        if self.use_kfac:
            print('Start calculating KFAC matrices...')
            self.relevant_param_index = [2,6,10]

            list_params = list(self.fully_ext_model.parameters())
            list_rel_params = [list_params[i] for i in range(len(list_params)) if i in self.relevant_param_index]
            # for each layer generate kfac matrices
            damping = 1e-2
            list_kfacs = []
            
            i = 0
            
            for p in list_rel_params:
                print(f'Inverting KFAC matrices for layer {i}...')
                kfac_sen = utils.inverse_by_cholesky(p.kfac[0], damping) 
                kfac_act = utils.inverse_by_cholesky(p.kfac[1], damping) 
                list_kfacs.append(tuple((copy.deepcopy(kfac_sen), copy.deepcopy(kfac_act))))  
                
                # det layer.sensitivity_w by doing dot(g, mat_to_grads(kfac_sen @ grads_to_mat(g)@ kfac_act))
                layer_curr = rel_layers[i]
                print(f'Calculating kfac-precond sensitivity for layer {i}...')
                layer_curr.sensitivity_w = torch.dot(torch.flatten(layer_curr.sensitivities_w), torch.flatten(kfac_sen @ utils.grads_to_mat(layer_curr.sensitivities_w) @ kfac_act))
                i +=1
            self.kfacs = list_kfacs

        self.ext_model_layer_grads = grad_norms_paramwise

        scale = 1/len(traindataloader)
        # set layer.sensitivity
        with torch.no_grad():
            for layer in self._get_new_layer_iterator():
                if hasattr(layer, 'weight'): 
                    layer.sensitivity_w += torch.sum(torch.square(scale*layer.sensitivities_w))
                    if layer.bias is not None:
                        layer.sensitivity_b += torch.sum(torch.square(scale*layer.sensitivities_b))


    def denominator_threshold(self, sens_norm):
        if True:
            denom=0
            with torch.no_grad():
                count=0
                for layer in self._get_new_layer_iterator():
                    if hasattr(layer, 'weight') and isinstance(layer, torch.nn.Conv2d):
                        if sens_norm == 'all':
                            sens_norm = 'op22'
                        if sens_norm == 'fro':
                            denom += layer.sensitivity_w
                        elif sens_norm=='fro_scaled':
                            denom += layer.sensitivity_w / utils.prod(layer.weight.shape)
                        elif sens_norm == 'op22':
                            input_shape = self.calc_input_shape(layer)
                            denom += np.square(utils.max_sing_value(layer.sensitivities_w, input_shape))
                        elif sens_norm == 'op22_1':
                            input_shape = self.calc_input_shape(layer)
                            no_out_channels = layer.sensitivities_w.shape[0]
                            norm=0
                            for i in range(no_out_channels):
                                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                                norm += s
                            norm=np.square(norm)
                            denom += norm
                        elif sens_norm == 'op22_2':
                            input_shape = self.calc_input_shape(layer)
                            no_out_channels = layer.sensitivities_w.shape[0]
                            norm = 0
                            for i in range(no_out_channels):
                                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                                norm += s*s
                            denom += norm
                        count += 1
                denom = denom/count
            return denom
    


    def weight_grad_scaled(self,layer):
        return layer.sensitivity_w / utils.prod(layer.weight.shape)
    
    def full_grad_sq_scaled(self):
        if self.use_kfac: 
            g = 0
            for p in self.fully_ext_model.parameters():
                if hasattr(p, 'sum_grad_squared'):
                    no = utils.prod(p.shape)
                    g += torch.sum(torch.square(p.grad))/no
                    
            self.ext_full_grad_sq_scaled = g
            return g 
        else:
            g=0
            no =0
            for layer in self._get_new_layer_iterator():
                if hasattr(layer, 'weight') and not layer.freezed:
                    g += layer.sensitivity_w 
                    no += utils.prod(layer.weight.shape)
                    if layer.bias is not None:
                        g += layer.sensitivity_b 
                        no += utils.prod(layer.bias.shape)
            g /= no
            self.ext_full_grad_sq_scaled = g.cpu().numpy()      
        return g    


    def select_new_model(self, rule, sens_norm) -> None:
        """
        Select a new model based on the selected layer.
        """
        layer_to_insert, pos, sens = self._select_layer(rule, sens_norm)
        if layer_to_insert is not None:
            self.new_weight_param = layer_to_insert.weight
        # turn sens tensor to no tensor
        sens = [s.item() for s in sens]
        print('All sensitivities:', sens)
        print(f'Position {pos} chosen for layer insertion!')
        self.pos = pos
        self.sensitivities = sens
        if self.pos is not None:
            new_block = self._layer_predecessor(layer_to_insert) + [layer_to_insert] + self._layer_successor(layer_to_insert)

            layers = []

            for layer in self._get_new_layer_iterator():
                if not layer.freezed:
                    layers.append(layer)
                    continue
                if layer == layer_to_insert:
                    layers += new_block
                    continue
            self._handle_new_layer_list(layers)
        
        self.selection_phase = False


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.model(x)
    
    def __call__(self,x):
        return self.forward(x)

    @staticmethod
    @torch.no_grad()
    def _to_id(layer):
        """
        Set the weights and biases of a layer to identity.

        Args:
            layer: The layer to set the weights and biases to identity.
        """
        raise NotImplementedError()


class VGGExtendableModel(ExtendableModel):
    """
    Extendable VGG model.
    """

    def __init__(self,m, BN=False, use_kfac = False) -> None:
        """
        Initialize the VGGExtendableModel.

        Args:
            BN (bool, optional): Whether to use Batch Normalization. Defaults to False.
        """
        super().__init__()
        self.model = nets.build_vgg_baseline(BN,small = False,m=m)
        self.BN = BN
        self.use_kfac = use_kfac

    def _build_fullyext(self):
        """
        Build the fully extended VGG model.

        Returns:
            The fully extended VGG model.
        """
        return nets.get_fullext_model(self.model)
    
    def _get_layer_iterator(self):
        """
        Get an iterator over the layers of the original VGG model.
        """
        return chain(chain(self.model.features.children(),torch.nn.Sequential(torch.nn.Flatten())), self.model.classifier.children())

    def _get_new_layer_iterator(self):
        """
        Get an iterator over the layers of the fully extended VGG model.
        """
        return self.fully_ext_model.children()

    def _layer_is_relevant(self, layer):
        """
        Check if a layer is relevant for selection.

        Args:
            layer: The layer to check.

        Returns:
            True if the layer is relevant, False otherwise.
        """
        return isinstance(layer, torch.nn.Conv2d)
    
    def calc_input_shape(self, layer):
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding

        shape = layer.sensitivities_w.shape
        output_shape = shape[2]

        s = (output_shape - 1) * stride - 2 * padding + kernel_size
        return (s,s)


    def _layer_metric(self, layer, sens_norm):
        """
        Calculate the metric for a layer.

        Args:
            layer: The layer to calculate the metric for.
            sens_norm: The sensitivity norm to use. Includes ['all', 'fro_squared', 'fro_squared_scaled','op22_1','op22_2','op22']. 
            if 'all' is chosen, all norms are calculated and op22_1 is used as a decision metric.
        Returns:
            The metric value for the layer.
        """

        if sens_norm=='all':
            
            input_shape = self.calc_input_shape(layer)
            

            norm_fro=layer.sensitivity_w

            norm_fro_scaled=layer.sensitivity_w / utils.prod(layer.weight.shape)
            no_out_channels = layer.sensitivities_w.shape[0]
            norm_op22_1=0 
            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm_op22_1 += s
            norm_op22_1 = np.square(norm_op22_1)
            

            norm_op22_2= 0 
            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm_op22_2 += s*s
            

            norm_op22 = np.square(utils.max_sing_value(layer.sensitivities_w, input_shape))

            norm = norm_op22 
            norms = [norm_fro.item(), norm_fro_scaled.item(), norm_op22_1, norm_op22_2, norm_op22]
            # make norms attribute of layer
            setattr(layer, 'sens_norms_all', norms)

        if sens_norm=='fro_squared':
            norm = layer.sensitivity_w
        if sens_norm=='fro_squared_scaled':
            norm = layer.sensitivity_w / utils.prod(layer.weight.shape)
        if sens_norm=='op22_1':
            no_out_channels = layer.sensitivities_w.shape[0]
            input_shape = self.calc_input_shape(layer)
            norm=0
            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm += s
            norm=np.square(norm)

        if sens_norm=='op22_2':
            no_out_channels = layer.sensitivities_w.shape[0]
            norm = 0 
            input_shape = self.calc_input_shape(layer)

            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm += s*s
            
            
        if sens_norm=='op22':
            input_shape = self.calc_input_shape(layer)
            
            norm = np.square(utils.max_sing_value(layer.sensitivities_w, input_shape))

        return norm

    def _handle_new_layer_list(self, new_layer_list): 
        """
        Handle the new layer list after selecting a layer.

        Args:
            new_layer_list: The new layer list.
        """

        layer_feat = []
        layer_class = []

        flag_for_flatten = False
        for layer in new_layer_list:
            if isinstance(layer, torch.nn.Flatten):
                flag_for_flatten = True
                continue
            if not flag_for_flatten:
                layer_feat.append(layer)
            else:
                layer_class.append(layer)
            

        self.model.features = torch.nn.Sequential(*layer_feat)
        self.model.classifier = torch.nn.Sequential(*layer_class)

        

    def _layer_predecessor(self, layer):
        """
        Get the predecessor layers of a given layer.

        Args:
            layer: The layer to get the predecessor layers for.

        Returns:
            The predecessor layers.
        """
        return []

    def _layer_successor(self, layer):
        """
        Get the successor layers of a given layer.

        Args:
            layer: The layer to get the successor layers for.

        Returns:
            The successor layers.
        """
        res = []
        if self.BN:
            res.append(torch.nn.BatchNorm2d(layer.out_channels))
        res.append(torch.nn.ReLU())
        return res

    @torch.no_grad()
    def _copy_layer(self, old_layer, new_layer):
        """
        Copy the parameters of a layer from the original VGG model to the fully extended VGG model.

        Args:
            old_layer: The layer from the original VGG model.
            new_layer: The layer in the fully extended VGG model.
        """
        if not hasattr(old_layer, 'weight'):
            return
        new_layer.weight.copy_(old_layer.weight)
        if old_layer.bias is not None:
            new_layer.bias.copy_(old_layer.bias)

    def find_insertion_points(self):
        """
        Find the insertion points in the VGG model.

        Returns:
            The insertion points.
        """
        res = []
        for k, child in enumerate(self.model.features):
            if isinstance(child, torch.nn.ReLU):
                res.append(k+1)
        return res

    @staticmethod
    @torch.no_grad()
    def _to_id(layer):
        """
        Set the weights and biases of a layer to identity.

        Args:
            layer: The layer to set the weights and biases to identity.
        """
        if not isinstance(layer, torch.nn.Conv2d):
            return

        channels = layer.out_channels
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, Tuple):
            kernel_size = kernel_size[0]

        k_index = kernel_size // 2
        layer.weight *= 0
        layer.weight[:, :, k_index, k_index].copy_(torch.eye(channels, channels))

        layer.bias *= 0




class SmallVGGExtendableModel(ExtendableModel):
    """
    Extendable VGG model.
    """

    def __init__(self, m=[2,4,8], BN=False, use_kfac = False) -> None:
        """
        Initialize the VGGExtendableModel.

        Args:
            BN (bool, optional): Whether to use Batch Normalization. Defaults to False.
        """
        super().__init__()
        self.m = m
        self.model = nets.build_vgg_baseline(BN, small=True, m=m)
        self.BN = BN
        self.use_kfac = use_kfac

    def _build_fullyext(self):
        """
        Build the fully extended VGG model.

        Returns:
            The fully extended VGG model.
        """
        return nets.get_fullext_model_small(self.m)
    
    def _get_layer_iterator(self):
        """
        Get an iterator over the layers of the original VGG model.
        """
        return chain(chain(self.model.features.children(),torch.nn.Sequential(torch.nn.Flatten())), self.model.classifier.children())

    def _get_new_layer_iterator(self):
        """
        Get an iterator over the layers of the fully extended VGG model.
        """
        return self.fully_ext_model.children()

    def _layer_is_relevant(self, layer):
        """
        Check if a layer is relevant for selection.

        Args:
            layer: The layer to check.

        Returns:
            True if the layer is relevant, False otherwise.
        """
        return isinstance(layer, torch.nn.Conv2d)

    def _layer_metric(self, layer, sens_norm):
        """
        Calculate the metric for a layer.

        Args:
            layer: The layer to calculate the metric for.
            sens_norm: The sensitivity norm to use. Includes ['all', 'fro_squared', 'fro_squared_scaled','op22_1','op22_2','op22'].
        Returns:
            The metric value for the layer.
        """
        if sens_norm=='all':
            shape = layer.sensitivities_w.shape
            no_out_channels = shape[0]
            #no_in_channels = shape[1]
            if no_out_channels==2: input_shape = (28,28)
            if no_out_channels==4: input_shape = (14,14)
            if no_out_channels==8: input_shape = (7,7)

            norm_fro=layer.sensitivity_w

            norm_fro_scaled=layer.sensitivity_w / utils.prod(layer.weight.shape)

            norm_op22_1=0 
            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm_op22_1 += s

            norm_op22_2= 0 
            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm_op22_2 += s*s
            norm_op22_2 = np.sqrt(norm_op22_2)

            norm_op22 = utils.max_sing_value(layer.sensitivities_w, input_shape)

            norm = norm_op22
            norms = [norm_fro.item(), norm_fro_scaled.item(), norm_op22_1, norm_op22_2, norm_op22]
            # make norms attribute of layer
            setattr(layer, 'sens_norms_all', norms)

        if sens_norm=='fro_squared':
            norm = layer.sensitivity_w
        if sens_norm=='fro_squared_scaled':
            norm = layer.sensitivity_w / utils.prod(layer.weight.shape)
        if sens_norm=='op22_1':
            norm = 0 
            shape = layer.sensitivities_w.shape
            no_out_channels = shape[0]
            #no_in_channels = shape[1]
            if no_out_channels==2: input_shape = (28,28)
            if no_out_channels==4: input_shape = (14,14)
            if no_out_channels==8: input_shape = (7,7)

            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm += s

        if sens_norm=='op22_2':
            norm = 0 
            shape = layer.sensitivities_w.shape
            no_out_channels = shape[0]
            #no_in_channels = shape[1]
            if no_out_channels==2: input_shape = (28,28)
            if no_out_channels==4: input_shape = (14,14)
            if no_out_channels==8: input_shape = (7,7)

            for i in range(no_out_channels):
                s = utils.max_sing_value(layer.sensitivities_w[i,:,:,:], input_shape)
                norm += s*s
            norm = np.sqrt(norm)
            
        if sens_norm=='op22':
            shape = layer.sensitivities_w.shape
            no_out_channels = shape[0]
            #no_in_channels = shape[1]
            if no_out_channels==2: input_shape = (28,28)
            if no_out_channels==4: input_shape = (14,14)
            if no_out_channels==8: input_shape = (7,7)

            norm = utils.max_sing_value(layer.sensitivities_w, input_shape)

        return norm

    def _handle_new_layer_list(self, new_layer_list):
        """
        Handle the new layer list after selecting a layer.

        Args:
            new_layer_list: The new layer list.
        """
        layer_feat = new_layer_list[:11]
        # 11 is flatten which is not needed in baseline
        layer_class = new_layer_list[12:]

        self.model.features = torch.nn.Sequential(*layer_feat)
        self.model.classifier = torch.nn.Sequential(*layer_class)

        

    def _layer_predecessor(self, layer):
        """
        Get the predecessor layers of a given layer.

        Args:
            layer: The layer to get the predecessor layers for.

        Returns:
            The predecessor layers.
        """
        return []

    def _layer_successor(self, layer):
        """
        Get the successor layers of a given layer.

        Args:
            layer: The layer to get the successor layers for.

        Returns:
            The successor layers.
        """
        res = []
        if self.BN:
            res.append(torch.nn.BatchNorm2d(layer.out_channels))
        res.append(torch.nn.ReLU())
        return res

    @torch.no_grad()
    def _copy_layer(self, old_layer, new_layer):
        """
        Copy the parameters of a layer from the original VGG model to the fully extended VGG model.

        Args:
            old_layer: The layer from the original VGG model.
            new_layer: The layer in the fully extended VGG model.
        """
        if not hasattr(old_layer, 'weight'):
            return
        new_layer.weight.copy_(old_layer.weight)
        if old_layer.bias is not None:
            new_layer.bias.copy_(old_layer.bias)

    def find_insertion_points(self):
        """
        Find the insertion points in the VGG model.

        Returns:
            The insertion points.
        """
        res = []
        for k, child in enumerate(self.model.features):
            if isinstance(child, torch.nn.ReLU):
                res.append(k+1)
        return res

    @staticmethod
    @torch.no_grad()
    def _to_id(layer):
        """
        Set the weights and biases of a layer to identity.

        Args:
            layer: The layer to set the weights and biases to identity.
        """
        if not isinstance(layer, torch.nn.Conv2d):
            return

        channels = layer.out_channels
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, Tuple):
            kernel_size = kernel_size[0]

        k_index = kernel_size // 2
        layer.weight *= 0
        layer.weight[:, :, k_index, k_index].copy_(torch.eye(channels, channels))

        layer.bias *= 0
