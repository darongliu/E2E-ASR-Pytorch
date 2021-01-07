import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableAugment(nn.Module):
    def __init__(self, T_num_masks=1, F_num_masks=1, noise_dim=None, dim=[10, 10, 10], norm=[True, True, True], replace_with_zero=False):
        super(DifferentiableAugmentation, self).__init__()
        self.T_num_masks = T_num_masks
        self.F_num_masks = F_num_masks

        self.dim = dim
        self.norm = norm
        self.replace_with_zero = replace_with_zero
        self.training_aug = True

        self.output_dim = (self.T_num_masks+self.F_num_masks)*2 # position and width
        if noise_dim is None:
            self.noise_dim = self.output_dim
        else:
            self.noise_dim = noise_dim

        assert (len(dim) == len(norm))
        module_list = []
        prev_dim = self.noise_dim
        for d, n in zip(self.dim, self.norm):
            module_list.append(nn.Linear(prev_dim, d))
            prev_dim = d
            module_list.append(nn.Relu())
            if n:
                module_list.append(nn.BatchNorm1d(d))
        module_list.append(nn.Linear(prev_dim, self.output_dim))
        module_list.append(nn.Sigmoid())

        # init last linear 
        # last layer output [T_mask_center, T_mask_width, F_mask_center, F_mask_width] 
        module_list[-2].bias[self.T_num_masks:2*self.T_num_masks] = -3.
        module_list[-2].bias[2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks] = -3.
        self.layers = nn.Sequential(*module_list)
        
    def forward(self, spec):
        filling_value = 0. if self.replace_with_zero else spec.mean()

        _, F, T = spec.shape

        # generate aug parameters
        noise = torch.randn(self.noise_dim)
        aug_param = self.layers(noise)

        # mask T
        T_log_mask_weight = self._get_soft_weight(T, aug_param[:self.T_num_masks], aug_param[self.T_num_masks:2*self.T_num_masks])
        # mask F
        F_log_mask_weight = self._get_soft_weight(F, aug_param[2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], aug_param[2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks])

        total_log_mask_weight = T_log_mask_weight.unsqueeze(0)+F_log_mask_weight.unsqueeze(1)
        total_mask_weight = torch.exp(total_log_mask_weight).unsqueeze(0)

        new_spec = spec*(1-total_mask_weight) + filling_value*total_mask_weight
        return new_spec

    def _get_soft_weight(self, length, mask_center, mask_width):
        '''
        length: int
        mask_center: [num_mask]
        mask_width: [num_mask]
        output: [length]
        '''
        position = torch.range(start=0, end=length-1)
        dist_to_center = mask_center.unsqueeze(-1) - position.unsqueeze(0) # [num_mask, length]
        log_mask_weight = torch.LogSigmoid((-torch.abs(dist_to_center)/mask_width.unsqueeze(-1)*10)+5) # [num_mask, length]
        log_mask_weight = log_mask_weight.sum(0) # length
        return log_mask_weight

    def train_aug(self):
        self.training_aug = True

    def not_train_aug(self):
        self.training_aug = False
'''
class TrainableAugment(object):
    def __init__(self, model, train):
        self.aug = _TrainableAugmentModule(**model)
        self.optimizer = torch.optim.Adam(self.aug.parameters(), betas=(0.5, 0.999), **train)

    def forward(self, spec):
        return self.aug(spec)

    def train_aug(self, input_train, target_train, input_valid, target_valid, eta):
        # TODO only assume SGD as model optimizer, try to design the loss from model optimizer 
        self.optimizer.zero_grad()
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        self.optimizer.step()

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
        if v.grad is None:
            v.grad = Variable(g.data)
        else:
            v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
'''
class SpecAugment(nn.Module):
    def __init__(self, T=40, num_masks=1, replace_with_zero=False, F=27):#ori: T = 40
        super(Augment, self).__init__()    
        self.T=T
        self.num_masks=num_masks
        self.replace_with_zero=replace_with_zero
        self.F=F
        self.spec=None
    #@torch.jit.script_method
    def forward(self, spec):
        spec = spec.permute(1, 0)

        spec = self.time_mask(spec, T=self.T, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = self.freq_mask(spec, F=self.F, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = spec.permute(1, 0)
        
        return spec
    def normalize(self, spec):
        spec = (spec-spec.mean())/spec.std()
        return spec        

    def time_mask(self, spec, T=100, num_masks=1, replace_with_zero=False):
        cloned = spec
        len_spectro = cloned.shape[1]
        
        for i in range(0, num_masks):
            t = torch.randint(0, self.T, (1,)).item()
            t_zero = torch.randint(0, len_spectro-t, (1,)).item()
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = torch.randint(t_zero, t_zero+t, (1, )).item()

            if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned

    def freq_mask(self, spec, F=27, num_masks=1, replace_with_zero=False):
        cloned = spec
        num_mel_channels = cloned.shape[0]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if (replace_with_zero): cloned[f_zero:mask_end, :] = 0
            else: cloned[f_zero:mask_end, :] = cloned.mean()

        return cloned



'''
class DifferentiableAugmentation(nn.Module):
    def __init__(self, T_num_masks=1, F_num_masks=1, T_position_peak_num=3, F_position_peak_num=3, replace_with_zero=False):
        super(DifferentiableAugmentation, self).__init__()
        self.T_num_masks = T_num_masks
        self.F_num_masks = F_num_masks
        self.T_position_peak_num = T_position_peak_num
        self.F_position_peak_num = F_position_peak_num
        self.replace_with_zero = replace_with_zero

        self.T_width = nn.parameter.Parameter(torch.tensor([-3.]*self.T_num_masks), requires_grad=True)
        self.F_width = nn.parameter.Parameter(torch.tensor([-3.]*self.F_num_masks), requires_grad=True)

        self.T_position_mean = nn.parameter.Parameter(torch.rand(self.T_num_masks, self.T_position_peak_num)-0.5, requires_grad=True) # pass through sigmoid to get real position mean # init being at center
        self.F_position_mean = nn.parameter.Parameter(torch.rand(self.F_num_masks, self.F_position_peak_num)-0.5, requires_grad=True)

        self.T_position_weight = nn.parameter.Parameter(torch.rand(self.T_num_masks, self.T_position_peak_num), requires_grad=True) # pass through softmax to get real position weight
        self.F_position_weight = nn.parameter.Parameter(torch.rand(self.F_num_masks, self.F_position_peak_num), requires_grad=True)

    def forward(self, spec):
        _, F, T = spec.shape
        total_mask = torch.ones(F, T)

        # mask T
        for i in range(self.T_num_masks):

        # mask F
        for i in range(self.)
        # mask T

    def _get_soft_position(self, length, position_mean, position_weight):
        # position mean: [num_mask, peak_num]
        # position weight: [num_mask, peak_num]
        # output: [num_mask, length]
        position_mean = torch.sigmoid(position_mean)
        position_weight = torch.softmax(position_weight, -1)

        position = torch.range(start=0, end=length-1)

        dist_to_mean = position_mean.unsqueeze(-1) - position.unsqueeze(0).unsqueeze(0) # [num_mask, peak_num, length]

        log_prob = -(dist_to_mean**2)
'''