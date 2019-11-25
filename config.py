# coding: UTF-8
import warnings
import torch

class DefaultConfig(object):



    model = 'PyGraphsage'
    use_gpu = False
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    load_model_path = None
    # 以上模型基本信息

    network = 'cora'
    batch_size = 8
    num_workers = 3
    max_epoch = 10
    lr = 0.005
    lr_decay = 0.9
    weight_decay = 1e-5
    train_rate = 0.05
    val_rate = 0.05
    droput = 0.5


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()