import six
from chainermn.optimizers import _MultiNodeOptimizer

def patch_mn_optimizer(optimizer):
    def reset_prev_params(self):
        super(_MultiNodeOptimizer, self).__setattr__(
            'target_params', [])
    optimizer.reset_prev_params = six.create_bound_method(reset_prev_params,
                                                          optimizer)
    
