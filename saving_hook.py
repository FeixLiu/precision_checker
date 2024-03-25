import paddle
import os
import numpy as np

first_layer_name_of_each_micro_step = None
micro_step = -1
tensor_idx = 0
grad_idx = 0
tensor_limit = 50
grad_limit = 100
should_save_tensor = True
should_save_grad = True


def _register_backward_hook_impl(path, param):
    def __impl__(grad):
        global micro_step, grad_idx, grad_limit, should_save_grad
        if should_save_grad:
            if grad.dtype == paddle.bfloat16:
                tgt = paddle.cast(grad.detach(), paddle.float32)
            else:
                tgt = grad.detach()
            np.save(
                f'./{path}/grads/rank_{paddle.distributed.get_rank()}/'
                f'grad_idx_{grad_idx}_of_param_name_{param.name}_'
                f'micro_step_{micro_step}.npy',
                tgt.numpy())
            grad_idx += 1
            should_save_grad = (grad_idx != grad_limit)

    return __impl__


def _register_forward_post_hook_impl(path, target_class_names):
    # have to pass target_class_names to this function to increase micro_step
    def __impl__(layer, i, o):
        global first_layer_name_of_each_micro_step, micro_step, tensor_idx, tensor_limit, should_save_tensor
        if layer.full_name() == first_layer_name_of_each_micro_step:
            micro_step += 1
        if layer.__class__.__name__ in target_class_names and should_save_tensor:
            if o.dtype == paddle.bfloat16:
                o = paddle.cast(o.detach(), paddle.float32)
            else:
                o = o.detach()
            np.save(
                f'./{path}/tensors/rank_{paddle.distributed.get_rank()}/'
                f'tensor_idx_{tensor_idx}_layer_class_{layer.__class__.__name__}_'
                f'layer_name_{layer.full_name()}_micro_step_{micro_step}.npy',
                o.numpy())
            tensor_idx += 1
            should_save_tensor = (tensor_idx != tensor_limit)

    return __impl__


def register_saving_hook(model, path='./tmp_tensors', target_class_names=None, max_saved_tensors=50, max_saved_grads=100):
    """
    Register post forward hook to layers indicated by `target_class_names`.
    All temp tensors of target layers will be saved in `path/tensor`.
    All grads of parameters in target layers will be saved in `path/grad`.
    :param model: The model to be registered with hook.
    :param path: The path to save the temp tensors.
    :param target_class_names: A str or a list of str, indicate the name of target layer to be saved.
                               Note that, only the name of the layer should be passed, not the layer itself.
                               Warning: the more target layers are given, the more disk capacities will be consumed.
    :param max_saved_tensors: The maximum number of tensors will be saved.
                              The default value is 50. Set to any negative number to save all tensors.
    :param max_saved_grads: The maximum number of grads will be saved.
                            The default value is 100. Set to any negative number to save all tensors.
                            In general, more grads than tensors should be saved since one layer may contain
                            more than one parameter.
    :return: The model after registering hook.
    """
    assert isinstance(model, paddle.nn.Layer), "[HOOK] Teh model must be of type paddle.nn.Layer."
    assert isinstance(path, str), "[HOOK] The path must be of type str."
    if target_class_names is None:
        print('[HOOK] No target class to save, skip register hook.')
        return
    assert isinstance(target_class_names, (str, list)), "[HOOK] The target_class_names must be str or list of str."
    if isinstance(target_class_names, str):
        target_class_names = [target_class_names]
    if os.path.exists(path):
        print(f'[HOOK] The path {path} is already exists, will reuse the path to save temp tensors.')

    print(f'[HOOK] Registering hook for layers: {target_class_names}.')
    if max_saved_tensors > 0:
        print(f'[HOOK] Up to {max_saved_tensors} tensors will be save.')
    else:
        print(f'[HOOK] All tensors will be saved.')

    global first_layer_name_of_each_micro_step, tensor_limit, grad_limit
    tensor_limit = max_saved_tensors
    grad_limit = max_saved_grads

    os.makedirs(path, exist_ok=True)
    os.makedirs(f'{path}/tensors', exist_ok=True)
    os.makedirs(f'{path}/grads', exist_ok=True)
    os.makedirs(f'./{path}/tensors/rank_{paddle.distributed.get_rank()}', exist_ok=True)
    os.makedirs(f'./{path}/grads/rank_{paddle.distributed.get_rank()}', exist_ok=True)
    for layer in model.sublayers():
        if len(layer.sublayers()) == 0:
            if first_layer_name_of_each_micro_step is None:
                first_layer_name_of_each_micro_step = layer.full_name()
            layer.register_forward_post_hook(_register_forward_post_hook_impl(path, target_class_names))
            if layer.__class__.__name__ in target_class_names:
                for param in layer.parameters():
                    param._register_grad_hook(_register_backward_hook_impl(path, param))
    return model
