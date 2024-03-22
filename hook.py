import paddle
import os
import numpy as np

first_layer_name_of_each_micro_step = None
micro_step = -1
tensor_idx = 0
tensor_limit = 50
should_save = True


def _register_hook_impl(path, target_class_names):
    def __impl__(layer, i, o):
        global first_layer_name_of_each_micro_step, micro_step, tensor_idx, tensor_limit, should_save
        if layer.full_name() == first_layer_name_of_each_micro_step:
            micro_step += 1
        if o.dtype == paddle.bfloat16:
            o = paddle.cast(o.detach(), paddle.float32)
        else:
            o = o.detach()
        if should_save and layer.__class__.__name__ in target_class_names:
            np.save(
                f'./{path}/rank_{paddle.distributed.get_rank()}/'
                f'tensor_idx_{tensor_idx}_layer_class_{layer.__class__.__name__}_'
                f'layer_name_{layer.full_name()}_micro_step_{micro_step}.npy',
                o.numpy())
            tensor_idx += 1
            should_save = (tensor_idx != tensor_limit)

    return __impl__


def register_hook(model, path='./tmp_tensors', target_class_names=None, max_saved_tensors=50):
    """
    Register post forward hook to layers indicated by `target_class_names`. All temp tensors will be saved in `path`.
    :param model: The model to be registered with hook.
    :param path: The path to save the temp tensors.
    :param target_class_names: A str or a list of str, indicate the name of target class to be saved.
                               Note that, only the name of the class should be passed, not the class itself.
                               Warning: the more target classes are given, the more disk capacities will be consumed.
    :param max_saved_tensors: The maximum number of tensors will be saved.
                              The default value is 50. Set to any negative number to save all tensors.
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

    global first_layer_name_of_each_micro_step, tensor_limit
    tensor_limit = max_saved_tensors

    os.makedirs(path, exist_ok=True)
    os.makedirs(f'./{path}/rank_{paddle.distributed.get_rank()}', exist_ok=True)
    for layer in model.sublayers():
        if len(layer.sublayers()) == 0:
            if first_layer_name_of_each_micro_step is None:
                first_layer_name_of_each_micro_step = layer.full_name()
            layer.register_forward_post_hook(_register_hook_impl(path, target_class_names))
    return model
