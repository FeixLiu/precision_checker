# What is this repo for?
This repo is a tool to help saving some activation tensors for some specific Layers during training. And this repo 
provides a checker to calc some statistical data between two sets of activation tensors.

# How to use
## Step 1: Register hook
The `hook.py` script provides an interface `register_hook` to warp your model by registering hooks 
to the Layers specified by the user. Here is the signature of this interface:

```
register_hook(model, path='./tmp_tensors', target_class_names=None, max_saved_tensors=50)
```
- `model`: The model to be registered with hook.
- `path`: The path to save the temp tensors.
- `target_class_names`: A str or a list of str, indicate the name of target class to be saved. 
  Note that, only the name of the class should be passed, not the class itself.
  Warning: the more target classes are given, the more disk capacities will be consumed.
- `max_saved_tensors`: The maximum number of tensors will be saved.
  The default value is 50. Set to any negative number to save all tensors.

Users should warp their model before the start of training.

```python
from hook import register_hook
from paddlenlp import LLama13B, Trainer

model = LLama13B()
model = register_hook(
    model, 
    path='./tmp_tensors', 
    target_class_names=['ColumnSequenceParallelLinear', 'RowSequenceParallelLinear', 'LayerNorm'],
    max_saved_tensors=100
)
trainer = Trainer(model)
trainer.train()
```

For distributed training, tensors will be saved in different folders named by the rank name.

After training, the folder will have a structure like this:
```
├── tmp_tensors
│   ├── rank_0
│   │   ├── tensor_idx_XX_layer_class_XX_layer_name_XX_micro_step_XX.npy
│   │   ├── tensor_idx_XX_layer_class_XX_layer_name_XX_micro_step_XX.npy
│   │   ├── tensor_idx_XX_layer_class_XX_layer_name_XX_micro_step_XX.npy
│   │   ├── ...
│   ├── rank_1
│   ├── rank_2
│   ├── rank_3
│   ├── ...
```

## Step 2: Parse the result
After saving two sets of activation tensors, lets say these tenors are saved in folder `tmp_tensors` and 
folder `tmp_tensors_0` separately, user can use the `checker.py` to parse the tensors to get some statistical data.
Users can use this command to parse the tensors:

```bash
python checker.py --pathA tmp_tensors --pathB tmp_tensors_0
```

With this command, progress bars will be shown.

Some rules the checker is following:
1. The checker only check the intersection part of these two paths.
2. The result will be saved in folder `pathA/compare_rst/`.
3. Each rank will have its own result csv file such as `rank_0_rst.csv`.
4. `cosine similarity`, `mean`, `standard deviation`, `variance` will be calculated.

## Step 3: Analyse the result
The rst.csv files contains 11 columns:
- `tensor_name`: the name of the tensor.
- `cosine similarity`: the cosine similarity between two tensors.
- `meanA`: the mean value of tensor under pathA.
- `meanB`: the mean value of tensor under pathB.
- `meanA - meanB`: the difference of the mean value between two tensors.
- `stdA`: the standard deviation value of tensor under pathA.
- `stdB`: the standard deviation value of tensor under pathA.
- `stdA - stdB`: the difference of the standard deviation value between two tensors.
- `varA`: the variance value of tensor under pathA.
- `varB`: the variance value of tensor under pathA.
- `varA - varB`: the difference of the variance value between two tensors.