# Renate's output

The result of a Renate update job is written to the folder specified via the `next_state_url` attribute. This folder
contains the *Renate state*. It will be required for the next update job.

The Renate state folder contains three files:

```{list-table}
:header-rows: 1

* - File
  - Description
* - model.ckpt
  - This is the checkpoint of the trained model and the only file required for deployment.
    Use and [load this file](#loading-the-updated-model) to make predictions.
* - learner.ckpt
  - This contains the state of the Renate updater. Only used by Renate.
* - hpo.csv
  - A summary of all previous updates. The `update_id` with highest value refers to the last update step.
    Among other things, it contains information about selected hyperparameters and logged metrics.
    It might be used in the next update step to accelerate the hyperparameter tuning step.
```

## Loading the Updated Model

In the following, we refer with `model_fn` to the function defined by the user in the [Renate config file](./how_to_renate_config.rst).

### Output Saved Locally

If `next_state_url` is a path to a local folder, loading the updated model can be done as follows:

```python
from renate.defaults import current_state_folder, model_file

my_model = model_fn(model_file(current_state_folder(next_state_url)))
```

### Output Saved on S3

If the Renate output was saved on S3, the model checkpoint `model.ckpt` can be downloaded from

```python
from renate.defaults import current_state_folder, model_file

print(model_file(current_state_folder(next_state_url)))
```

and then loaded via

```python
my_model = model_fn("model.ckpt")
```