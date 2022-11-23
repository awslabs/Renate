# How to Run a Training Job

Renate offers possibility to run training jobs using both CLIs and functions that can be called
programmatically in python. The best choice may be different depending on the requirements (e.g.,
a CLI can be convenient to run remote jobs). In the following we illustrate the solution that we
find to be the simplest and more convenient, but more documentation is available [here](TODO).

## Running a training job

The first that needs to be completed before running a training job is to define which model needs
to be trained and on which data. This can be achieved in Renate by creating the `renate_config.py` file,
the details about how to do it are provided [here](./how_to_renate_config.md).

Once completed the first step, a simple way to run a training job is to use the `execute_tuning_job`, this can work for most training
needs: it can launch trainings with and without HPO, either locally or on Amazon SageMaker.

The function is designed to launch training jobs that include HPO but when a single configuration
is provided it just executes a normal training. In fact, the functions requires a `config_space`
argument, a dictionary containing the search space for the HPO process.
When all the keys in the dictionary are associated to a single value (e.g., use `0.5` instead of `uniform(0,1)`),
the search space contains a single configuration
and no HPO is performed. The model is just trained with the available configuration.
When the search space contains more than a single configuration (e.g., ranges like `choice(["SGD","Adam"])`),
then Renate will perform HPO. 
When running HPO, it is important to specify the `mode` with either `min` or `max` so that the
optimizer will know if the metric will need to be minimized or maximized. It is also important
to specify which metric to optimize: metrics measured on the validation set are prefixed with
`val_`, while metrics measured on the training set are prefixed with `train_`. When optimizing
a validation metric is also important to make sure that that `data_module_fn` provides a reasonable
fraction of the data as validation set. It also possible to define more aspects of the HPO process,
for example it is possible to specify 
the number of workers evaluating configurations in parallel using `n_workers` (useful for multi-cpu
or multi-gpu machines), and to decide which algorithm to use for the HPO by specifying
it with the `scheduler` argument and the stopping criterion of the HPO (a few are [available](TODO)).
Disregarding the presence of default values, we recommend to specify the termination criteria of the HPO, 
for example by setting `max_time` to specify a maximum duration.
If you want to run HPO but do not have a clear understanding of which search space to use,
it is possible to use a default search space for the selected algorithm using the
`config_space` function as in the example below.

The choice of the backend on which to execute the training is also specified as an argument. 
When `backend="local"` is used, then the training job will be executed on the local machine.
When `backend="sagemaker"` is used, the training job will be executed on Amazon SageMaker. 
Additional arguments are not required but it is highly recommended to specify at least the
`instance_type` (the kind of machine to be used) selecting one from  [this list](https://aws.amazon.com/sagemaker/pricing/).
Optional arguments such the `job_name` can help identifying the training job in the AWS control
panel when a large number of training jobs is launched.

There are two more arguments that play an important role:
* `updater`: this argument selects the algorithm to be used for updating the model.
* `max_epochs`: the maximum number of training epochs
* `input_state_url`: this is the location at which the state of learner and the model are made available. 
This contains the starting point for our training job, the model the needs to be updated.
* `output_state_url`: this is the location at which the output of the training job (model, state)
will be stored.

In both cases these urls can point to local folders or S3 locations.

Putting everything together should give file similar to the following one.

```python
from renate.tuning import execute_tuning_job
from renate.tuning.config_spaces import config_space

if __name__ == "__main__":

    # we run the first training job on the MNIST classes [0-4]
    execute_tuning_job(
        config_space=config_space("RD"),  # getting the default search space
        mode="max",
        metric="val_accuracy",
        updater="RD",  # use the RepeatedDistillationModelUpdater
        max_epochs=50,
        config_file="renate_config.py",
        output_state_url="./output_folder/",  # this is where the model will be stored
        backend="local",  # the training job will run on the local machine
        scheduler="asha",
        max_num_trials_finished=50,
    )
```

A working version of this example is available in `/examples/train_mlp_locally/start_with_hpo.py`.
Another example using this function for local training is available [here](../examples/train_mlp_locally.rst),
while an example with SageMaker training is available [here](../examples/train_classifier_sagemaker.rst).

The complete documentation of the `execute_tuning_job` function is available [here](TODO).

### Running experiments

If you want to run a large number of training jobs, for example for experimentation, it will
be useful to take a look at the [tutorial on experimenting with Renate](TODO).