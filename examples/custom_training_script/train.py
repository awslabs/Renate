# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import torch
import transformers
from torch.optim import Adam

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.models.renate_module import RenateWrapper
from renate.updaters.experimental.fine_tuning import FineTuningModelUpdater

# Create model.
transformer_model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, return_dict=False
)
model = RenateWrapper(transformer_model)
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

# Prepare data.
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
data_module = HuggingFaceTextDataModule(
    "data", dataset_name="rotten_tomatoes", tokenizer=tokenizer, val_size=0.2
)
data_module.prepare_data()  # For multi-GPU, call only on rank 0.
data_module.setup()

# Instantiate renate ModelUpdater and run fine-tuning.
optimizer = partial(Adam, learning_rate=3e-4)
updater = FineTuningModelUpdater(
    model,
    loss_fn,
    optimizer=optimizer,
    batch_size=32,
    max_epochs=3,
    input_state_folder=None,
    output_state_folder="renate_output",
)
updater.update(data_module.train_data(), data_module.val_data())


# Do something with model, e.g., save its weights for later use.
torch.save(model.state_dict(), "model_weights.pt")
