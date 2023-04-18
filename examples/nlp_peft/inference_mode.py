import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


def main():
    model_name = "EleutherAI/gpt-j-6B"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    
    print("Making model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto',
        load_in_8bit=True,
        max_memory={i: "3GB" for i in range(n_gpus)},
         output_hidden_states=False, output_attentions=False, use_cache=False, pad_token_id=0, cache_dir="/home/ec2-user/SageMaker/hub").eval()
    print("Instantiated the models!\n\n")

    text = 'Hamburg is in which country?\n'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.expand(100, -1)

    import pdb
    pdb.set_trace()
    
    with torch.inference_mode():
        print(model(input_ids))


class PlModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        print(batch)
        return self(batch.input_ids)


def main2():
    model_name = "EleutherAI/gpt-j-6B"
    model = make_transformers_model(model_name, enable_gradient_checkpointing=False, load_in_8bit=True, output_hidden_states=False, output_attentions=False, use_cache=False, pad_token_id=0, cache_dir="/home/ec2-user/SageMaker/hub")
    # model = RenateWrapper(model, loss_fn=lambda x: x.sum())
    # model = torch.nn.Linear(10, 2)
    model = PlModel(model)
    dataset = data_module_fn(chunk_id=1, data_path="../../renate_working_dir")
    dataset.setup()

    trainer = Trainer()
    print(trainer.predict(model, dataset._test_data))


if __name__ == "__main__":
    main()