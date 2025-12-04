from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import transformers, torch

# load training data
print('======================================loading training data.')
dataset = load_dataset("json", data_files="feed/health_advice_train.jsonl")
print('======================================loaded training data.')

# load tokenizer from pretrained checkpoint automatically using huggingface.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# apply tokenizer.
def tokenize_data(data):
    prompt = f"Instruction: {data['instruction']}\nInput: {data['input']}\nResponse:"

    text = prompt + " " + data["output"]

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = dataset.map(tokenize_data, batched=False)

# instantiate the lm from pretrained checkpoint for QLORA finetuning.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

# setup LORA config. experimental.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model ready for selective fine tuning.
print('======================================setting up PEFT model')
model = get_peft_model(model, lora_config)
print('======================================done setting up PEFT model')

# configure training args. experimental.
training_args = TrainingArguments(
    output_dir="./finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    max_steps=10,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

print('======================================intiating finetuning')
trainer.train()
print('======================================finetuning complete')
