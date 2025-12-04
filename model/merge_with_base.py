from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("./TinyLlama/TinyLlama-1.1B-Chat-v1.0")
lora = PeftModel.from_pretrained(base, "./finetuned/checkpoint-10")
print('======================================initiating merge.')
merged = lora.merge_and_unload()
print('====================================== merge complete.')
merged.save_pretrained("./merged-model")
print('====================================== saving merged model.')
