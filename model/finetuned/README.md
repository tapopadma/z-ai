---
base_model: tapopadma/myai
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
- llama-cpp
- gguf-my-lora
---

# tapopadma/myai-F16-GGUF
This LoRA adapter was converted to GGUF format from [`tapopadma/myai`](https://huggingface.co/tapopadma/myai) via the ggml.ai's [GGUF-my-lora](https://huggingface.co/spaces/ggml-org/gguf-my-lora) space.
Refer to the [original adapter repository](https://huggingface.co/tapopadma/myai) for more details.

## Use with llama.cpp

```bash
# with cli
llama-cli -m base_model.gguf --lora myai-f16.gguf (...other args)

# with server
llama-server -m base_model.gguf --lora myai-f16.gguf (...other args)
```

To know more about LoRA usage with llama.cpp server, refer to the [llama.cpp server documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).
