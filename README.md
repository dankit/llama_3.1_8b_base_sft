# LLM Finetuning & Evaluation

Objective: supervised finetuning on the base llama 3.1 8b model, specifically using the alpaca instruction dataset to see how much of an improvement we can get on instruction following. I am not so much trying to make it conversational/multi-turn as much as simple instruction following.

llama 3.1 8b base vs llama 3.1 8b instruct in the IFEval and tinyMMLU.
MMLU (16bf):
-Base scored 64.2% on 0-shot tinyMMLU
-Instruct scored 62.9% on 0-shot tinyMMLU
-base model had a slightly higher tinyMMLU compared to instruct variant, this may be because it's testing general knowledge, and base has the highest "raw knowledge"
  -0-shot = knowledge, 5-shot = in-context ability?
-calculated score using log prob scoring (logits for a/b/c/d and highest probability)

IFEval (16bf, 8bit, 4bit):
-tested quantizing, and specifically used unsloth's dynamic quantization for 4bit, 8bit, observing minimal loss in IFEval accuracy.
-observed 8bit inference was much slower than bf16 and 4bit with unsloth, I then compared this to normal huggingface + bitsandbytes and had similar outcomes.
  -this is my second time observing this, the both times using Ampere. I'm guessing newer cards use FP8 so no one cares about int8 and its not optimized?
  -I might have been suboptimal with my batch sizes on some runs, but unsloth was definitely faster than the base huggingface lib.
-Had issues with unsloth loader for bf16 llama 3.1 8b instruct evals which had worse performance than the quantized 8bit version, moved to huggingface loader which fixed it.


random notes:
-Was using vllm early on but consolidated everything into unsloth for simplicity for this small scale project
-Had to take evals out from eval generation loop, as the machine i was training on was CPU bottlenecked while doing so.
  -moved eval scoring (cpu bound work), off gpu machine and onto personal computer to maximize throughput
-SFT'd models such as chat/instruct models have chat templates typically.
  -"add_generation_prompt" in the chat template adds the "turn" token so that model/assistant knows its their turn
-usually use left side padding for inference (all tokens to the right are "true" tokens), even if its typically right side during training
-pin memory: it page locks memory so that ram cant be swapped out to disk making it a stable physical address to access for high speed data transfers between cpu and gpu (direct memory access)


Training:
The goal is to experiment with LoRA, in which I calculated it would take me atleast 100GB in an unoptimized training setup to run training for 1 example with seq len of 2048.
Most of this memory comes from the optimizer states when using Adamw which takes up ~60gb, and gradients which take up ~15gb. With Lora, the same training example should comfortably
fit on 1xA100 (40gb) as base models have requires_grad=False meaning no stored gradients/optimizers. LoRA adapter is added ontop of the frozen base weights. W_new = W_frozen + W_lora = W_frozen + BA, where A and B are low rank matrices.

backpropagation still happens, as we still need to compute the error. During backward, it calculates the error per layer depending on their activations. It's only if the parameter is trainable, that this calculated error gets stored, in which we see gradient memory take up space. Activations are also stored before backward pass which is where the other bottleneck starts to show itself. Activations are released once the layers gradient is computed (i think)

Training dataset:
Utilizing alpaca-cleaned containng 52,000 instructions, found here https://huggingface.co/datasets/yahma/alpaca-cleaned (credits to original authors)


## Setup

```bash
sudo snap install astral-uv --classic
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
git clone https://github.com/felipemaiapolo/tinyBenchmarks.git
uv pip install -e tinyBenchmarks
wandb login
```

## Training

```bash
# Train (automatically logs to W&B)
python train.py
```

## Evaluation (Two-Step)

### Step 1: Generate responses (GPU)

```bash
# Base model (no chat template)
python eval_generate.py --model unsloth/Llama-3.1-8B --precision 4bit --no-chat-template

# Instruct model
python eval_generate.py --model unsloth/Llama-3.1-8B-Instruct --precision 4bit

#use huggingface instead of loading with unsloth
python eval_generate.py --model meta-llama/Llama-3.1-8B-Instruct --precision bf16 --batch-size 8 --limit 50 --use-hf

# Finetuned model (with LoRA)
python eval_generate.py --model unsloth/Llama-3.1-8B --lora ./lora_model_4bit --precision 4bit

# Quick test (10 samples)
python eval_generate.py --model unsloth/Llama-3.1-8B --precision 4bit --limit 10
```

### Step 2: Score responses (CPU)

```bash
powershell -ExecutionPolicy Bypass -File setup_ifeval.ps1
python eval_score.py --input ./eval_responses/YOUR_FILE.json
```

### View results in Weights & Biases

```bash
# First time: login to W&B (free account)
wandb login

# Log results to W&B dashboard
python eval_wandb.py

# After finetuning, create comparison report
python eval_wandb.py --compare
```

## Inference

```bash
python inference.py
python inference.py --lora ./lora_model_4bit
```



## Misc commands

Eval tinyMMLU:
lm_eval --model hf \
  --model_args pretrained=unsloth/Llama-3.1-8B,dtype=float16 \
  --tasks tinyMMLU \
  --batch_size 32 \
  --output_path ./eval_results/

Push logs to wandb:
"python eval_wandb.py --mmlu", assuming they are in the folder "eval_results"

Compare different ifevals:
python compare_ifeval.py --model1 eval_responses/lora_model_bf16_bf16_20260125_084605.json --model2 eval_responses/unsloth_Llama-3.1-8B_bf16_20260124_222342.json
  '''