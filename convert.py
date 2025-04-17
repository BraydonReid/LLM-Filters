import torch
import einops
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# loads the trained model that was trained
hooked = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen-7B",
    local_files_only=True,
    trust_remote_code=True,
    device="cpu",
    dtype=torch.float32,
)
hooked.load_state_dict(torch.load("deepseek_r1_7b(no filter).pth", map_location="cpu"), strict=False)

# loads the hugging face model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B",
    local_files_only=True,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# coppies the weights from the ablated model into the hugging face model
print("Copying ablated weights into HF model…")
state = hooked.state_dict()
n_layers = hooked.cfg.n_layers

for l in range(n_layers):
    wo = state[f"blocks.{l}.attn.W_O"]
    wo2 = einops.rearrange(wo, "n h m -> m (n h)")
    hf_model.model.layers[l].self_attn.o_proj.weight.data.copy_(wo2)
    W_in  = state[f"blocks.{l}.mlp.W_in"] 
    W_out = state[f"blocks.{l}.mlp.W_out"]

    hf_model.model.layers[l].mlp.down_proj.weight.data.copy_(W_in)
    hf_model.model.layers[l].mlp.up_proj.weight.data.copy_(W_out)

# save as a hugging face model
OUT_DIR = "abliterated-deepseek-r1-7b"
hf_model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Saved converted model + tokenizer to ./{OUT_DIR}/")
