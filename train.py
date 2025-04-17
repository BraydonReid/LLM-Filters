import os
import gc
import torch
import einops
import functools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from torch import Tensor
from typing import List

torch.set_grad_enabled(False)

# loading the model and tokenizer only on the CPU becuase ROCM is not available
print("Loading model onto CPU (float32)...")
model = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen-7B",
    local_files_only=True,
    trust_remote_code=True,
    device="cpu",
    dtype=torch.float32,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B",
    local_files_only=True,
    trust_remote_code=True 
)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded successfully.\n")

# this formats the data into a format that the model can understand
def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

def get_harmful_instructions():
    d = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(d['train']['text']), reformat_texts(d['test']['text'])

def get_harmless_instructions():
    d = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(d['train']['text']), reformat_texts(d['test']['text'])

print("Downloading harmful instructions dataset...")
harmful_inst_train, harmful_inst_test = get_harmful_instructions()
print("Downloading harmless instructions dataset...")
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

n_inst_train = min(256, len(harmful_inst_train), len(harmless_inst_train))


# applies a chat like template to the instructions which helps the model understand the data
def tokenize_instructions(tok, instructions):
    return tok.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

print("\nTokenizing training data...")
harmful_tokens   = tokenize_instructions(tokenizer, harmful_inst_train[:n_inst_train])
harmless_tokens  = tokenize_instructions(tokenizer, harmless_inst_train[:n_inst_train])

# here i am using a rather small batch size because the model is on CPU
batch_size = 16
harmful_acts = defaultdict(list)
harmless_acts = defaultdict(list)

print("\nCollecting residual stream activations (CPU only)...")
num_batches = (n_inst_train + batch_size - 1) // batch_size
for i in tqdm(range(num_batches)):
    start, end = i * batch_size, min(n_inst_train, (i+1)*batch_size)

    h_logits, h_cache = model.run_with_cache(
        harmful_tokens[start:end],
        names_filter=lambda name: 'resid' in name,
        device="cpu",
        reset_hooks_end=True
    )
    g_logits, g_cache = model.run_with_cache(
        harmless_tokens[start:end],
        names_filter=lambda name: 'resid' in name,
        device="cpu",
        reset_hooks_end=True
    )

    for key in h_cache:
        harmful_acts[key].append(h_cache[key])
        harmless_acts[key].append(g_cache[key])

    del h_logits, h_cache, g_logits, g_cache
    gc.collect()

harmful_acts  = {k: torch.cat(v, dim=0) for k,v in harmful_acts.items()}
harmless_acts = {k: torch.cat(v, dim=0) for k,v in harmless_acts.items()}

# collects the residual stream activations for the harmful and harmless instructions
def get_act_idx(cache, name, layer):
    return cache[utils.get_act_name(name, layer)]

activation_layers = ["resid_pre", "resid_mid", "resid_post"]
activation_refusals = defaultdict(list)
pos = -1

print("\nComputing refusal directions...")
for layer_num in range(1, model.cfg.n_layers):
    for act_name in activation_layers:
        mean_h = get_act_idx(harmful_acts,  act_name, layer_num)[:, pos, :].mean(dim=0)
        mean_g = get_act_idx(harmless_acts, act_name, layer_num)[:, pos, :].mean(dim=0)
        dir_ = mean_h - mean_g
        activation_refusals[act_name].append(dir_ / dir_.norm())

# only using "resid_pre" for simplicity 
activation_scored = [
    activation_refusals["resid_pre"][i]
    for i in range(model.cfg.n_layers-1)
]
activation_scored.sort(key=lambda x: abs(x.mean()), reverse=True)

# tests the model with the harmful instructions
# uses foward hooks to generate the tokens
N_INST_TEST = 20

def _generate_with_hooks(model, tok, tokens, max_tokens_generated=64, fwd_hooks=[]):
    bsz, seq_len = tokens.shape
    all_ids = torch.zeros((bsz, seq_len + max_tokens_generated), dtype=torch.long, device="cpu")
    all_ids[:, :seq_len] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_ids[:, :seq_len+i])
            all_ids[:, seq_len+i] = logits[:, -1, :].argmax(dim=-1)
    return tok.batch_decode(all_ids[:, seq_len:], skip_special_tokens=True)

def get_generations(model, tok, instructions, fwd_hooks=[], max_tokens_generated=64, batch_size=4):
    gens = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        toks = tok.apply_chat_template(
            batch,
            padding=True, truncation=False,
            return_tensors="pt", return_dict=True,
            add_generation_prompt=True
        ).input_ids.to("cpu")
        gens.extend(_generate_with_hooks(model, tok, toks, max_tokens_generated, fwd_hooks))
    return gens

def direction_ablation_hook(activation, hook, direction):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1,1), "... d, d single -> ... single") * direction
    return activation - proj

# tests the model with the harmful instructions to get the baseline completions
print("\nGetting baseline completions...")
baseline_generations = get_generations(
    model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=[]
)

print(f"🔍 len(baseline_generations) = {len(baseline_generations)} (expected {N_INST_TEST})\n")
for i, gen in enumerate(baseline_generations):
    print(f"[Harmful Inst {i} Baseline] {gen}\n{'-'*40}")

# evaluate the model with the top 20 refusal directions
EVAL_N = min(20, len(activation_scored))
evals = []

print("\nEvaluating inference-time ablations (top directions)...")
for dir_idx in range(EVAL_N):
    rd = activation_scored[dir_idx]
    hook_fn = functools.partial(direction_ablation_hook, direction=rd)
    streams = ["resid_pre","resid_mid","resid_post"]
    hooks = [
        (utils.get_act_name(s, l), hook_fn)
        for l in range(model.cfg.n_layers)
        for s in streams
    ]
    gens = get_generations(model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=hooks)
    evals.append(gens)

for idx, candidate in enumerate(evals):
    print(f"🔍 Candidate #{idx} has {len(candidate)} completions")

# this allows us to see how each candidate direction changes the generation.
print("\nShowing best non‑refusal candidate per instruction:")
blacklist = ["I cannot", "I can't"]

for i in range(N_INST_TEST):
    print(f"\n--- Instruction #{i} ---")
    print("User:", harmful_inst_test[i][0]["content"])
    if i >= len(baseline_generations):
        print("⚠️ No baseline completion.")
        continue
    print("Baseline:", baseline_generations[i])

    found = False
    for dir_idx, cand in enumerate(evals):
        if i < len(cand) and not any(w in cand[i] for w in blacklist):
            print(f"Candidate #{dir_idx}:", cand[i])
            found = True
            break
    if not found:
        print("No acceptable candidate found.")


# this is where we pick the direction we want to use for the orthogonalization.
BEST_DIR_IDX = 0
refusal_dir = activation_scored[BEST_DIR_IDX].to("cpu")

def get_orthogonalized_matrix(matrix, vec):
    proj = einops.einsum(matrix, vec.view(-1,1), "... d, d single -> ... single") * vec
    return matrix - proj

print(f"\nOrthogonalizing weights using direction #{BEST_DIR_IDX}...")
model.W_E.data = get_orthogonalized_matrix(model.W_E.data, refusal_dir)
for b in tqdm(model.blocks, desc="Orthogonalizing blocks"):
    b.attn.W_O.data = get_orthogonalized_matrix(b.attn.W_O.data, refusal_dir)
    b.mlp.W_out.data   = get_orthogonalized_matrix(b.mlp.W_out.data, refusal_dir)

# tests again with the orthogonalized weights
print("\nGenerating after orthogonalization...")
orth_gens = get_generations(model, tokenizer, harmful_inst_test[:N_INST_TEST])

for i in range(N_INST_TEST):
    base = baseline_generations[i] if i < len(baseline_generations) else "<missing>"
    orth = orth_gens[i] if i < len(orth_gens) else "<missing>"
    print(f"\n[Inst {i}]")
    print("Baseline   :", base)
    print("Orthogonal :", orth)
    print("-"*50)
    
# saves the model
SAVE_PATH = "./deepseek_r1_7b(no filter).pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved HookedTransformer state dict to {SAVE_PATH}")
