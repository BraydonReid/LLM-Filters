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

MODEL_ID = "deepseek-r1-7b"
MODEL_TYPE = "deepseek-r1/deepseek-r1-7b"

# I am using the CPU because the GPU is not available.
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

print("Model and tokenizer loaded successfully.")

# this formats the data into a format that the model can understand
def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

def get_harmful_instructions():
    dataset = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def get_harmless_instructions():
    dataset = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

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

print("Tokenizing training data...")
harmful_tokens = tokenize_instructions(tokenizer, instructions=harmful_inst_train[:n_inst_train])
harmless_tokens = tokenize_instructions(tokenizer, instructions=harmless_inst_train[:n_inst_train])

# here i am using a rather small batch size because the model is on CPU
batch_size = 8

# We'll store activations in dicts
harmful_acts = defaultdict(list)
harmless_acts = defaultdict(list)

print("Collecting residual stream activations for harmful/harmless prompts (CPU only)...")
num_batches = (n_inst_train + batch_size - 1) // batch_size
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min(n_inst_train, start_idx + batch_size)

    # processes the harmful tokens
    harmful_logits, harmful_cache = model.run_with_cache(
        harmful_tokens[start_idx:end_idx],
        names_filter=lambda hook_name: 'resid' in hook_name,
        device="cpu",
        reset_hooks_end=True
    )
    # processes the harmful tokens
    harmless_logits, harmless_cache = model.run_with_cache(
        harmless_tokens[start_idx:end_idx],
        names_filter=lambda hook_name: 'resid' in hook_name,
        device="cpu",
        reset_hooks_end=True
    )

    # stores the activations in the dictionaries
    for key in harmful_cache:
        harmful_acts[key].append(harmful_cache[key])
        harmless_acts[key].append(harmless_cache[key])

    # fres up the cache to save memory
    del harmful_logits
    del harmful_cache
    del harmless_logits
    del harmless_cache
    gc.collect()

# concatenate the activations into a single tensor for each layer
harmful_acts = {k: torch.cat(v, dim=0) for k, v in harmful_acts.items()}
harmless_acts = {k: torch.cat(v, dim=0) for k, v in harmless_acts.items()}

# collects the residual stream activations for the harmful and harmless instructions
def get_act_idx(cache_dict, act_name, layer):
    key = utils.get_act_name(act_name, layer)
    return cache_dict[key]

activation_layers = ["resid_pre", "resid_mid", "resid_post"]
activation_refusals = defaultdict(list)

# difference of means to compute the refusal direction
pos = -1

print("Computing refusal direction for each layer...")
for layer_num in range(1, model.cfg.n_layers):
    for layer in activation_layers:
        harmful_mean_act = get_act_idx(harmful_acts, layer, layer_num)[:, pos, :].mean(dim=0)
        harmless_mean_act = get_act_idx(harmless_acts, layer, layer_num)[:, pos, :].mean(dim=0)
        refusal_dir = harmful_mean_act - harmless_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        activation_refusals[layer].append(refusal_dir)

# resid_pre is the first layer
selected_layers = ["resid_pre"]
activation_scored = []

for l in range(1, model.cfg.n_layers):
    for layer in selected_layers:
        direction = activation_refusals[layer][l - 1]
        activation_scored.append(direction)

# sorts the directions by their mean value in descending order
activation_scored = sorted(
    activation_scored,
    key=lambda x: abs(x.mean()),
    reverse=True
)

# tests the model with the harmful instructions
N_INST_TEST = 4

# uses foward hooks to generate the tokens
def _generate_with_hooks(
    model: HookedTransformer,
    tok: AutoTokenizer,
    tokens: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks=[]
) -> List[str]:
    bsz, seq_len = tokens.shape
    all_tokens = torch.zeros((bsz, seq_len + max_tokens_generated), dtype=torch.long, device="cpu")
    all_tokens[:, :seq_len] = tokens

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            # slices the tokens to the current length and passes them through the model to get the logits for the next token
            logits = model(all_tokens[:, : seq_len + i])
            # Pick top-1
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            # appends the next token
            all_tokens[:, seq_len + i] = next_tokens

    # decodes only the newly generated portion
    generated_portion = all_tokens[:, seq_len:]
    return tok.batch_decode(generated_portion, skip_special_tokens=True)

# generates the completions for the harmful instructions in order to test the model which is done in batches to help save memory
def get_generations(
    model: HookedTransformer,
    tok: AutoTokenizer,
    instructions: List[str],
    fwd_hooks=[],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    gens = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i : i + batch_size]
        tokens = tok.apply_chat_template(
            batch,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids.to("cpu")

        # generates the tokens for the batch
        out = _generate_with_hooks(
            model,
            tok,
            tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        gens.extend(out)
    return gens

# the actual ablation hook that subtracts out the projection onto refusal_dir from each residual stream.
# this is done to see how the model behaves when it is forced to not use the refusal direction 
def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(
        activation, direction.view(-1, 1),
        "... d_act, d_act single -> ... single"
    ) * direction
    return activation - proj

print("Getting baseline completions (no hooks) for the first 4 harmful test instructions...")
baseline_generations = get_generations(
    model, tokenizer, [t[0]["content"] for t in harmful_inst_test[:N_INST_TEST]], fwd_hooks=[]
)

print("\n\nBASELINE COMPLETIONS:")
for i, gen in enumerate(baseline_generations):
    print(f"[Test Harmful Inst {i}]\n{gen}\n{'-'*50}")

# evaluate the model with the top 20 refusal directions
EVAL_N = 20
evals = []

print("\nEvaluating with inference-time ablations for top 20 refusal directions...\n")
for refusal_dir in activation_scored[:EVAL_N]:
    hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
    #this appllies it to the same three major streams in every layer
    act_streams = ["resid_pre", "resid_mid", "resid_post"]
    fwd_hooks = [
        (utils.get_act_name(act_name, layer_idx), hook_fn)
        for layer_idx in range(model.cfg.n_layers)
        for act_name in act_streams
    ]
    intervention_gens = get_generations(
        model,
        tokenizer,
        [t[0]["content"] for t in harmful_inst_test[:N_INST_TEST]],
        fwd_hooks=fwd_hooks,
    )
    evals.append(intervention_gens)

# this allows us to see how each candidate direction changes the generation.
# this will allow us to read them and pick the best direction that "uncensors" or modifies the refusal style that we are looking for.
print("\n\nShowing the results for the 4 test instructions with top 20 direction candidates.\n")
print("We skip any that still produce 'I cannot' or 'I can't' (just a naive check).")

blacklist = ["I cannot", "I can't"]
for i in range(N_INST_TEST):
    user_prompt = harmful_inst_test[i][0]["content"]
    baseline = baseline_generations[i]
    print(f"\n\033[1mINSTRUCTION #{i}\033[0m\nUser: {user_prompt}\n")
    print(f"Baseline Completion:\n{baseline}")

    for dir_idx, directions_gens in enumerate(evals):
        candidate_gen = directions_gens[i]
        if not any(x in candidate_gen for x in blacklist):
            print(f"\n  -- Direction Candidate #{dir_idx} completion (no 'I cannot'/'I can't'): --")
            print(candidate_gen)
            break

# this is where we pick the direction we want to use for the orthogonalization.
BEST_DIR_IDX = 0
refusal_dir = activation_scored[BEST_DIR_IDX]
refusal_dir = refusal_dir.to("cpu")

# this is the orthogonalization step where we subtract out the projection of the weights onto the refusal direction
def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"],
    vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    proj = einops.einsum(
        matrix, vec.view(-1, 1),
        "... d_model, d_model single -> ... single"
    ) * vec
    return matrix - proj

print(f"\nPerforming permanent weight orthogonalization with direction index {BEST_DIR_IDX}...")

model.W_E.data = get_orthogonalized_matrix(model.W_E.data, refusal_dir)

for block_index in tqdm(range(model.cfg.n_layers)):
    block = model.blocks[block_index]
    block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O.data, refusal_dir)
    block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out.data, refusal_dir)

print("\nWeight orthogonalization complete. Let's do a final generation test.")

# tests again with the orthogonalized weights
orthogonalized_gens = get_generations(
    model, tokenizer, [t[0]["content"] for t in harmful_inst_test[:N_INST_TEST]]
)

print("\nCOMPARISON after orthogonalization:\n")
for i in range(N_INST_TEST):
    user_prompt = harmful_inst_test[i][0]["content"]
    base_gen = baseline_generations[i]
    orth_gen = orthogonalized_gens[i]

    print(f"\033[92m[User] {user_prompt}\n[Baseline Completion]\n{base_gen}\033[0m")
    print(f"\033[95m[Orthogonalized Completion]\n{orth_gen}\n{'-'*60}\033[0m")

