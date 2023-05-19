from typing import List, Optional
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch import Tensor
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ray


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device='cpu', parallelize: bool = False):
    if parallelize:
        assert device in [0, '0', 'cuda:0'], f'Device ({device}) must be set to "0" with parallelize model-arg'
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto' if parallelize else None
        )
    except ValueError:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )
        if parallelize:
            model = model.parallelize()
    if not parallelize:
        model = model.to(device=device)
    tokenizer = load_tokenizer(model_name)

    return model, tokenizer


def flatten(ll: List[List]) -> List:
    return [item for batch in ll for item in batch]


def concat(lT: List[torch.Tensor], pad_value=torch.nan) -> torch.Tensor:
    max_y = max([t.shape[1] for t in lT])
    if lT[0].dim() == 2:
        return torch.cat([torch.nn.functional.pad(t, pad=(0, max_y - t.shape[1]), value=pad_value) for t in lT])
    elif lT[0].dim() == 3:
        return torch.cat([torch.nn.functional.pad(t, pad=(0, 0, 0, max_y - t.shape[1]), value=pad_value) for t in lT])


def tokenize(text, tokenizer):
    return tokenizer(text, return_tensors='pt', return_attention_mask=True,
                     padding=True, add_special_tokens=False)


def forward(text, tokenizer, model):
    model.eval()
    device = model.device
    with torch.no_grad():
        model_input = tokenize(text, tokenizer)
        for key in model_input.keys():
            model_input[key] = model_input[key].to(device=device)
        model_output = model(input_ids=model_input['input_ids'],
                             attention_mask=model_input['attention_mask'],
                             output_hidden_states=True,
                             return_dict=True)
    return model_input, model_output


def generate(*, prompts: list, tokenizer, model, max_new_tokens=100, gen_args: str, batch_size: int):
    gen_kwargs = dict(
        greedy=dict(
            num_beams=1,
            num_beam_groups=1,
            do_sample=False
        ),
        deterministic=dict(
            num_beams=10,
            num_beam_groups=1,
            num_return_sequences=1,
            temperature=1.,
            top_p=1.0,
            do_sample=False
        ),
        sampling=dict(
            num_beams=1,
            temperature=0.7,
            top_p=0.9,
            num_beam_groups=1,
            num_return_sequences=1,
            do_sample=True
        ))
    model.eval()
    device = model.device
    seqs = []
    with torch.no_grad():
        for batch_start in range(0, len(prompts), batch_size):
            model_input = tokenizer(prompts[batch_start: batch_start + batch_size],
                                    return_tensors='pt', return_attention_mask=True, padding=True)
            input_ids = model_input.input_ids.to(device=device)
            attention_mask = model_input.attention_mask.to(device=device)
            kwargs = dict(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                early_stopping=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # return_dict_in_generate=True,
                # output_scores=True
            )
            kwargs.update(gen_kwargs[gen_args])
            outputs = model.generate(**kwargs)
            seqs.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return seqs


def get_sim(text, tokenizer, model):
    model.eval()
    with torch.no_grad():
        model_input, model_output = forward(text, tokenizer, model)
        embeddings = model.get_input_embeddings()(model_input['input_ids'])[:, 1:, :]
        last_layer = model_output.hidden_states[-1][:, :-1, :]
        attention_mask = model_input.attention_mask[:, :-1]
        dps = torch.linalg.vecdot(embeddings, last_layer) * attention_mask
    return dps


def get_all_probs(text, tokenizer, model, align_with_inp: bool = False):
    model.eval()
    with torch.no_grad():
        model_input, model_output = forward(text, tokenizer, model)
        all_probs = torch.softmax(model_output.logits, -1)
    all_probs = all_probs[:, :-1, :] if align_with_inp else all_probs
    return all_probs.detach(), model_input, model_output


def get_probs(texts, tokenizer, model, batch_size):
    model.eval()
    probs_list, attention_mask_list, all_probs_list = [], [], []
    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            all_probs, model_input, _ = get_all_probs(texts[batch_start: batch_start + batch_size], tokenizer, model)
            probs = all_probs[:, :-1, :].gather(-1, model_input['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
            attention_mask = model_input.attention_mask[:, :-1]
            probs = (probs * attention_mask).detach().cpu()
            all_probs = (all_probs[:, :-1, :] * attention_mask.unsqueeze(-1)).detach().cpu()
            probs_list.append(probs)
            attention_mask_list.append(attention_mask.detach().cpu())
            all_probs_list.append(all_probs)

    return probs_list, attention_mask_list, all_probs_list  # concat(all_probs_list)


def cuda_device_count():
    # Return the number of visible CUDA devices.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        return torch.cuda.device_count()


def ray_init(num_gpus_per_run: Optional[int] = None, cluster: str = 'local'):
    """Initialize or join ray cluster"""
    if cluster == 'local':
        # Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
        # ray cluster that was started in another environment.
        NUM_GPUS = cuda_device_count()
        MAX_PARALLEL_RUNS = NUM_GPUS // (num_gpus_per_run or 1)
        print(f'num_gpus_per_run={num_gpus_per_run}')
        ray.init(address='local', num_cpus=MAX_PARALLEL_RUNS + 2)
    else:
        # run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
        # If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
        # To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
        # ray.init(address='auto')
        ray.init(address='auto')


@ray.remote
def ray_generate(*, model_name, NUM_GPUS_PER_INSTANCE, **kwargs):
    import os
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    return generate(**kwargs, model=model, tokenizer=tokenizer)


@ray.remote
def ray_get_probs(*, model_name, NUM_GPUS_PER_INSTANCE, **kwargs):
    import os
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    return get_probs(**kwargs, tokenizer=tokenizer, model=model)


def avg_token_score(score: Tensor, mask: Tensor, min_occur: int):
    mt = torch.masked.masked_tensor(score, mask.to(dtype=torch.bool))
    meanp = mt.mean(0)
    stdp = mt.std(0)
    pos_count = mask.sum(0)  # monotonically decreasing with position
    trunc_pos = (pos_count < min_occur).sum()  # number of positions from the end with too few occurrences
    assert (~meanp.get_mask()).sum() == 0
    assert (~stdp.get_mask()).sum() == 0
    meanp = meanp.to_tensor(torch.nan)
    stdp = stdp.to_tensor(torch.nan)
    if trunc_pos > 0:
        return meanp[:-trunc_pos], stdp[:-trunc_pos], pos_count[:-trunc_pos]
    else:
        return meanp, stdp, pos_count


# Compute entropy of probability distributions over tokens for each position from all_prob
# all_prob is a tensor of shape (num_samples, num_tokens, vocab_size) and
# att_mask is a tensor of shape (num_samples, num_tokens)
def compute_entropy(all_prob, att_mask):
    att_mask = att_mask.cpu().unsqueeze(-1).expand(all_prob.shape)
    all_prob = torch.masked.masked_tensor(all_prob.cpu(), att_mask.to(dtype=torch.bool))
    entropy = -(all_prob * torch.log(all_prob)).sum(-1)
    return entropy.to_tensor(torch.nan)


def compute_entropy_from_list(all_prob_list, att_mask_list):
    return concat([compute_entropy(all_prob, att_mask) for all_prob, att_mask in zip(all_prob_list, att_mask_list)])


def compute_num_top_tokens(all_prob, mask, p):
    mask = mask.unsqueeze(-1)
    all_prob = all_prob * mask
    sorted_probs, sorted_indices = torch.sort(all_prob, descending=True, dim=-1)
    cum_probs = sorted_probs.cumsum(-1)  # (... , vocab_size)
    mask = (cum_probs <= p)
    # make sure we have at least one token
    mask[..., 0] = 1
    num_top_p_tokens = mask.sum(-1)  # (...)
    return num_top_p_tokens


def compute_num_top_tokens_from_list(all_prob_list, att_mask_list, p):
    return concat([compute_num_top_tokens(all_prob, att_mask, p).to(dtype=float) for all_prob, att_mask in zip(all_prob_list, att_mask_list)])


def plot_band(meanp, stdp, title=None, legend=None):

    fig = go.Figure([
        go.Scatter(
            name=legend or 'mean',
            y=meanp,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='+std dev',
            y=meanp + stdp,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='-std dev',
            y=meanp - stdp,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='Probability',
        title=title or 'Mean Position Probability With Std Dev',
        xaxis_title='Sequence Position',
        hovermode="x"
    )
    return fig


def lens_to_mask(seq_lens):
    max_len = max(seq_lens)
    return torch.arange(max_len).expand(len(seq_lens), max_len) < torch.tensor(seq_lens).unsqueeze(1)


def stack(lT: List[torch.Tensor]) -> torch.Tensor:
    seq_lens = [len(t) for t in lT]
    max_len = max(seq_lens)
    return torch.stack([torch.nn.functional.pad(t, pad=(0, max_len - len(t))) for t in lT]), lens_to_mask(seq_lens)


def remove_prompt(prob, att_mask, prompt_token_lens):
    assert len(prob) == len(att_mask) == len(prompt_token_lens)
    prob_list = []
    for i in range(len(prob)):
        _t_len = att_mask[i, prompt_token_lens[i]:].sum()
        _t_prob = prob[i, prompt_token_lens[i]:prompt_token_lens[i] + _t_len]
        prob_list.append(_t_prob)
    return stack(prob_list)
