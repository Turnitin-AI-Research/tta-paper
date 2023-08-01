from typing import List, Optional, Dict, Any
import os
import pickle
import math
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

from datasets import load_dataset_builder
from datasets import load_dataset
import torch
from torch import Tensor
import numpy as np
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go
import ray
import pandas as pd
from params import Params, NDict
from viz.viz_knn import knn_index


def is_enc_dec(model) -> bool:
    """Return True if this is an encoder-decoder model like T5."""
    if isinstance(model, (transformers.T5ForConditionalGeneration,
                          transformers.MT5ForConditionalGeneration)):
        return True
    elif isinstance(model, (transformers.GPT2LMHeadModel,
                            transformers.GPTNeoForCausalLM,
                            transformers.GPTJForCausalLM,
                            transformers.GPTNeoXForCausalLM,
                            transformers.BloomForCausalLM
                            )):
        return False
    elif type(model).__name__ == 'RWForCausalLM':
        return False
    elif type(model).__name__ == 'FalconForCausalLM':
        return False
    elif type(model).__name__.endswith('ForCausalLM'):
        return False
    else:
        raise NotImplementedError()


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device='cpu', parallelize: bool = False):
    if parallelize:
        assert device in [0, '0', 'cuda:0'], f'Device ({device}) must be set to "0" with parallelize model-arg'
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto' if parallelize else None,
            trust_remote_code=True
        )
    except ValueError:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map='auto' if parallelize else None
        )
        # if parallelize:  # T5 requires an explicit call to parallelize()
        #     model.parallelize()
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


def tokenize(text, tokenizer, add_special_tokens=False, return_offsets_mapping=False):
    return tokenizer(text, return_tensors='pt', return_attention_mask=True,
                     padding=True, add_special_tokens=add_special_tokens, return_offsets_mapping=return_offsets_mapping)


def forward(text, tokenizer, model):
    model.eval()
    device = model.device
    enc_dec = is_enc_dec(model)
    batch_size = 1 if isinstance(text, str) else len(text)
    with torch.no_grad():
        if not enc_dec:
            model_input = tokenize(text, tokenizer)
        else:
            model_input = tokenize('' if isinstance(text, str) else [''] * batch_size, tokenizer, add_special_tokens=True)
            decoder_input = tokenize(text, tokenizer)
            # model_input['decoder_input_ids'] = model._shift_right(decoder_input['input_ids'])
            # model_input['decoder_attention_mask'] = model._shift_right(decoder_input['attention_mask'])
            # model_input['decoder_attention_mask'][:, 0] = 1
            model_input['decoder_input_ids'] = decoder_input['input_ids']
            model_input['decoder_attention_mask'] = decoder_input['attention_mask']
        for key in model_input.keys():
            model_input[key] = model_input[key].to(device=device)
        if not enc_dec:
            model_output = model(input_ids=model_input['input_ids'],
                                 attention_mask=model_input['attention_mask'],
                                 output_hidden_states=True,
                                 return_dict=True)
        else:
            model_output = model(input_ids=model_input['input_ids'],
                                 attention_mask=model_input['attention_mask'],
                                 decoder_input_ids=model_input['decoder_input_ids'],
                                 decoder_attention_mask=model_input['decoder_attention_mask'],
                                 output_hidden_states=True,
                                 return_dict=True)
    return model_input, model_output


def forward2(text, tokenizer, model, decoder_text: str = '', output_attentions: bool = True):
    model.eval()
    device = model.device
    enc_dec = is_enc_dec(model)
    batch_size = 1 if isinstance(text, str) else len(text)
    with torch.no_grad():
        if not enc_dec:
            model_input = tokenize(text, tokenizer, return_offsets_mapping=True)
        else:
            model_input = tokenize(text, tokenizer, add_special_tokens=True, return_offsets_mapping=True)
            decoder_input = tokenize(decoder_text, tokenizer, add_special_tokens=True, return_offsets_mapping=True)
            model_input['decoder_offset_mapping'] = decoder_input['offset_mapping']
            model_input['decoder_input_ids'] = model._shift_right(decoder_input['input_ids'])
            model_input['decoder_attention_mask'] = model._shift_right(decoder_input['attention_mask'])
            model_input['decoder_attention_mask'][:, 0] = 1
            # model_input['decoder_input_ids'] = decoder_input['input_ids']
            # model_input['decoder_attention_mask'] = decoder_input['attention_mask']
        for key in model_input.keys():
            model_input[key] = model_input[key].to(device=device)
        if not enc_dec:
            model_output = model(input_ids=model_input['input_ids'],
                                 attention_mask=model_input['attention_mask'],
                                 output_hidden_states=True,
                                 output_attentions=output_attentions,
                                 return_dict=True)
        else:
            model_output = model(input_ids=model_input['input_ids'],
                                 attention_mask=model_input['attention_mask'],
                                 decoder_input_ids=model_input['decoder_input_ids'],
                                 decoder_attention_mask=model_input['decoder_attention_mask'],
                                 output_hidden_states=True,
                                 output_attentions=output_attentions,
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
        for batch_start in tqdm(range(0, len(prompts), batch_size)):
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


def hook_activations_t5(model: T5ForConditionalGeneration):
    module_names = ['encoder.final_layer_norm',
                    'decoder.final_layer_norm',
                    ]
    param_name = Params()
    for _l in range(len(model.encoder.block)):
        module_names.extend([f'encoder.block.{_l}.layer.0.layer_norm',
                             f'encoder.block.{_l}.layer.0.SelfAttention.q',
                             f'encoder.block.{_l}.layer.0.SelfAttention.k',
                             f'encoder.block.{_l}.layer.0.SelfAttention.v',
                             f'encoder.block.{_l}.layer.0.SelfAttention.o',
                             f'encoder.block.{_l}.layer.0.SelfAttention',
                             f'encoder.block.{_l}.layer.0',
                             f'encoder.block.{_l}.layer.1.layer_norm',
                             f'encoder.block.{_l}.layer.1.DenseReluDense.wi',
                             f'encoder.block.{_l}.layer.1.DenseReluDense.wo',
                             f'encoder.block.{_l}.layer.1'
                             ])
        param_name[f'encoder.block.{_l}.layer.0'] = f'encoder.block.{_l}.layer.0.o+'
        param_name[f'encoder.block.{_l}.layer.1'] = f'encoder.block.{_l}.layer.1.ff2+'
    for _l in range(len(model.decoder.block)):
        module_names.extend([f'decoder.block.{_l}.layer.0.layer_norm',
                             f'decoder.block.{_l}.layer.0.SelfAttention.q',
                             f'decoder.block.{_l}.layer.0.SelfAttention.k',
                             f'decoder.block.{_l}.layer.0.SelfAttention.v',
                             f'decoder.block.{_l}.layer.0.SelfAttention.o',
                             f'decoder.block.{_l}.layer.0.SelfAttention',
                             f'decoder.block.{_l}.layer.0',
                             f'decoder.block.{_l}.layer.1.layer_norm',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.q',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.k',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.v',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.o',
                             f'decoder.block.{_l}.layer.1.EncDecAttention',
                             f'decoder.block.{_l}.layer.1',
                             f'decoder.block.{_l}.layer.2.layer_norm',
                             f'decoder.block.{_l}.layer.2.DenseReluDense.wi',
                             f'decoder.block.{_l}.layer.2.DenseReluDense.wo',
                             f'decoder.block.{_l}.layer.2'
                             ])
        param_name[f'decoder.block.{_l}.layer.0'] = f'decoder.block.{_l}.layer.0.o+'
        param_name[f'decoder.block.{_l}.layer.1'] = f'decoder.block.{_l}.layer.1.o+'
        param_name[f'decoder.block.{_l}.layer.2'] = f'decoder.block.{_l}.layer.2.ff2+'

    activations_dict: Dict = Params({'input_embeddings': []})
    module_dict = dict(model.named_modules())
    for module_name in module_names:
        def hook(_module: torch.nn.Module, input_: Any, output: Any, name: str = module_name) -> None:
            # print(f'Hooking {param_name[name] or name}')
            if name.endswith('.SelfAttention') or name.endswith('.EncDecAttention'):
                if len(output) >= 5:
                    # T5Attention.forward was enhanced to output attention scores
                    activations_dict[f'{name}.attention_score'] = output[4].cpu().detach().squeeze(0)
                    # activations_dict[f'{name}.attention_weights'] = output[3].cpu().detach().squeeze(0)
                activations_dict[f'{name}.position_bias'] = output[2].cpu().detach().squeeze(0)
            else:
                activations_dict[param_name[name] or name] = (
                    output[0] if isinstance(output, tuple) else output).cpu().detach().squeeze(0)
            if name.endswith('.DenseReluDense.wo'):
                activations_dict[name + ':in'] = input_[0].cpu().detach().squeeze(0)
        try:
            module_dict[module_name].register_forward_hook(hook)
        except KeyError:
            print(f'Could not find activation: {module_name}')

    def collect_embeddings(_embedding_module: torch.nn.Module, _input: Any, output: torch.Tensor) -> None:
        """Collect encoder and decoder input-embeddings"""
        activations_dict['input_embeddings'].append(output.cpu().detach())
    module_dict['shared'].register_forward_hook(collect_embeddings)

    return activations_dict


def strip_t5_token(s: str) -> str:
    """Strip leading underscores from sentencepice tokens"""
    return s.lstrip(chr(9601))


def get_activations_t5(context, completion, *, model=None, tokenizer=None, model_name, distance_measure='IP', NUM_GPUS_PER_INSTANCE):
    assert distance_measure in ['IP', 'L2']
    assert isinstance(context, str)
    assert isinstance(completion, str)
    if model is None:
        model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    activations_dict = hook_activations_t5(model)
    model_input, model_output = forward2(context, tokenizer, model, decoder_text=completion)

    data_dict = NDict({k: v.cpu().detach() for k, v in model_input.items()})
    data_dict['logits'] = model_output.logits.cpu().detach()
    data_dict['encoder_hidden_states'] = [layer.cpu().detach() for layer in model_output.encoder_hidden_states]
    data_dict['decoder_hidden_states'] = [layer.cpu().detach() for layer in model_output.decoder_hidden_states]
    data_dict['encoder_last_hidden_state'] = model_output.encoder_last_hidden_state.cpu().detach()
    data_dict.activations = activations_dict
    data_dict.embedding_vectors = model.get_input_embeddings().weight.cpu().detach()
    data_dict['inp_index'] = knn_index(data_dict.embedding_vectors, distance_measure)
    data_dict.tokenizer = tokenizer
    data_dict.input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.input_ids.squeeze(0))
    data_dict.input_tokens = [strip_t5_token(token) for token in data_dict.input_tokens_unstripped]
    data_dict.encoder_attention_weights = [t.cpu().detach() for t in model_output.encoder_attentions]
    data_dict.decoder_attention_weights = [t.cpu().detach() for t in model_output.decoder_attentions]
    if 'decoder_input_ids' in model_input:
        data_dict.decoder_input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.decoder_input_ids.squeeze(0))
        data_dict.decoder_input_tokens = [strip_t5_token(token) for token in data_dict.decoder_input_tokens_unstripped]

    return model, tokenizer, data_dict


def hook_activations_falcon(model, activations_dict):
    # list(dict(model.named_modules()).keys())
    for module_name, module in model.named_modules():
        def hook(_module: torch.nn.Module, input_: Any, output: Any, name: str = module_name) -> None:
            if isinstance(output, torch.Tensor):
                activations_dict[name] = output.cpu().detach()
            else:
                print(f'Skipping activation {name or module_name}')
        if 'layernorm' in module_name or 'ln' in module_name or 'word_embeddings' in module_name:
            module.register_forward_hook(hook)
        else:
            print(f'Skipping module {module_name}')


def get_activations_falcon(text, *, model=None, model_name, tokenizer=None, distance_measure='IP', NUM_GPUS_PER_INSTANCE):
    assert distance_measure in ['IP', 'L2']
    assert isinstance(text, str)
    if model is None:
        model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    activations_dict = NDict()
    hook_activations_falcon(model, activations_dict)
    model_input, model_output = forward2(text, tokenizer, model)

    data_dict = NDict({k: v.cpu().detach() for k, v in model_input.items()})
    data_dict['logits'] = model_output.logits.cpu().detach()
    data_dict['hidden_states'] = [layer.cpu().detach() for layer in model_output.hidden_states]
    data_dict.activations = activations_dict
    data_dict.embedding_vectors = model.get_input_embeddings().weight.cpu().detach()
    data_dict['inp_index'] = knn_index(data_dict.embedding_vectors, distance_measure)
    data_dict.tokenizer = tokenizer
    data_dict.input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.input_ids.squeeze(0))
    data_dict.input_tokens = data_dict.input_tokens_unstripped  # [_strip(token) for token in data_dict.input_tokens_unstripped]
    if 'decoder_input_ids' in model_input:
        data_dict.decoder_input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.decoder_input_ids)
        data_dict.decoder_input_tokens = data_dict.decoder_input_tokens_unstripped  # [_strip(token) for token in data_dict.decoder_input_tokens_unstripped]

    return model, tokenizer, data_dict


def hook_activations_gpt_neo(model, activations_dict):
    # list(dict(model.named_modules()).keys())
    for module_name, module in model.named_modules():
        def hook(_module: torch.nn.Module, input_: Any, output: Any, name: str = module_name) -> None:
            if isinstance(output, torch.Tensor):
                activations_dict[name] = output.cpu().detach()
        if 'layernorm' in module_name or 'ln' in module_name or 'word_embeddings' in module_name:
            module.register_forward_hook(hook)


def get_activations_gpt_neo(text, *, model=None, model_name, tokenizer=None, distance_measure='IP', NUM_GPUS_PER_INSTANCE):
    assert distance_measure in ['IP', 'L2']
    assert isinstance(text, str)
    if model is None:
        model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    activations_dict = NDict()
    hook_activations_gpt_neo(model, activations_dict)
    model_input, model_output = forward2(text, tokenizer, model)

    data_dict = NDict({k: v.cpu().detach() for k, v in model_input.items()})
    data_dict['logits'] = model_output.logits.cpu().detach()
    data_dict['hidden_states'] = [layer.cpu().detach() for layer in model_output.hidden_states]
    data_dict.activations = activations_dict
    data_dict.embedding_vectors = model.get_input_embeddings().weight.cpu().detach()
    data_dict['inp_index'] = knn_index(data_dict.embedding_vectors, distance_measure)
    data_dict.tokenizer = tokenizer
    data_dict.attention_weights = [t.cpu().detach() for t in model_output.attentions]
    data_dict.input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.input_ids.squeeze(0))
    data_dict.input_tokens = data_dict.input_tokens_unstripped  # [_strip(token) for token in data_dict.input_tokens_unstripped]
    if 'decoder_input_ids' in model_input:
        data_dict.decoder_input_tokens_unstripped = tokenizer.convert_ids_to_tokens(model_input.decoder_input_ids)
        data_dict.decoder_input_tokens = data_dict.decoder_input_tokens_unstripped  # [_strip(token) for token in data_dict.decoder_input_tokens_unstripped]

    return model, tokenizer, data_dict


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
    if is_enc_dec(model):
        assert not align_with_inp, 'align_with_inp not supported for encoder-decoder models'
    with torch.no_grad():
        model_input, model_output = forward(text, tokenizer, model)
        all_probs = torch.softmax(model_output.logits, -1)
    if align_with_inp:
        all_probs = all_probs[:, :-1, :]
    return all_probs.detach(), model_input, model_output


def get_probs(texts, tokenizer, model, batch_size):
    model.eval()
    enc_dec = is_enc_dec(model)
    probs_list, attention_mask_list, all_probs_list = [], [], []
    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            all_probs, model_input, _ = get_all_probs(texts[batch_start: batch_start + batch_size], tokenizer, model)
            if enc_dec:
                input_ids = model_input['decoder_input_ids']
                attention_mask = model_input['decoder_attention_mask']
            else:
                input_ids = model_input['input_ids']
                attention_mask = model_input['attention_mask']
            probs = all_probs[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            attention_mask = attention_mask[:, :-1]
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model, tokenizer = load_model(model_name, device='0', parallelize=(NUM_GPUS_PER_INSTANCE > 1))
    return get_probs(**kwargs, tokenizer=tokenizer, model=model)


def masked_percentile(x, mask, p):
    # Compute percentile across batch dimension
    assert x.dim() == 2
    assert x.shape == mask.shape
    mask = mask.to(dtype=torch.bool)
    return torch.stack([torch.quantile(x[mask[:, j], j], p) for j in range(x.shape[1])])


def run_get_probs(probs_save_filename, model_name, texts, batch_size2, device, NUM_GPUS_PER_INSTANCE, model, tokenizer, use_ray: bool = True):
    if os.path.exists(probs_save_filename):
        with open(probs_save_filename, 'rb') as fp:
            prob_list, att_mask_list, all_prob_list = pickle.load(fp).values()
    else:
        responses = []
        num_raylets = math.ceil(cuda_device_count() / NUM_GPUS_PER_INSTANCE)
        num_samples_per_raylet = math.ceil(len(texts) / num_raylets)
        if (num_raylets > 1) or use_ray:
            ray.init(ignore_reinit_error=True, address='local')
            futures = [
                ray_get_probs.options(num_gpus=NUM_GPUS_PER_INSTANCE).remote(model_name=model_name, NUM_GPUS_PER_INSTANCE=NUM_GPUS_PER_INSTANCE,
                                                                             texts=texts[i: i + num_samples_per_raylet], batch_size=batch_size2)
                for i in range(0, len(texts), num_samples_per_raylet)]
            responses = ray.get(futures)
            _p, _a, _ap = zip(*responses)
            prob_list, att_mask_list, all_prob_list = flatten(_p), flatten(_a), flatten(_ap)
            ray.shutdown()
        else:
            del_model = False
            if model is None:
                model, tokenizer = load_model(model_name, device, (NUM_GPUS_PER_INSTANCE > 1))
                del_model = True
            prob_list, att_mask_list, all_prob_list = get_probs(
                texts=texts, tokenizer=tokenizer, model=model, batch_size=batch_size2)
            if del_model:
                del model
        with open(probs_save_filename, 'wb') as fp:
            pickle.dump(dict(prob_list=prob_list, att_mask_list=att_mask_list, all_prob_list=all_prob_list), fp)
    prob, att_mask = concat(prob_list), concat(att_mask_list, 0)
    return all_prob_list, att_mask_list, prob, att_mask


def get_seqs(path, name, split, size, seq_len, model_name):
    tokenizer = load_tokenizer(model_name)
    if name != 'random':
        dataset = load_dataset(path, name, split=split, streaming=True)
        dataset_iter = iter(dataset)
        texts, seqs = [], []
        for i, row in enumerate(dataset_iter):
            tokenized = tokenize(row['text'], tokenizer, return_offsets_mapping=True)
            seq = tokenized.input_ids
            if seq.shape[1] >= seq_len:
                seqs.append(seq[:seq_len])
                texts.append(row['text'][:tokenized.offset_mapping[0, seq_len-1, 1]])
            if len(texts) >= size:
                break
    else:
        # Generate random id sequences
        ids = list(tokenizer.get_vocab().values())
        seqs = np.random.choice(ids, size=(size, seq_len), replace=True)
        texts = [tokenizer.decode(seq) for seq in seqs]

    display(pd.Series([len(text) for text in texts]).describe())
    return seqs, texts


def avg_token_score(score: Tensor, mask: Tensor, min_occur: int, device='cpu', percentile: Optional[float] = 0.05):
    score_device = score.device
    mt = torch.masked.masked_tensor(score, mask.to(dtype=torch.bool))
    meanp = mt.mean(0)
    stdp = mt.std(0)
    if percentile is not None:
        minp = masked_percentile(score, mask, percentile)
        maxp = masked_percentile(score, mask, 1. - percentile)
    else:
        minp = mt.amin(0)
        maxp = mt.amax(0)
    pos_count = mask.sum(0)  # monotonically decreasing with position
    trunc_pos = (pos_count < min_occur).sum()  # number of positions from the end with too few occurrences
    assert (~meanp.get_mask()).sum() == 0
    assert (~stdp.get_mask()).sum() == 0
    meanp = meanp.to_tensor(torch.nan)
    stdp = stdp.to_tensor(torch.nan)
    if percentile is None:
        minp = minp.to_tensor(torch.nan)
        maxp = maxp.to_tensor(torch.nan)

    if trunc_pos > 0:
        # remove positions with too few occurrences
        meanp, stdp, pos_count, minp, maxp = meanp[:-trunc_pos], stdp[:-
                                                                      trunc_pos], pos_count[:-trunc_pos], minp[:-trunc_pos], maxp[:-trunc_pos]

    return meanp.to(device=score_device), stdp.to(device=score_device), pos_count.to(device=score_device), minp.to(device=score_device), maxp.to(device=score_device)


# Compute entropy of probability distributions over tokens for each position from all_prob
# all_prob is a tensor of shape (num_samples, num_tokens, vocab_size) and
# att_mask is a tensor of shape (num_samples, num_tokens)
def compute_entropy(all_prob, att_mask, device='cpu'):
    all_prob_device = all_prob.device
    att_mask = att_mask.to(device=device).unsqueeze(-1).expand(all_prob.shape)
    all_prob = torch.masked.masked_tensor(all_prob.to(device=device), att_mask.to(dtype=torch.bool))
    entropy = -(all_prob * torch.log(all_prob + 1e-13)).sum(-1)
    return entropy.to_tensor(torch.nan).to(device=all_prob_device)


def compute_entropy_from_list(all_prob_list, att_mask_list, device='cpu'):
    return concat([compute_entropy(all_prob, att_mask, device) for all_prob, att_mask in zip(all_prob_list, att_mask_list)])


def compute_num_top_tokens(all_prob, mask, p, device='cpu'):
    all_prob_device = all_prob.device
    mask = mask.unsqueeze(-1).to(device=device)
    all_prob = all_prob.to(device=device) * mask
    sorted_probs, sorted_indices = torch.sort(all_prob, descending=True, dim=-1)
    cum_probs = sorted_probs.cumsum(-1)  # (... , vocab_size)
    mask = (cum_probs <= p)
    # make sure we have at least one token
    mask[..., 0] = 1
    num_top_p_tokens = mask.sum(-1)  # (...)
    return num_top_p_tokens.to(device=all_prob_device)


def compute_num_top_tokens_from_list(all_prob_list, att_mask_list, p, device='cpu'):
    return concat([compute_num_top_tokens(all_prob, att_mask, p, device).to(dtype=float) for all_prob, att_mask in zip(all_prob_list, att_mask_list)])


def plot_band(meanp, minp, maxp, title=None, legend=None, show_smoothed: bool = False):

    fig = go.Figure([
        go.Scatter(
            name=legend or 'mean',
            y=meanp,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)')
        ),
        go.Scatter(
            name='max',
            y=maxp,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='min',
            y=minp,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(180, 180, 180, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    if show_smoothed:
        fig.add_trace(
            go.Scatter(
                name=f'Smoothed {legend}',
                mode='lines',
                y=signal.savgol_filter(meanp,
                                       100,  # window size used for filtering
                                       3),  # order of fitted polynomial,
                # marker=dict(
                #     size=1,
                #     # color='mediumpurple',
                #     symbol='circle-dot'
                # )
            )
        )

    fig.update_layout(
        yaxis_title=legend,
        title=title,
        xaxis_title='Token Position',
        hovermode="x"
    )
    return fig


def plot_band_(meanp, stdp, title=None, legend=None):
    plot_band(meanp, meanp - stdp, meanp + stdp, title=title, legend=legend)


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

