"""Model related code for vizualization"""
from typing import List, Dict, Any, Optional, Mapping, Tuple, Union, Sequence, MutableMapping
import numpy as np
import torch
from commons.params import Params, PathName, NDict
from model.t5 import T5Grader
from pl.utils import load_model_from_checkpoint
import dataprep.pkshot_dataset as pkshot_dataset
from viz.viz_knn import knn_index
from viz.viz_commons import to_device, to_cpu
# import dataprep.iterable_dataset as iterable_dataset


def _view_activation_norms(a: torch.Tensor, w: torch.Tensor):
    return None
    # d_model = a.shape[-1]
    # assert w.shape[-1] == a.shape[-1]
    # # w = w.clone().detach()
    # norm = (a).pow(2).sum(dim=-1).sqrt()
    # print(f'shape={a.shape}\nd_model = {d_model}, sqrt(d_model)={math.sqrt(d_model)}')
    # print(f'activation norm: {pd.Series(norm.flatten().cpu()).describe()}')
    # # # print(f'1/sqrt(d_model)={1. / math.sqrt(d_model)}\nlayer_norm.weight: {pd.Series(w.cpu()).describe()}')


def _get_data(params: Params,
              model: T5Grader,
              datum_id: str,
              dataset: Optional[pkshot_dataset.PKShotDataset] = None) -> Tuple[NDict, pkshot_dataset.PKShotDataset]:
    """Get data for running through the model for visualization"""
    if dataset is None:
        params.run.ddp = False
        params.data.val.num_data_processes_per_gpu = 0
        params.data.val.batch_size_per_gpu = 1
        assert params.data.sampler.name == 'sampler3_pkshot'
        data_module = pkshot_dataset.DataModule(params, task='eval')
        data_module.prepare_data()
        data_module.setup('test')
        dataset = data_module.val_dataloader().dataset
    model.eval()

    def _match_item_category(out_batch: Mapping, input_batch: Mapping, cat: Optional[str]) -> bool:
        """Return tp, tn, fp, fn based on results"""
        if cat is None:
            return True

        if out_batch['topk_ids'][:, 0].squeeze().item() == input_batch['labels'].squeeze().item():
            if int(out_batch['preds'][0]) == 0:
                _cat = 'tn'
            else:
                _cat = 'tp'
        else:
            if int(out_batch['preds'][0]) == 0:
                _cat = 'fn'
            else:
                _cat = 'fp'

        return _cat == cat

    def _strip(s: str) -> str:
        """Strip leading underscores from sentencepice tokens"""
        return s.lstrip(chr(9601))

    def sample_dict(iD: int, in_batch: MutableMapping, out_batch: MutableMapping) -> NDict:
        in_batch['input_tokens'] = np.asarray([
            [_strip(token) for token in model.tokenizer.convert_ids_to_tokens(in_ids)]
            for in_ids in in_batch['input_ids']])
        in_batch['label_tokens'] = np.asarray([
            [_strip(token) for token in model.tokenizer.convert_ids_to_tokens(in_ids)]
            for in_ids in in_batch['labels']])
        in_batch['input_tokens_unstripped'] = np.asarray([
            [token for token in model.tokenizer.convert_ids_to_tokens(in_ids)]
            for in_ids in in_batch['input_ids']])
        assert len(in_batch['input_ids']) == 1
        # datum = dataloader.pkshot_dataset[iD]
        return NDict({
            'i': iD,
            'x': in_batch['x'][0],  # datum['x'],
            'y': in_batch['y'][0],  # datum['y'],
            'pred': out_batch['preds'][0],
            'input_ids': in_batch['input_ids'][0],
            'label_ids': in_batch['labels'][0],
            'pred_score': out_batch['topk_scores'][0, 0].item(),
            'in_batch': in_batch,
            # 'out_batch': out_batch
        })

    item_num = 1 if datum_id.isnumeric() else int(datum_id[2:])
    i_start = int(datum_id) if datum_id.isnumeric() else 0
    num_items = 0
    for i in range(i_start, i_start + 1000):
        input_batch = dataset.collate_fn([dataset[i]])  # type: ignore
        if 'batch_mask' in input_batch and not input_batch['batch_mask'][0]:  # Skip empty batch
            continue
        with torch.set_grad_enabled(False):  # Run a forward pass
            to_device(input_batch, model.device)
            # Need to run a forward pass to determine if the model's prediction is correct
            with torch.set_grad_enabled(False):
                out_batch = model.test_step(input_batch, _batch_idx=None, get_hidden=True)  # type: ignore
                if _match_item_category(out_batch, input_batch, None if datum_id.isnumeric() else datum_id[:2]):
                    num_items += 1
                    if num_items == item_num:
                        sample = sample_dict(iD=i, in_batch=input_batch, out_batch=out_batch)
                        break

    return NDict(datum=sample, tokenizer=dataset.tokenizer), dataset


def _hook_activations(model: T5Grader) -> Dict[str, Any]:
    module_names = ['encoder.final_layer_norm',
                    'decoder.final_layer_norm',
                    'encoder.block.0.layer.0.SelfAttention.relative_attention_bias',
                    'decoder.block.0.layer.0.SelfAttention.relative_attention_bias']
    param_name = NDict()
    for _l in range(len(model.t5.encoder.block)):
        module_names.extend([f'encoder.block.{_l}.layer.0.layer_norm',
                             f'encoder.block.{_l}.layer.0.SelfAttention.q',
                             f'encoder.block.{_l}.layer.0.SelfAttention.k',
                             f'encoder.block.{_l}.layer.0.SelfAttention.v',
                             f'encoder.block.{_l}.layer.0.SelfAttention.o',
                             f'encoder.block.{_l}.layer.0',
                             f'encoder.block.{_l}.layer.1.layer_norm',
                             f'encoder.block.{_l}.layer.1.DenseReluDense.wi',
                             f'encoder.block.{_l}.layer.1.DenseReluDense.wo',
                             f'encoder.block.{_l}.layer.1'
                             ])
        param_name[f'encoder.block.{_l}.layer.0'] = f'encoder.block.{_l}.layer.0.o+'
        param_name[f'encoder.block.{_l}.layer.1'] = f'encoder.block.{_l}.layer.1.ff2+'
    for _l in range(len(model.t5.decoder.block)):
        module_names.extend([f'decoder.block.{_l}.layer.0.layer_norm',
                             f'decoder.block.{_l}.layer.0.SelfAttention.q',
                             f'decoder.block.{_l}.layer.0.SelfAttention.k',
                             f'decoder.block.{_l}.layer.0.SelfAttention.v',
                             f'decoder.block.{_l}.layer.0.SelfAttention.o',
                             f'decoder.block.{_l}.layer.0',
                             f'decoder.block.{_l}.layer.1.layer_norm',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.q',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.k',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.v',
                             f'decoder.block.{_l}.layer.1.EncDecAttention.o',
                             f'decoder.block.{_l}.layer.1',
                             f'decoder.block.{_l}.layer.2.layer_norm',
                             f'decoder.block.{_l}.layer.2.DenseReluDense.wi',
                             f'decoder.block.{_l}.layer.2.DenseReluDense.wo',
                             f'decoder.block.{_l}.layer.2'
                             ])
        param_name[f'decoder.block.{_l}.layer.0'] = f'decoder.block.{_l}.layer.0.o+'
        param_name[f'decoder.block.{_l}.layer.1'] = f'decoder.block.{_l}.layer.1.o+'
        param_name[f'decoder.block.{_l}.layer.2'] = f'decoder.block.{_l}.layer.2.ff2+'

    activations_dict: Dict = {'input_embeddings': []}
    module_dict = dict(model.t5.named_modules())
    for module_name in module_names:
        def hook(_module: torch.nn.Module, input_: Any, output: Any, name: str = module_name) -> None:
            activations_dict[param_name[name] or name] = (
                output[0] if isinstance(output, tuple) else output).detach().clone()
            if name.endswith('.DenseReluDense.wo'):
                activations_dict[name + ':in'] = input_[0].detach().clone()
        module_dict[module_name].register_forward_hook(hook)

    def collect_embeddings(_embedding_module: torch.nn.Module, _input: Any, output: torch.Tensor) -> None:
        """Collect encoder and decoder input-embeddings"""
        activations_dict['input_embeddings'].append(output.detach().clone())
    module_dict['shared'].register_forward_hook(collect_embeddings)

    return activations_dict


def backprop_selfatt_attribs(attentions: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    Backprop through self-attention matrices and return position attributions from each layer's input positions towards
    top layers output positions.
    """
    attribs: List[torch.Tensor] = []
    with torch.set_grad_enabled(False):
        for i, att in enumerate(reversed(attentions)):
            # att.shape = (batch_size, num_heads, T, S) or (1, 1, T, S) or (B, H, S, S) or (B, H, T, T)
            assert att.shape[0] == 1, f'Batch size is not 1. Shape={att.shape}'
            att = att.squeeze(0).mean(0)  # (T, S)
            if i == 0:
                T, S = att.shape
                attribs.append(att.detach())  # (T, S)
            else:
                assert att.shape == (S, S)
                _att = torch.matmul(attribs[-1], att)  # (T, S)
                assert _att.shape == (T, S)
                attribs.append(_att.detach())  # (T, S)
            assert attribs[-1].shape == (T, S), f'attribs.shape = {attribs[-1].shape} != ({(T, S)})'
            # Ensure all attentions sum to 1.0
            assert all(torch.isclose(attribs[-1].sum(1), attribs[-1].new_ones([1]))), f'{attribs[-1].sum(1)}'
    return list(reversed(attribs))


def backprop_crossatt_attribs(cross_attentions: Sequence[torch.Tensor],
                              dec_attribs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    Backprop through cross-attention matrices and return position attributions from encoder output positions towards
    decoder output.
    """
    attribs: torch.Tensor
    attribs_l: List[torch.Tensor] = []
    cross_attentions = list(reversed(cross_attentions))
    dec_attribs = list(reversed(dec_attribs))
    with torch.set_grad_enabled(False):
        for i, att in enumerate(cross_attentions):
            # att.shape = (batch_size, num_heads, T, S)
            assert att.shape[0] == 1, f'Batch size is not 1. Shape={att.shape}'
            att = att.squeeze(0).mean(0)  # (T, S)
            if i == 0:
                T, S = att.shape
                attribs = att  # (T, S)
                attribs_l.append(att.detach())
            else:
                assert att.shape == (T, S)
                assert dec_attribs[i-1].shape == (T, T)
                _att = torch.matmul(dec_attribs[i-1], att)  # (T, S)
                assert _att.shape == (T, S)
                attribs = (_att + attribs)  # (T, S)
                attribs_l.append(torch.nn.functional.normalize(attribs, p=1, dim=1).detach())
            assert attribs.shape == (T, S), f'attribs.shape = {attribs.shape} != ({(T, S)})'
            assert all(torch.isclose(attribs.sum(1), attribs.new_full((1,), i + 1.))), f'{attribs.sum(1)}'
    return list(reversed(attribs_l))  # torch.nn.functional.normalize(attribs, p=1, dim=1).detach()  # (T, S)


def _collect_activations(model: T5Grader, input_batch: Dict, activations_dict: Dict) -> None:
    """Run a forward pass and collect activations"""
    model.eval()
    with torch.set_grad_enabled(False):  # Run a forward pass
        out_batch = model.test_step(input_batch, _batch_idx=None, get_hidden=True)  # type: ignore
    # Backprop through attention matrices collecting attribution from each level
    activations_dict['enc_attribs'] = backprop_selfatt_attribs(out_batch.t5_output.encoder_attentions)  # (S, S)
    activations_dict['dec_attribs'] = backprop_selfatt_attribs(out_batch.t5_output.decoder_attentions)  # (T, T)
    activations_dict['dec_cross_attribs'] = backprop_crossatt_attribs(
        out_batch.t5_output.cross_attentions, activations_dict['dec_attribs'])  # (T, S)
    activations_dict['enc_cross_attribs'] = backprop_selfatt_attribs(  # (T, S)
        out_batch.t5_output.encoder_attentions + (activations_dict['dec_cross_attribs'][0].unsqueeze(0).unsqueeze(0),))
    # Remove batch-dimension from per-layer activations
    activations_dict['enc_self_att'] = [att.squeeze(0)  # .mean(dim=0)
                                        for att in out_batch.t5_output.encoder_attentions]  # (H, S, S)
    activations_dict['dec_self_att'] = [att.squeeze(0)  # .mean(dim=0)
                                        for att in out_batch.t5_output.decoder_attentions]  # (H, T, T)
    activations_dict['cross_att'] = [att.squeeze(0)  # .mean(dim=0)
                                     for att in out_batch.t5_output.cross_attentions]  # (H, T, S)

    activations_dict['input_ids'] = input_batch['input_ids']
    activations_dict['label_ids'] = input_batch['labels']
    activations_dict['input_tokens'] = input_batch['input_tokens']
    activations_dict['label_tokens'] = input_batch['label_tokens']
    activations_dict['input_tokens_unstripped'] = input_batch['input_tokens_unstripped']


def _format_tensors(*,
                    model: T5Grader,
                    from_encoder: bool = True,
                    activations_dict: Mapping[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """Get tensor dictionary for display"""
    stack = model.t5.encoder if from_encoder else model.t5.decoder
    activations_p = Params(activations_dict)
    model_params = Params(dict(model.t5.named_parameters()))

    def unstack_activations(a: Optional[torch.Tensor], n_heads: int, proj_dim: Optional[int]) -> Optional[torch.Tensor]:
        if a is None:
            return None

        _N, S, inner_dim = a.shape
        assert _N == 1  # batch-size
        if proj_dim is not None:
            assert inner_dim == (n_heads * proj_dim)
        else:
            assert n_heads == 1
            proj_dim = inner_dim
        a = a.squeeze(0)  # (S, n_heads * proj_dim)
        return a.view(S, n_heads, proj_dim).permute(1, 2, 0)  # (n_heads, proj_dim, S)

    def layer_tensors(_l: int,
                      n_heads: int,
                      d_model: int,
                      key_value_proj_dim: int,
                      num_layers: int) -> Dict[str, torch.Tensor]:
        layer_path = f'{"encoder" if from_encoder else "decoder"}.block.{_l}'
        satt_weights = model_params[layer_path + '.layer.0']
        satt_activations = activations_p[layer_path + '.layer.0']
        ff_layer_path = layer_path + f'.layer.{1 if from_encoder else 2}'
        ff_weights = model_params[ff_layer_path]
        ff_activations = activations_p[ff_layer_path]
        if _l == 0:
            input_embeddings = activations_p['input_embeddings'][0 if from_encoder else 1]
            # (query_length, key_length, num_heads)
            relative_attention_bias = satt_activations.SelfAttention.relative_attention_bias
        else:
            input_embeddings = None
            relative_attention_bias = None
        # Relative attention bias is shared across layers
        relative_attention_bias = activations_p[
            f'{"encoder" if from_encoder else "decoder"}.block.0.layer.0.SelfAttention.relative_attention_bias']
        middle_row = min((relative_attention_bias.shape[0] + 1) // 2, relative_attention_bias.shape[0] - 1)
        relative_attention_bias = relative_attention_bias[middle_row].unsqueeze(0).permute([2, 0, 1])

        tensors_dict = Params({
            'W_NormSA': satt_weights['layer_norm.weight'].unsqueeze(0).unsqueeze(-1),  # (1, d_model, 1)
            # (d_model, n_heads * key_value_proj_dim) -> (n_heads, d_model, key_value_proj_dim)
            'Wq': satt_weights.SelfAttention.q.weight.view(
                d_model, n_heads, key_value_proj_dim).transpose(0, 1),
            'Wk': satt_weights.SelfAttention.k.weight.view(
                d_model, n_heads, key_value_proj_dim).transpose(0, 1),
            'Pos': relative_attention_bias,
            'Wv': satt_weights.SelfAttention.v.weight.view(
                d_model, n_heads, key_value_proj_dim).transpose(0, 1),
            # (self.n_heads * self.key_value_proj_dim, d_model) -> (self.n_heads, self.key_value_proj_dim, d_model)
            'Wo': satt_weights.SelfAttention.o.weight.view(n_heads, key_value_proj_dim, d_model),
            'W_NormFF': ff_weights['layer_norm.weight'].unsqueeze(0).unsqueeze(-1),  # (1, d_model, 1)
            'Wff1': ff_weights.DenseReluDense.wi.weight.unsqueeze(0),  # (1, d_model, d_ff)
            'Wff2': ff_weights.DenseReluDense.wo.weight.unsqueeze(0),  # (1, d_ff, d_model)

            'E': unstack_activations(input_embeddings, 1, d_model),
            'input_tokens': activations_p['input_tokens'][0],
            'input_ids': activations_p['input_ids'][0],
            # (S, S) or (T, T)
            'TopLayerAttribs': (activations_p['enc_attribs'][_l].unsqueeze(0) if from_encoder
                                else activations_p['dec_attribs'][_l].unsqueeze(0)),
            # (T, S)
            'DecOutAttribs': (activations_p['enc_cross_attribs'][_l].unsqueeze(0) if from_encoder
                              else None),
            # (n_heads, S, proj_dim)
            'NormSA': unstack_activations(satt_activations.layer_norm, 1, d_model),
            'Q': unstack_activations(satt_activations.SelfAttention['q'], n_heads, key_value_proj_dim),
            'K': unstack_activations(satt_activations.SelfAttention['k'], n_heads, key_value_proj_dim),
            'V': unstack_activations(satt_activations.SelfAttention['v'], n_heads, key_value_proj_dim),
            # (H, T, T) or (H, S, S)
            'SelfAtt': activations_p['enc_self_att' if from_encoder else 'dec_self_att'][_l],
            # (1, T, T) or (1, S, S)
            'SelfAttM': activations_p['enc_self_att' if from_encoder else 'dec_self_att'][_l].mean(0).unsqueeze(0),
            'O': unstack_activations(satt_activations.SelfAttention['o'], 1, d_model),  # (1, d_model, S)
            'O+': unstack_activations(satt_activations['o+'], 1, d_model),  # (1, d_model, S)
            'NormFF': unstack_activations(ff_activations.layer_norm, 1, d_model),
            'FF1': unstack_activations(ff_activations.DenseReluDense.wi, 1, None),
            'RELU': unstack_activations(activations_p[ff_layer_path + '.DenseReluDense.wo:in'], 1, None),
            'FF2': unstack_activations(ff_activations.DenseReluDense.wo, 1, d_model),
            'FF2+': unstack_activations(ff_activations['ff2+'], 1, d_model),
            'FinalDecOutAttribs': (activations_p['enc_cross_attribs'][_l+1].unsqueeze(0)
                                   if from_encoder and (_l == (num_layers - 1)) else None),
        })
        _view_activation_norms(satt_activations.layer_norm, satt_weights['layer_norm.weight'])
        _view_activation_norms(ff_activations.layer_norm, ff_weights['layer_norm.weight'])
        if not from_encoder:
            catt_weights = model_params[layer_path + '.layer.1']
            catt_activations = activations_p[layer_path + '.layer.1']
            tensors_dict.update({
                'W_NormCA': catt_weights['layer_norm.weight'].unsqueeze(0).unsqueeze(-1),  # (1, d_model, 1)
                # (d_model, n_heads * key_value_proj_dim) -> (n_heads, d_model, key_value_proj_dim)
                'CWq': catt_weights.EncDecAttention.q.weight.view(
                    d_model, n_heads, key_value_proj_dim).transpose(0, 1),
                'CWk': catt_weights.EncDecAttention.k.weight.view(
                    d_model, n_heads, key_value_proj_dim).transpose(0, 1),
                'CWv': catt_weights.EncDecAttention.v.weight.view(
                    d_model, n_heads, key_value_proj_dim).transpose(0, 1),
                # (self.n_heads * self.key_value_proj_dim, d_model) -> (self.n_heads, self.key_value_proj_dim, d_model)
                'CWo': catt_weights.EncDecAttention.o.weight.view(n_heads, key_value_proj_dim, d_model),
                # (n_heads, S, proj_dim)
                'CQ': unstack_activations(catt_activations.EncDecAttention['q'], n_heads, key_value_proj_dim),
                'CK': unstack_activations(catt_activations.EncDecAttention['k'], n_heads, key_value_proj_dim),
                'CV': unstack_activations(catt_activations.EncDecAttention['v'], n_heads, key_value_proj_dim),
                'CrossAtt': activations_p['cross_att'][_l],  # (H, T, S)
                'CrossAttM': activations_p['cross_att'][_l].mean(0).unsqueeze(0),  # (1, T, S)
                'CrossAtr': activations_p['dec_cross_attribs'][_l].unsqueeze(0),  # (1, T, S)
                'CO': unstack_activations(catt_activations.EncDecAttention['o'], 1, d_model),  # (1, d_model, T)
                'CO+': unstack_activations(catt_activations['o+'], 1, d_model),  # (1, d_model, T)
                'NormCA': unstack_activations(catt_activations.layer_norm, 1, d_model),
            })
            _view_activation_norms(catt_activations.layer_norm, catt_weights['layer_norm.weight'])
        if _l == (num_layers - 1):
            final_norm_path = 'encoder.final_layer_norm' if from_encoder else 'decoder.final_layer_norm'
            # (1, d_model, T)
            tensors_dict['FinalNorm'] = unstack_activations(activations_p[final_norm_path], 1, d_model)
            # (1, d_model, 1)
            tensors_dict['W_FinalNorm'] = model_params[final_norm_path].weight.unsqueeze(0).unsqueeze(-1)
            _view_activation_norms(activations_p[final_norm_path], model_params[final_norm_path].weight)
        return tensors_dict

    self_att_layers = [block.layer[0].SelfAttention for block in stack.block]
    return [
        layer_tensors(_l, sal.n_heads, sal.d_model, sal.key_value_proj_dim, len(self_att_layers))
        for _l, sal in enumerate(self_att_layers)
    ]


def get_viz_tensors(*,
                    checkpoint: Optional[PathName],
                    params_path: Optional[PathName],
                    device: Union[str, torch.device, None],
                    datum_id: str = 'tp1',
                    distance_measure: str,
                    model: Optional[T5Grader] = None,
                    dataset: Optional[pkshot_dataset.PKShotDataset] = None,
                    get_inp_index: bool = True
                    ) -> Tuple[NDict, pkshot_dataset.PKShotDataset, T5Grader]:
    """Produce data for visualization"""
    # Load Model
    if model is None:
        model, params = load_model_from_checkpoint(checkpoint, Params.read_file(params_path))
        model = model.to(device=device)
    else:
        params = Params.read_file(params_path)
        params.model = model.hyper.model

    # Get Activations for Visualization
    activations_dict: Dict
    data_dict, dataset = _get_data(params, model, datum_id, dataset)
    activations_dict = _hook_activations(model)
    _collect_activations(model, data_dict.datum['in_batch'], activations_dict)
    del data_dict.datum['in_batch']

    # Collect Tensors for Visualization
    encoder_T_l = _format_tensors(model=model,
                                  from_encoder=True,
                                  activations_dict=activations_dict)
    decoder_T_l = _format_tensors(model=model,
                                  from_encoder=False,
                                  activations_dict=activations_dict)

    # for name, t in activations_dict.items():
    #     shape = t.shape if isinstance(t, (torch.Tensor, np.ndarray)) else [
    #         _t.shape for _t in t]
    #     print(f'{name}: {shape}')
    # for _l, T_d in enumerate(encoder_T_l):
    #     for T_name, T in T_d.items():
    #         print(f'Encoder Layer {_l}.{T_name}.shape={T.shape if T is not None else None}')

    embedding_vectors = model.t5.shared.weight.cpu().detach()  # (32K, 512)
    # embedding_vectors_out = model.t5.lm_head.weight.detach().cpu()  # (32K, 512)
    # model_name = model.t5.config._name_or_path  # pylint: disable=protected-access
    # del model

    ret_dict: NDict = to_cpu(NDict({'encoder_T_l': encoder_T_l,  # type: ignore
                                    'decoder_T_l': decoder_T_l,
                                    'datum': data_dict.datum}),
                             detach=True, contiguous=True)
    ret_dict.update({'tokenizer': data_dict.tokenizer})  # type: ignore
    ret_dict.update({'inp_index': knn_index(embedding_vectors, distance_measure) if get_inp_index else None,  # type: ignore
                     'tokenizer': data_dict.tokenizer,
                     'embedding_vectors': embedding_vectors,
                     # 'embedding_vectors_out': embedding_vectors_out
                     'input_tokens_unstripped': activations_dict['input_tokens_unstripped'].squeeze(0)
                     })

    return ret_dict, dataset, model  # type: ignore
