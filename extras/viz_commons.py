"""Visualization commons"""
from typing import MutableMapping, Any, Union, Mapping, Dict, List, Tuple
import os
import numpy as np
import torch
from commons.params import NDict
from commons.logging import get_logger

_LOGGER = get_logger(os.path.basename(__file__))


def _to_device(data: Union[MutableMapping, List], device: Union[str, torch.device]) -> Union[MutableMapping, List]:
    """Copy pytorch tensors to device. Modifies collection inplace"""
    assert isinstance(data, (MutableMapping, List))
    itr = data.items() if isinstance(data, MutableMapping) else enumerate(data)
    for k, v in itr:
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device=device)
        elif isinstance(v, (MutableMapping, Dict, List)):
            _to_device(v, device)
        elif isinstance(v, (Mapping, Tuple)):  # type: ignore
            raise ValueError(f'Cannot modify {k} of type {type(v)} inplace')
    return data


def to_device(data: Union[Mapping, Tuple, List],
              device: Union[str, torch.device],
              *,
              detach: bool = False,
              contiguous: bool = False) -> Union[MutableMapping, List]:
    """Copy pytorch tensors to device. Modifies collection inplace"""
    if not isinstance(data, (MutableMapping, List)):
        if isinstance(data, Tuple):  # type: ignore
            data = list(data)
        elif isinstance(data, Mapping):
            data = dict(data)
        else:
            raise ValueError(f'Cannot modify data of type {type(data)} inplace')

    itr = data.items() if isinstance(data, MutableMapping) else enumerate(data)
    for k, v in itr:
        try:
            if isinstance(v, torch.Tensor):
                if detach:
                    v = v.detach()
                v = v.to(device=device)
                if contiguous:
                    v = v.contiguous()
                data[k] = v
            elif isinstance(v, (MutableMapping, Dict, List)):
                # print(f'Descending into {k}')
                to_device(v, device, detach=detach, contiguous=contiguous)
            elif isinstance(v, Mapping):
                _LOGGER.warning(f'Creating map for {k}')
                data[k] = to_device(v, device, detach=detach, contiguous=contiguous)
            elif isinstance(v, Tuple):  # type: ignore
                _LOGGER.warning(f'Creating new tuple for {k}')
                data[k] = to_device(v, device, detach=detach, contiguous=contiguous)
            elif isinstance(v, (str, int, float, np.ndarray)):
                continue
            else:
                _LOGGER.warning(f'to_device: Not moving {k} of type {type(v)})')
        except Exception as e:
            raise ValueError(f'Could not modify item {k} of type {type(v)}') from e
    return data


def to_cpu(data: Union[Mapping, Tuple, List],
           *,
           detach: bool = False,
           contiguous: bool = False) -> Union[MutableMapping, List]:
    """Detach and move to cpu, pytorch tensors. Modifies collection inplace"""
    return to_device(data, device='cpu', detach=detach, contiguous=contiguous)


def parse_options(global_options: List[str], tensor_options: List[str]) -> NDict:
    """Parse UI options"""
    return NDict(
        zero_mean='zero-mean' in tensor_options + global_options,
        prob_rows='prob-rows' in tensor_options + global_options,
        prob_cols='prob-cols' in tensor_options + global_options,
        unit_variance='unit-variance' in tensor_options + global_options,
        pre_unitvar_rows='pre-unitvar-rows' in tensor_options + global_options,
        pre_unitvar_cols='pre-unitvar-cols' in tensor_options + global_options,
        pre_zeromean_cols='pre-zeromean-cols' in tensor_options + global_options,
        pre_unitnorm_cols='pre-unitnorm-cols' in tensor_options + global_options,
        normalize_cols='normalize-cols' in tensor_options + global_options,
        normalize_rows='normalize-rows' in tensor_options + global_options,
        absolute='absolute' in tensor_options + global_options,
        log_scale='log-scale' in tensor_options,
        dot_product='dot-product' in tensor_options,
        exp='exp' in tensor_options,
        softmax_rows='softmax-rows' in tensor_options,
        softmax_cols='softmax-cols' in tensor_options,
        dp_cols_w_op='dp-cols-w-op' in tensor_options
    )


def preprocess(*,
               W: torch.Tensor,
               global_options: List[str] = [],
               tensor_options: List[str],
               data_dict: NDict
               ) -> torch.Tensor:
    """
    Preprocess tensor based on options.
    """
    options = parse_options(global_options, tensor_options)
    if options.dp_cols_w_op:
        # dot-product cols with decoder output vector
        # to get a similarity map similar to attributions map
        # (1, d_model, T) -> (1, T, d_model)
        out = data_dict['decoder_T_l'][-1]['FinalNorm'].transpose(1, 2)
        _, T, _D = out.shape
        if len(W.shape) == 3:
            H, _D, S = W.shape
            W = torch.matmul(out, W)
            assert W.shape == (H, T, S)
        elif len(W.shape) == 2:
            _D, S = W.shape
            W = torch.matmul(out, W).squeeze(0)
            assert W.shape == (T, S)
    if options.prob_rows:
        W = W.abs() / W.norm(p=1, dim=-1, keepdim=True)
    if options.prob_cols:
        W = W.abs() / W.norm(p=1, dim=-2, keepdim=True)
    if options.pre_zeromean_cols:
        W = W - W.mean(dim=-2, keepdim=True)
    if options.pre_unitnorm_cols:
        W = W / W.norm(p=2, dim=-2, keepdim=True)
    if options.pre_unitvar_rows:
        W = W / W.std(dim=-1, keepdim=True)
    if options.pre_unitvar_cols:
        W = W / W.std(dim=-2, keepdim=True)
    if options.dot_product:
        if len(W.shape) == 3:
            H, _D, S = W.shape
            W = torch.bmm(W.transpose(1, 2), W)  # (H, S, D) x (H, D, S) => (H, S, S)
            assert W.shape == (H, S, S)
        else:
            _D, S = W.shape
            W = torch.mm(W.transpose(0, 1), W)  # (S, D) x (D, S) => (S, S)
            assert W.shape == (S, S)
    if options.absolute:
        W = W.abs()
    if options.log_scale:
        W = W.log()
    if options.unit_variance:
        W = W / W.std()
    if options.zero_mean:
        W = W - W.mean()
    if options.normalize_cols:
        W = W / W.norm(p=2, dim=-2, keepdim=True)
    if options.normalize_rows:
        W = W / W.norm(p=2, dim=-1, keepdim=True)
    if options.exp:
        W = W.exp()
    if options.softmax_rows:
        W = torch.nn.functional.softmax(W, dim=-1)
    if options.softmax_cols:
        W = torch.nn.functional.softmax(W, dim=-2)

    return W


def normalize_vecs(T: torch.Tensor, *, dim: int):
    """Normalize the magnitude (p2-norm) of vectors to 1."""
    _LOGGER.info(f'Vec-norm: T.shape = {T.shape}, dim={dim}')
    T = T / T.norm(p=2, dim=dim, keepdim=True)


def _strip(s: str) -> str:
    return s.lstrip(chr(9601))


def _ids_to_tokens(ids: List[int], tokenizer: Any) -> List[str]:
    return [_strip(token) for token in tokenizer.convert_ids_to_tokens(ids)]


def get_knn(vectors: torch.Tensor, data_dict: NDict, K: int) -> np.ndarray:
    """Return K nearest neighbors"""
    if len(vectors.shape) == 1:
        vectors = vectors.unsqueeze(0)
    num_vecs, _dim = vectors.shape
    inp_index = data_dict['inp_index']
    _, _nn = inp_index.search(vectors.contiguous().numpy(), K)
    nn = np.asarray([_ids_to_tokens(_nn[:, k], data_dict['tokenizer']) for k in range(K)])  # (K, seq-len)
    return nn  # (#vecs,)
