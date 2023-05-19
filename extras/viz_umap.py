"""tsne and umap related code"""
import os
import pickle
from typing import Callable, Optional, Tuple, Union
import json
import hashlib
import numba
import numpy as np
import umap
import torch
from umap import umap_
from commons.logging import get_logger
from commons.params import NDict, Params, to_ndict

_LOGGER = get_logger(os.path.basename(__file__))


@numba.njit()
def inner_product(X, Y):
    """NUMBA compiled inner product function"""
    assert X.shape == Y.shape
    assert len(X.shape) == 1
    return X.dot(Y), Y  # X.Y and d/dx (IP)


# @numba.njit()
# def r_inner_product(X, Y):
#     """NUMBA compiled reciprocal inner product function"""
#     assert X.shape == Y.shape
#     assert len(X.shape) == 1
#     return 1.0 / X.dot(Y)


@numba.njit()
def n_inner_product(X, Y):
    """NUMBA compiled negative inner product function"""
    assert X.shape == Y.shape
    assert len(X.shape) == 1
    return -(X.dot(Y)), -(Y.copy())


# @numba.njit()
# def cos_theta(x, y):
#     """NUMBA compiled cos-theta i.e., cosine similarity, not cosine distance"""
#     result = 0.0
#     norm_x = 0.0
#     norm_y = 0.0
#     for i in range(x.shape[0]):
#         result += x[i] * y[i]
#         norm_x += x[i] ** 2
#         norm_y += y[i] ** 2

#     if norm_x == 0.0 and norm_y == 0.0:
#         return 0.0
#     elif norm_x == 0.0 or norm_y == 0.0:
#         return 1.0
#     else:
#         return result / np.sqrt(norm_x * norm_y)


@numba.njit()
def n_cos_theta(x, y):
    """NUMBA compiled -cos-theta"""
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0, np.zeros(x.shape)
    else:
        return result / np.sqrt(norm_x * norm_y), -(x * result - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)


# @numba.njit(fastmath=True)
# def cosine_grad(x, y):
#     result = 0.0
#     norm_x = 0.0
#     norm_y = 0.0
#     for i in range(x.shape[0]):
#         result += x[i] * y[i]
#         norm_x += x[i] ** 2
#         norm_y += y[i] ** 2

#     if norm_x == 0.0 and norm_y == 0.0:
#         dist = 0.0
#         grad = np.zeros(x.shape)
#     elif norm_x == 0.0 or norm_y == 0.0:
#         dist = 1.0
#         grad = np.zeros(x.shape)
#     else:
#         grad = -(x * result - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)
#         dist = 1.0 - (result / np.sqrt(norm_x * norm_y))

#     return dist, grad


def _map_metric(metric_name: str) -> Union[Callable, str]:
    if metric_name.lower() == 'ip':
        return inner_product
    # elif metric_name.lower() == 'rip':
    #     params.metric_name = r_inner_product
    elif metric_name.lower() == 'nip':
        return n_inner_product
    # elif metric_name.lower() == 'cos':
    #     params.metric_name = cos_theta
    elif metric_name.lower() == 'ncos':
        return n_cos_theta
    else:
        return metric_name


def _apply_umap_defaults(umap_spec: Params) -> Params:
    """Fill in defaults for missing params."""
    umap_spec.setdefault('metric', 'nip')
    umap_spec.setdefault('out_metric', 'euclidean')
    umap_spec.setdefault('cache', True)
    return umap_spec


def train_umapper(*,
                  training_vectors: torch.Tensor,
                  #   num_embeddings_to_train: Optional[int],  # Only used for naming cache file
                  model_name: str,
                  #   cache: bool = False,
                  cachedir: str = '_cache/viz_umap/',
                  map_key: Optional[str] = None,
                  num_dims: int,
                  config: Params,
                  #   metric: str = 'IP',
                  #   n_neighbors: int,
                  #   out_metric: str = 'euclidean',
                  #   seed: Optional[int] = None,
                  #   dens_map: bool = True,
                  #   dens_lambda: float = 2.0
                  ) -> umap.UMAP:
    """
    Train a umapper on which the transform function can be called later. Cache it to disk for fetching the next time.
    """
    map_key = map_key or f'umap{num_dims}d'
    params = Params(n_neighbors=config.n_neighbors,
                    min_dist=0.,
                    n_components=num_dims,
                    metric=config.metric,
                    output_metric=config.out_metric,
                    densmap=config.dens_map,
                    n_epochs=config.n_epochs,
                    random_state=config.seed,
                    transform_seed=config.seed,
                    )
    if config.dens_map:
        params.update(dict(dens_lambda=config.dens_lambda))
    if config.from_cache_only:
        config.cache = True
    if config.cache:
        file_params = params.copy()
        if config.num_embeddings_to_train is not None:
            file_params.update(num_embeddings=config.num_embeddings_to_train)
        hasher = hashlib.shake_128()
        json_str = json.dumps(file_params)
        # hasher.update(bytes(json_str, encoding='utf-8'))
        hasher.update(training_vectors.cpu().numpy().tobytes())
        digest = hasher.hexdigest(20)  # pylint: disable=too-many-function-args
        cachefile = f'{cachedir}/{model_name}/{map_key}|{json_str}|{digest}.pkl'

    if config.cache and os.path.exists(cachefile):  # type: ignore
        _LOGGER.info(f'Loading umapper from {cachefile}')
        with open(cachefile, 'rb') as f:  # type: ignore
            umapper = pickle.load(f)
    else:
        if config.from_cache_only:
            raise ValueError(f'Cache file {cachefile} not found')
        _LOGGER.info(f'Training UMAP cluster {cachefile}. This will take a while')
        params.metric = _map_metric(config.metric)
        params.output_metric = _map_metric(config.out_metric)
        umapper = umap.UMAP(**params).fit(training_vectors.cpu().numpy())
        if config.cache:
            os.makedirs(cachedir, exist_ok=True)
            with open(cachefile, 'wb') as f:  # type: ignore
                pickle.dump(umapper, f)

    return umapper


def get_points(*,
               vectors: torch.Tensor,
               embedding_vectors: torch.Tensor,
               dims: int = 3,
               preprocess: Params,
               umap_spec: Params,
               #    cache: bool = True,
               #    n_neighbors: int,
               # metric: str = 'IP',
               # out_metric: str = 'euclidean',
               # seed: Optional[int] = None,
               # num_embeddings_to_train: Optional[int],
               # dens_map: bool = True,
               # dens_lambda: float = 2.0,
               ) -> Tuple[torch.Tensor, umap.UMAP]:
    """Return Umapped points for plotting"""
    umap_spec = _apply_umap_defaults(umap_spec)
    num_embeddings_to_train = umap_spec.num_embeddings_to_train
    if umap_spec.dens_map:
        if umap_spec.out_metric.lower() != 'euclidean':
            raise NotImplementedError('Non euclidean out_metric is not supported by UMAP when dens_map=True')

    training_vectors = torch.cat((embedding_vectors if num_embeddings_to_train is None
                                  else embedding_vectors[:num_embeddings_to_train], vectors))
    if preprocess.unit_norm or preprocess['unit-norm']:
        _LOGGER.info(f'Normalizing embeddings for training = {training_vectors.shape}')
        training_vectors = training_vectors / training_vectors.norm(p=2, dim=-1).unsqueeze(-1)

    umapper = train_umapper(training_vectors=training_vectors,
                            # num_embeddings_to_train=num_embeddings_to_train,
                            model_name="t5-small",
                            num_dims=dims,
                            config=umap_spec
                            # metric=umap_spec.metric or 'nip',
                            # n_neighbors=umap_spec.n_neighbors,
                            # out_metric=umap_spec.out_metric or 'euclidean',
                            # cache=umap_spec.cache if umap_spec.cache is not None else True,
                            # seed=umap_spec.seed,
                            # dens_map=umap_spec.dens_map,
                            # dens_lambda=umap_spec.dens_lambda
                            )

    return umapper.embedding_[-len(vectors):], umapper
