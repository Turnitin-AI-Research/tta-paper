"""Clustering and Similarity Search"""
from typing import Any
import torch
import faiss
# import faiss.contrib.torch_utils  # Do not use this: Does not work reliably and causes OverflowError.


def knn_index(embedding_vectors: torch.Tensor, distance_measure: str) -> Any:
    """
    Return index for looking up nearest token vectors

    Arguments:
        token_embedding_matrix: shape=(num_tokens, embedding_dimension)
        device: Optional device to build/run index on,
        distance_measure: 'L2' or 'IP' (inner / dot product)
    """
    assert len(embedding_vectors.shape) == 2
    # if embedding_vectors.device.type == 'cuda':
    #     gpu_res = faiss.StandardGpuResources()
    #     flat_config = faiss.GpuIndexFlatConfig()
    #     flat_config.device = embedding_vectors.device.index
    #     index = faiss.GpuIndexFlatL2(gpu_res, embedding_vectors.shape[1], flat_config)
    # else:
    if distance_measure == 'L2':
        index = faiss.IndexFlatL2(embedding_vectors.shape[1])
    else:
        index = faiss.IndexFlatIP(embedding_vectors.shape[1])

    index.add(embedding_vectors.cpu().detach().numpy())
    return index
