"""Plotly code for visualization"""
from typing import Any, Dict, List, Optional, Mapping, Union
import os
import numpy as np
import torch
from dash import dcc
from dash.development.base_component import Component
import plotly.graph_objects as go
from commons.params import NDict, Params, to_ndict
from commons.logging import get_logger
from viz.viz_umap import get_points

_LOGGER = get_logger(os.path.basename(__file__))


def _strip(s: str) -> str:
    return s.lstrip(chr(9601))


def _ids_to_tokens(ids: List[int], tokenizer: Any) -> List[str]:
    return [_strip(token) for token in tokenizer.convert_ids_to_tokens(ids)]


def get_knn(vectors: torch.Tensor, data_dict: NDict, K: int) -> np.ndarray:
    """Return K nearest neighbors"""
    inp_index = data_dict['inp_index']
    _, _nn = inp_index.search(vectors.contiguous().numpy(), K)
    # [_strip(token) for token in data_dict['tokenizer'].convert_ids_to_tokens(_nn[:, k])]
    nn = np.asarray([_ids_to_tokens(_nn[:, k], data_dict['tokenizer']) for k in range(K)])  # (K, seq-len)
    nn_hovertext = ['KNNs<br>' + '<br>'.join(nn[:, s]) for s in range(nn.shape[1])]
    assert len(nn_hovertext) == len(vectors)
    return nn_hovertext  # (#vecs,)


def _preprocess(vec: torch.Tensor, options: Optional[Dict]) -> torch.Tensor:
    if options is not None and (options['unit_norm'] or options['unit-norm']):
        # assert len(vec.shape) == 1
        _LOGGER.info(f'Vec-norm: vec.shape = {vec.shape}')
        vec = vec / vec.norm(p=2, dim=-1).unsqueeze(-1)
    return vec


def plot_embeddings(spec: List, data_dict: NDict) -> List[Component]:
    """Plot the requested embedding charts.
    The spec looks like this:
    [
        {
            "stack": "encoder",
            "pos": 0,
            "support_set": {"stack": "encoder", "layer": 0, "tensor_key": "E"},
            "path": [
                {"layer": 0, "tensor_key": "NormSA"}
            ]
        }
    ]
    """
    graphs: List[dcc.Graph] = []
    spec = to_ndict(spec)
    color_options = dict(backgroundcolor='white', color='lightgray',
                         showaxeslabels=True,
                         showline=True, linecolor='black', linewidth=0.5,
                         showticklabels=True,
                         showgrid=False, gridcolor='gray', gridwidth=0.5,
                         title='')
    for chart_id, chart_spec in enumerate(spec):
        chart_spec = Params(chart_spec)
        chart_spec.setdefault('support_set.color', 'lightgray')
        chart_spec.setdefault('support_set.size', 2)
        fig = go.Figure(layout={'width': chart_spec['plot_size.width'] or 1024,
                                'height': chart_spec['plot_size.height'] or 1024,
                                'autosize': False,
                                'coloraxis': {'colorscale': 'cividis'},
                                'scene': dict(xaxis=color_options,
                                              yaxis=color_options,
                                              zaxis=color_options
                                              )
                                })
        chart_prep = Params(chart_spec.preprocess)
        ss_spec = NDict(chart_spec.support_set)
        unique_input_ids = list(set(data_dict.datum.input_ids.tolist()))
        support_set = data_dict.embedding_vectors[unique_input_ids]  # (#num-unique-ids, d_model)
        support_set = _preprocess(support_set, ss_spec.preprocess)
        support_set_hovertext = [
            'Inp<br>' + _strip(token) for i, token in
            enumerate(data_dict['tokenizer'].convert_ids_to_tokens(unique_input_ids))]

        chart_point_names: List[str] = []
        chart_point_vecs: List[torch.Tensor] = []
        paths: List[NDict] = []
        for paths_spec_id, paths_spec in enumerate(chart_spec.paths):
            stack = paths_spec.stack
            T_l: List[Mapping[str, torch.Tensor]] = data_dict[f'{stack}_T_l']
            for pos in paths_spec.positions:
                # Define a path per input-position per path-spec
                path_name: str = f'Pos_{pos}:Chart_{chart_id}:PathSpec_{paths_spec_id}:{"Enc" if stack == "encoder" else "Dec"}'
                _start_pos = len(chart_point_vecs)
                # Gather points of the path
                for layer in paths_spec.layers:
                    for point in paths_spec.points:
                        if point.tensor_key in T_l[layer]:
                            # _prep = Params(chart_prep).updated(point.preprocess)
                            _point_vec = _preprocess(T_l[layer][point.tensor_key][0, :, pos], point.preprocess)
                            chart_point_names.append(f'L_{layer}:T_{point.tensor_key}:{path_name}')
                            chart_point_vecs.append(_point_vec)
                paths.append(NDict(
                    chart_id=chart_id,
                    path_name=path_name,
                    path_spec_id=paths_spec_id,
                    input_pos=pos,
                    slice=slice(_start_pos, len(chart_point_vecs))
                    ))

        chart_point_vecs = torch.stack(chart_point_vecs)  # type: ignore
        chart_points_hovertext = [f'{chart_point_names[i]}<br>' + knn
                                  for i, knn in enumerate(get_knn(chart_point_vecs, data_dict, 4))]  # type: ignore

        vecs_to_plot = torch.cat([support_set, chart_point_vecs])  # type: ignore
        num_dims = chart_spec.umap.num_dims or 3
        points, umapper = get_points(vectors=vecs_to_plot,
                                     embedding_vectors=data_dict.embedding_vectors,
                                     dims=num_dims,
                                     preprocess=chart_prep,
                                     umap_spec=chart_spec.umap
                                     )
        assert points.shape[1] == num_dims
        support_set = points[:len(support_set)]
        chart_points = points[len(support_set):]

        # Retrieve the embedding matrix slice to show
        if chart_spec.umap.num_embeddings_to_plot > 0:
            if chart_spec.umap.num_embeddings_to_train is not None:
                num_embeddings_to_plot = min(
                    chart_spec.umap.num_embeddings_to_plot,
                    chart_spec.umap.num_embeddings_to_train)
            else:
                num_embeddings_to_plot = chart_spec.umap.num_embeddings_to_plot
            plot_emb = umapper.embedding_[:num_embeddings_to_plot]
            emb_hovertext = _ids_to_tokens(list(range(num_embeddings_to_plot)), data_dict['tokenizer'])

        if num_dims == 3:
            if chart_spec.umap.num_embeddings_to_plot > 0:
                fig.add_trace(go.Scatter3d(
                    name='embeddings',
                    x=plot_emb[:, 0], y=plot_emb[:, 1], z=plot_emb[:, 2],
                    mode='markers',
                    marker={'color': chart_spec.support_set.color, 'size': chart_spec.support_set.size},
                    hovertext=emb_hovertext,
                    hoverinfo='text'
                ))
            fig.add_trace(go.Scatter3d(
                name='input tokens',
                x=support_set[:, 0], y=support_set[:, 1], z=support_set[:, 2],
                mode='markers',
                marker={'color': chart_spec.support_set.color, 'size': chart_spec.support_set.size},
                hovertext=support_set_hovertext,
                hoverinfo='text'
            ))
        elif num_dims == 2:
            if chart_spec.umap.num_embeddings_to_plot > 0:
                fig.add_trace(go.Scatter(
                    name='embeddings',
                    x=plot_emb[:, 0], y=plot_emb[:, 1],
                    mode='markers',
                    marker={'color': chart_spec.support_set.color, 'size': chart_spec.support_set.size},
                    hovertext=emb_hovertext,
                    hoverinfo='text'
                ))
            fig.add_trace(go.Scatter(
                name='input tokens',
                x=support_set[:, 0], y=support_set[:, 1],
                mode='markers',
                marker={'color': chart_spec.support_set.color, 'size': chart_spec.support_set.size},
                hovertext=support_set_hovertext,
                hoverinfo='text'
            ))

        for path in paths:
            _path_points = chart_points[path.slice]
            _hovertext = chart_points_hovertext[path.slice]
            # _hovertext = [f'{_hovertext[i]}' for i in range(len(_path_points))]
            if np.isnan(_path_points).any():
                _LOGGER.warning('path {path.path_name} has nans')
            if num_dims == 3:
                fig.add_trace(go.Scatter3d(
                    name=path.path_name,
                    x=_path_points[:, 0],
                    y=_path_points[:, 1],
                    z=_path_points[:, 2],
                    mode='lines+markers',
                    marker=dict(size=2),
                    # marker={'color': list(range(len(_path_points))), 'size': 4,
                    #         'colorscale': 'Blues', 'showscale': True},
                    line=dict(width=1),
                    hovertext=_hovertext,
                    hoverinfo='text'
                ))
            elif num_dims == 2:
                fig.add_trace(go.Scatter(
                    name=path.path_name,
                    x=_path_points[:, 0], y=_path_points[:, 1],
                    mode='lines+markers',
                    marker={'size': 2},  # 'color': list(range(len(_path_points))), 'colorscale': 'Blues',
                    line=dict(width=1),
                    hovertext=_hovertext,
                    hoverinfo='text'
                ))

        fig.write_image('./plotly_fig.png')
        graphs.append(dcc.Graph(figure=fig))

    return graphs
