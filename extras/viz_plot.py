"""Plotly code for visualization"""
from typing import Any, Dict, List, Optional, Mapping, Tuple, Union
import os
import math
import numpy as np
import torch
from dash import dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from commons.params import NDict, Params
from commons.logging import get_logger
from viz.viz_spec import PLOT_MARGINS, empty_graph, TensorSpec
from viz.viz_commons import preprocess

_LOGGER = get_logger(os.path.basename(__file__))


def roundup(x: float) -> int:
    """Round up to the nearest integer away from zero"""
    return int(math.ceil(abs(x)) * np.sign(x))


def compute_plot1_size(*,
                       W: torch.Tensor,
                       T_name: str,
                       vscale: float,
                       hscale: float,
                       stack_heads: bool,
                       show_tokens: bool,
                       show_y_tokens: bool,
                       show_magnitude: bool,
                       knn: bool,
                       k: int,
                       tensor_options: Params,
                       data_dict: NDict,
                       do_preprocess: bool = True) -> NDict:
    """Compute size of plotly heatmap produced by _W_plot1"""
    num_heads = W.shape[0]
    if do_preprocess:
        W = [preprocess(W=W[head], tensor_options=tensor_options.checklist,
             data_dict=data_dict) for head in range(W.shape[0])]
        W0 = W[0]
    else:
        W0 = preprocess(W=W[0], tensor_options=tensor_options.checklist, data_dict=data_dict)
    W_shape1 = W0.shape[0]
    W_shape2 = W0.shape[1]
    horizontal_spacing = 0.002
    vertical_spacing = 0.005
    hist_lspace_px = 32  # pixels
    hist_lspace = None
    plot_margin = Params(PLOT_MARGINS)
    subplot_specs: List[List]
    if show_tokens:
        plot_margin['b'] = plot_margin['b'] + 50
    if show_y_tokens:
        plot_margin['l'] = plot_margin['l'] + 50
    if knn:
        plot_margin.b += (20 * k)  # pylint: disable=no-member
    if not stack_heads:
        # vertical_spacing = 0.
        W_height = W_shape1 * vscale
        plot_height = W_height + (64 if show_magnitude else 0)
        col_width = W_shape2 * hscale
        W_width = col_width * num_heads
        hist_width = 64 + hist_lspace_px
        num_rows = 1 + (1 if show_magnitude else 0)
        num_cols = num_heads + 1
        row_dist = None if not show_magnitude else [W_height, 64]
        plot_width = col_width * num_heads + hist_width
        col_dist = [col_width] * num_heads + [hist_width]
        plot_width = math.ceil(plot_width * (1.0 + (horizontal_spacing * (num_cols - 1))))
        plot_height = math.ceil(plot_height * (1.0 + (vertical_spacing * (num_rows - 1))))
        plot_width += plot_margin['l'] + plot_margin['r']
        plot_height += (plot_margin['t'] + plot_margin['b'])
        hist_lspace = (hist_lspace_px / plot_width)
        subplot_specs = [[{}] * num_heads + [{'l': hist_lspace}]]  # type: ignore
        if show_magnitude:
            subplot_specs += [[{}] * num_heads + [None]]  # type: ignore
    else:
        horizontal_spacing = 0.040
        plot_height = W_shape1 * num_heads * vscale
        W_width = W_shape2 * hscale
        hist_width = 64  # max(64, roundup(W_width / 3.))  # Min 5% col-width is required for plotly to work
        plot_width = (W_width + hist_width)
        num_rows = num_heads
        row_dist = None
        num_cols = 2
        col_dist = [W_width, hist_width]
        plot_width = math.ceil(plot_width * (1.0 + (horizontal_spacing * (num_cols - 1))))
        plot_height = math.ceil(plot_height * (1.0 + (vertical_spacing * (num_rows - 1))))
        plot_width += (plot_margin['l'] + plot_margin['r'])
        plot_height += (plot_margin['t'] + plot_margin['b'])
        col_width = None
        subplot_specs = [[{}, {'rowspan': num_rows}] if row == 0 else [{}, None] for row in range(num_rows)]

    hist_height = plot_height

    ret_dict = NDict(dict(
        num_rows=num_rows, num_cols=num_cols,
        row_dist=row_dist, col_dist=col_dist,
        vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing,
        num_heads=num_heads,
        plot_margin=plot_margin,
        plot_height=plot_height, plot_width=plot_width,
        hist_width=hist_width, hist_height=hist_height,
        col_width=col_width,
        hscale=hscale,
        vscale=vscale,
        hist_lspace=hist_lspace,
        subplot_specs=subplot_specs
    ))
    if do_preprocess:
        ret_dict.update(W=W)
    return ret_dict


def _hovertext(*,
               E: torch.Tensor,
               data_dict: Mapping,
               input_tokens: List[str],
               show_tokens: bool,
               y_tokens: List[str] = None,
               K: int = 0,
               plot_type: str) -> Tuple[List[str], np.ndarray, Optional[List[str]]]:
    """
    Return list of token strings to be displayed on x-axis

    Parameters
        E: Tensor of embeddings. Shape = (embedding vector size i.e. d_model, num_embeddings)
        data_dict: Global data-dict which should have the embedding matrix index (inp_index)
        input_tokens: Optional: List of input token strings to be displayed, one per row of E
        K: int: Number of nearest neighbors of the embedding vectors to display. K per row of E.
    """
    def _strip(s: str) -> str:
        return s.lstrip(chr(9601))

    ticktext_x, hovertext_x, hovertext_y, ticktext_y, hovertext = [], None, None, y_tokens, None
    if K > 0:
        inp_index = data_dict['inp_index']
        if E.shape[0] != inp_index.d:
            _LOGGER.warning(f'Cannot show KNNs because E.shape[0]({E.shape[0]}) != inp_index.d({inp_index.d})')
            K = 0
        else:
            _, _nn = inp_index.search(E.transpose(0, 1).contiguous().numpy(), K)
            nn = np.asarray([[_strip(token) for token in data_dict['tokenizer'].convert_ids_to_tokens(_nn[:, k])]
                             for k in range(K)])  # (K, seq-len)
            nn_hovertext = ['KNNs<br>' + '<br>'.join(nn[:, s]) for s in range(nn.shape[1])]
            nn_ticktext = ['\u2666'.join(nn[:, s]) for s in range(nn.shape[1])]

    if show_tokens and (len(input_tokens) != E.shape[1]):
        _LOGGER.warning(
            f'Cannot show tokens because length of input_tokens ({len(input_tokens)}) '
            f'does not match # of vectors ({E.shape[1]})')
        show_tokens = False

    if y_tokens is not None and (len(y_tokens) != E.shape[0]):
        _LOGGER.warning(f'Cannot show y-aixs tokens because len(y_tokens) {len(y_tokens)} != E.shape[0] {E.shape[0]}')
        y_tokens = None

    if show_tokens:
        if K <= 0:
            ticktext_x = input_tokens
            hovertext_x = [f'x={token}<br>' for i, token in enumerate(input_tokens)]
        else:
            ticktext_x = [f'{token}\u2666{nn_ticktext[s]}' for s, token in enumerate(input_tokens)]
            hovertext_x = [f'x={token}<br>{nn_hovertext[s]}<br>' for s,
                           token in enumerate(input_tokens)]
    elif K > 0:
        assert nn.shape[1] == E.shape[1], f'{nn.shape[1]} != {E.shape[1]}'
        ticktext_x = nn_ticktext
        if len(input_tokens) != nn.shape[1]:
            hovertext_x = [f'x={nn_hovertext[s]}<br>' for s in range(nn.shape[1])]
        else:
            # Display tokens in hover-text regardless of value of show_tokens flag
            hovertext_x = [f'x={token}<br>{nn_hovertext[s]}<br>' for s,
                           token in enumerate(input_tokens)]

    if y_tokens is not None:
        ticktext_y = y_tokens
        hovertext_y = np.char.array([f'y={token}<br>' for j, token in enumerate(y_tokens)])

    if hovertext_x is not None:
        assert len(hovertext_x) == E.shape[1], f'{len(hovertext_x)} != {E.shape[1]}'
        hovertext_x = np.expand_dims(np.char.array(hovertext_x), 0)
        if hovertext_y is None:
            hovertext = np.broadcast_to(hovertext_x, E.shape)
        else:
            hovertext = hovertext_x + np.expand_dims(hovertext_y, 1)
    elif hovertext_y is not None:
        hovertext = np.broadcast_to(hovertext_y, E.shape)
    col = np.expand_dims(np.char.asarray([f'{x})<br>' for x in range(E.shape[1])]), 0)
    row = np.expand_dims(np.char.asarray([f'({y}, ' for y in range(E.shape[0])]), 1)
    return ticktext_x, (row + col + hovertext) if hovertext is not None else (row + col), ticktext_y


def plot1(*,
          W: torch.Tensor,
          W_name: str,
          T_spec: TensorSpec,
          colorscale: str = 'cividis_r',
          cmin: Optional[float] = None,
          cmax: Optional[float] = None,
          cmid: Optional[float] = None,
          hscale: float = 1, vscale: float = 1,
          stack_heads: bool = True,
          transposed: bool = False,
          input_tokens: List[str],
          show_tokens: bool = False,
          show_y_tokens: bool = False,
          show_magnitude: bool = False,
          knn: bool,
          k: int,
          data_dict: NDict,
          global_options: List[str] = [],
          tensor_options: Params) -> go.Figure:
    """Create a plotly heatmap of the given tensor W"""
    def head_name(i: int) -> str:
        return f'$H^{{({i})}}$'

    dot_product: bool = 'dot-product' in tensor_options.checklist
    ps = compute_plot1_size(W=W, T_name=W_name, vscale=vscale, hscale=hscale, stack_heads=stack_heads,
                            show_tokens=show_tokens, show_y_tokens=show_y_tokens,
                            show_magnitude=show_magnitude, knn=knn, k=k,
                            tensor_options=tensor_options, data_dict=data_dict)
    if transposed:
        xaxis_title = T_spec.yaxis_title
        yaxis_title = T_spec.xaxis_title if not dot_product else xaxis_title
    else:
        xaxis_title = T_spec.xaxis_title
        yaxis_title = T_spec.yaxis_title if not dot_product else xaxis_title

    fig = make_subplots(rows=ps.num_rows,
                        row_heights=ps.row_dist,
                        cols=ps.num_cols,
                        column_widths=ps.col_dist,
                        shared_yaxes=False,
                        shared_xaxes='columns',  # if stack_heads else False,
                        horizontal_spacing=ps.horizontal_spacing,
                        vertical_spacing=ps.vertical_spacing,
                        column_titles=[head_name(i) for i in range(ps.num_heads)] if not stack_heads else None,
                        row_titles=['Hist'],
                        y_title=f'${T_spec["latexName"]}\\text{{: {yaxis_title} }}$',
                        specs=ps.subplot_specs,
                        x_title=xaxis_title)

    _W: List[torch.Tensor] = ps.W
    plot_type = tensor_options.plot_type
    # if plot_type == 'bar' and _W[0].shape[0] != 1:
    #     _LOGGER.info(f'Ignoring plot-type ({plot_type}) request. Shape of tensor is {_W[0].shape}')
    #     plot_type = 'heatmap'

    if not stack_heads:
        fig.update_yaxes(showticklabels=False)  # hide all the yticks
        fig.update_yaxes(showticklabels=True, col=1)  # show yticks of the first column
        fig.update_yaxes(showticklabels=True, row=1, col=ps.num_cols)  # histogram in last col
        # if ps.num_rows > 1:
        #     # Remove xticks of all but last row
        #     fig.update_xaxes(showticklabels=False)
        #     fig.update_xaxes(showticklabels=True, row=ps.num_rows)
        for j in range(ps.num_heads):
            # _W[j] = preprocess(W=W[j], global_options=global_options, tensor_checklist=tensor_options.checklist,
            #                    data_dict=data_dict)
            if plot_type == 'bar':
                if _W[j].shape[0] == 1:
                    _V = _W[j][0].cpu()
                    fig.add_trace(go.Bar(x=np.arange(len(_V)) - ((len(_V) + 1) // 2) if W_name.lower() == 'pos' else None,  # np.arange(len(_V)),
                                         y=_V,
                                         marker=dict(coloraxis='coloraxis', showscale=False, color=_V),
                                         name='',
                                         showlegend=False,
                                         ),
                                  row=1, col=j+1)
                else:
                    _V = _W[j][:, 0].cpu()
                    fig.add_trace(go.Bar(# y=np.arange(len(_V)),  # np.arange(len(_V)) - ((len(_V) + 1) // 2),
                                         x=_V,
                                         marker=dict(coloraxis='coloraxis', showscale=False, color=_V),
                                         name='',
                                         showlegend=False,
                                         orientation='h'
                                         ),
                                  row=1, col=j+1)
            elif plot_type == 'surface':
                fig.add_trace(go.Surface(z=_W[j].cpu(),
                                         coloraxis='coloraxis',
                                         showscale=False,
                                         cauto=True,
                                         name='',
                                         ),
                              row=1, col=j+1)
            else:
                fig.add_trace(go.Heatmap(z=_W[j].cpu(),
                                         coloraxis='coloraxis',
                                         showscale=False,
                                         zauto=True,
                                         name='',
                                         zsmooth=False),
                              row=1, col=j+1)
            if show_magnitude:
                _V = _W[j].norm(p=2, dim=0).cpu()
                fig.add_trace(
                    go.Bar(
                        # x=np.arange(len(_V)),
                        y=_V,
                        marker=dict(coloraxis='coloraxis3', showscale=False, color=_V),
                        name='',
                        showlegend=False),
                    row=2, col=j+1)
            if plot_type == 'heatmap':
                fig.update_yaxes(range=[_W[j].shape[0]-1, 0], row=1, col=j+1)  # reverse y-axes frame
            if plot_type in ['heatmap', 'bar']:
                if (show_tokens and not transposed) or knn:
                    ticktext, hovertext, ticktext_y = _hovertext(E=_W[j],
                                                                 data_dict=data_dict,
                                                                 input_tokens=input_tokens,
                                                                 show_tokens=show_tokens,
                                                                 y_tokens=input_tokens if show_y_tokens else None,
                                                                 K=k if knn else 0,
                                                                 plot_type=plot_type)
                    fig.update_xaxes(dict(tickvals=list(range(_W[j].shape[1])),
                                          ticktext=ticktext,
                                          ticks='', tickmode='array',
                                          tickangle=90
                                          ), row=ps.num_rows, col=j+1)
                    if show_y_tokens and (ticktext_y is not None):
                        fig.update_yaxes(dict(tickvals=list(range(_W[j].shape[0])),
                                              ticktext=ticktext_y,
                                              ticks='', tickmode='array',
                                              tickangle=0
                                              ), row=1, col=j+1)
                    if plot_type == 'heatmap':
                        fig.update_traces(dict(customdata=hovertext,
                                               hovertemplate='%{customdata}<br>z=%{z:.4f}'),
                                          row=1, col=j+1)
                    elif plot_type == 'bar':
                        fig.update_traces(dict(customdata=hovertext.squeeze(),
                                               hovertemplate='%{customdata}<br>y=%{y:.4f}'),
                                          row=1, col=j+1)

        fig.add_trace(go.Histogram(
            y=torch.stack(_W).flatten().cpu().numpy(),
            histnorm='probability',
            showlegend=False),
            row=1, col=ps.num_cols)

    else:
        for i in range(ps.num_heads):
            # _W[i] = preprocess(W=W[i], global_options=global_options, tensor_checklist=tensor_options.checklist,
            #                    data_dict=data_dict)
            row = ps.num_heads - i
            if plot_type == 'bar':
                if _W[i].shape[0] == 1:
                    _V = _W[i][0].cpu()
                    fig.add_trace(go.Bar(x=np.arange(len(_V)) - ((len(_V) + 1) // 2) if W_name.lower() == 'pos' else None,
                                         y=_V,
                                         marker=dict(coloraxis='coloraxis', color=_V, showscale=False),
                                         name='',
                                         showlegend=False,
                                         ),
                                  row=row, col=1)
                else:
                    assert _W[i].shape[1] == 1
                    _V = _W[i][:, 0].cpu()
                    fig.add_trace(go.Bar(# y=np.arange(len(_V)) - ((len(_V) + 1) // 2),
                                         x=_V,
                                         marker=dict(coloraxis='coloraxis', color=_V, showscale=False),
                                         name='',
                                         showlegend=False,
                                         orientation='h'
                                         ),
                                  row=row, col=1)
            else:
                fig.add_trace(go.Heatmap(z=_W[i].cpu(),
                                         coloraxis='coloraxis',
                                         showscale=False,
                                         zauto=True,
                                         name='',  # f'{W.mean():.4f}, {W.var():.4f}',
                                         zsmooth=False),
                              row=row, col=1)
            if plot_type == 'heatmap':
                fig.update_yaxes(title_text=head_name(i), range=[_W[i].shape[0]-1, 0], row=row, col=1)

        if plot_type in ['heatmap']:
            if (show_tokens and not transposed) or knn:
                ticktext, hovertext, ticktext_y = _hovertext(E=W[i],
                                                             data_dict=data_dict,
                                                             input_tokens=input_tokens,
                                                             show_tokens=show_tokens,
                                                             y_tokens=input_tokens if show_y_tokens else None,
                                                             K=k if knn else 0,
                                                             plot_type=plot_type)
                fig.update_xaxes(dict(
                    tickvals=list(range(W[i].shape[1])),
                    ticktext=ticktext,
                    ticks='', tickmode='array',
                    tickangle=90
                ), row=(ps.num_heads), col=1)
                if show_y_tokens and (ticktext_y is not None):
                    fig.update_yaxes(dict(
                        tickvals=list(range(W[i].shape[0])),
                        ticktext=ticktext_y,
                        ticks='', tickmode='array',
                        tickangle=0
                    ))  # , row=(ps.num_heads), col=1)

                if plot_type == 'heatmap':
                    fig.update_traces(dict(customdata=hovertext,
                                           hovertemplate='%{customdata}<br>z=%{z:.4f}'))  # ,
                elif plot_type == 'bar':
                    fig.update_traces(dict(customdata=hovertext.squeeze(),
                                           hovertemplate='%{customdata}<br>y=%{y:.4f}'))  # ,
                    #   row=(ps.num_heads), col=1)

        fig.add_trace(go.Histogram(y=torch.stack(_W).flatten().cpu().numpy(),
                                   histnorm='probability',
                                   showlegend=False),
                      row=1, col=2)

    colorbar_len = 1.0
    fig.update_layout({'coloraxis': {'colorscale': colorscale,
                                     'showscale': True,
                                     'cmin': cmin, 'cmax': cmax, 'cmid': cmid,
                                     'colorbar': {'lenmode': 'fraction', 'len': colorbar_len,
                                                  'yanchor': 'top', 'y': 1.0,
                                                  }
                                     },
                       'coloraxis2': dict(showscale=False),
                       'coloraxis3': dict(showscale=False, colorscale='gray')
                       },
                      )

    fig.update_layout(  # title_text=title,
        autosize=False,
        width=ps.plot_width,
        height=ps.plot_height,
        margin=ps.plot_margin,
        plot_bgcolor='white',
        xaxis=dict(automargin=True),  # will grow margin if needed
        yaxis=dict(automargin=True)
        # transition={'duration': 500}
    )
    _LOGGER.debug(
        f'plot1: {W_name}: num_heads={ps.num_heads}, num_rows={ps.num_rows}, num_cols={ps.num_cols}, '
        f'col_width={ps.col_width}, hist_width={ps.hist_width}, col_dist={ps.col_dist}, '
        f'plot height={ps.plot_height}, plot_width={ps.plot_width}'
    )
    return fig


def plot2(*,
          W: torch.Tensor,
          head: int,
          x: int, y: int,
          colorscale: Optional[str] = None,
          cmin: Optional[float] = None, axis_min: Optional[float] = None,
          cmax: Optional[float] = None, axis_max: Optional[float] = None,
          cmid: Optional[float] = None,
          iD: str,
          W_name: str,
          T_spec: TensorSpec,
          visible: str = 'vec',  # 'vec' or 'head'. Set visible=True else 'legendonly'
          plot1_size: NDict,
          plot_col: bool,
          global_options: List[str],
          tensor_options: Params,
          data_dict: NDict
          ) -> dcc.Graph:
    """Make a Wv row in html"""
    if W is not None:
        ps = plot1_size
        ps.horizontal_spacing = 0.2
        ps.col_dist = [0.67, 0.33]
        ps.plot_margin['r'] = 200

        W_head = preprocess(W=W[head], global_options=global_options, tensor_options=tensor_options.checklist,
                            data_dict=data_dict)
        if plot_col:
            V = W_head[:, int(x)].cpu()
        else:
            V = W_head[int(y), :].cpu()
        V_abs = V.abs()
        W_head_mean = W_head.mean(dim=1).cpu() if plot_col else W_head.mean(dim=0).cpu()
        W_head_mean_abs = W_head.abs().mean(dim=1).cpu() if plot_col else W_head.abs().mean(dim=0).cpu()
        fig = make_subplots(rows=1,
                            cols=2,
                            column_widths=ps.col_dist,
                            shared_yaxes=False,
                            shared_xaxes=False,
                            horizontal_spacing=ps.horizontal_spacing,
                            # column_titles=[f'$C_{{ {x} }}$', f'$H^{{ ({head}) }}$ PDF', ]
                            )
        col_or_row_label = "Col" if plot_col else "Row"
        fig.add_trace(
            go.Bar(
                x=V,
                name=f'{col_or_row_label} Vec',
                orientation='h',
                marker=dict(coloraxis='coloraxis2', color=V, showscale=False),
                visible=True if visible == 'vec' else 'legendonly',
                # legendgroup='Vector',
            ),
            row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=V_abs,
                name=f'{col_or_row_label} Abs',
                orientation='h',
                marker=dict(coloraxis='coloraxis2', color=V, showscale=False),
                visible='legendonly',
                # legendgroup='Vector'
            ),
            row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=W_head_mean,
                name='Head Mean',
                orientation='h',
                marker=dict(coloraxis='coloraxis2', color=V, showscale=False),
                visible='legendonly',
                # legendgroup='Vector'
            ),
            row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=W_head_mean_abs,
                name='Head Abs',
                orientation='h',
                marker=dict(coloraxis='coloraxis2', color=V, showscale=False),
                visible=True if visible == 'head' else 'legendonly',
                # legendgroup='Vector'
            ),
            row=1, col=1)
        fig.add_trace(go.Histogram(y=W_head.flatten().cpu().numpy(),
                                   histnorm='probability',
                                   name='Head',
                                   opacity=0.5,
                                   visible=True if visible == 'head' else 'legendonly',
                                   #    legendgroup='Dist',
                                   #    legendgrouptitle={'text': 'Dist'}
                                   ),
                      row=1, col=2)
        fig.add_trace(go.Histogram(y=V.flatten().cpu().numpy(), histnorm='probability',
                                   name='Vec',
                                   opacity=0.5,
                                   visible=True if visible == 'vec' else 'legendonly',
                                   #    legendgroup='Dist'
                                   ),
                      row=1, col=2)
        fig.update_layout(barmode='overlay')

        plot_width = roundup(ps.hist_width * 3 * (1.0 + ps.horizontal_spacing)) + \
            (ps.plot_margin['l'] + ps.plot_margin['r'])
        plot_height = roundup(ps.plot_height)
        plot_margin = ps.plot_margin
        _LOGGER.debug(
            f'plot2: {W_name}: width={plot_width}, height={plot_height}, margin={plot_margin}, '
            f'cmin={cmin}, cmax={cmax}, cmid={cmid}')
        fig.update_layout(
            title={
                'text': (f'${T_spec["latexName"]}, \\; H^{{ ({head}) }}, \\; '
                         f'{"C" if plot_col else "R"}_{{ {x if plot_col else y} }}$'),
                'x': 0.5, 'xanchor': 'center'
            },
            autosize=False,
            width=plot_width,
            height=plot_height,
            margin=plot_margin,
            xaxis=dict(automargin=True),  # will grow margin if needed
            plot_bgcolor='white',
            # transition={'duration': 500}
            coloraxis2={'colorscale': colorscale,
                        'showscale': False,
                        'cmin': cmin, 'cmax': cmax, 'cmid': cmid}
        )
        if axis_min is not None:
            fig.update_xaxes(range=[axis_min, axis_max], row=1, col=1)
        fig.update_yaxes(  # zeroline=True, zerolinecolor='black',
            range=[len(V) - 1, 0],  # reverse y-axes frame
            row=1, col=1)
        return dcc.Graph(id=iD,
                         figure=fig,
                         responsive=False,
                         config={'autosizable': False, 'displayModeBar': 'hover', 'showAxisDragHandles': False})
    else:
        return empty_graph(id=iD)
