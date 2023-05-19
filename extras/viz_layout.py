"""Dash based vizualization app"""
from typing import Dict, Union, List, MutableMapping, Any, Optional, Sequence, Tuple, Mapping
import math
import os
import json
import numpy as np
import ruamel.yaml
from ruamel.yaml import YAML
import dash
from dash import dcc
from dash import html
from dash.development.base_component import Component
from dash.dependencies import Input, Output, State
import plotly.express as px
import torch
from commons.params import Params, NDict
from commons.logging import get_logger
from viz.viz_plot import plot1, plot2, compute_plot1_size
from viz.viz_spec import ENCODER_LAYER_TENSOR_SPEC, ENCODER_TENSOR_TYPES
from viz.viz_spec import DECODER_LAYER_TENSOR_SPEC, DECODER_TENSOR_TYPES
from viz.viz_spec import empty_graph, TensorSpec, DEFAULT_EMBEDDING_VIZ_SPEC
from viz.viz_commons import preprocess
from viz.viz_plot_embeddings import plot_embeddings

_LOGGER = get_logger(os.path.basename(__file__))


def _config_bar_layout(config, data_dict: NDict, app: dash.Dash) -> List:
    """Return list of elements comprisign the config-bar."""
    config_display = html.Div(
        className='col-6 border-end border-1',
        children=[
            html.Img(src=app.get_asset_url('endec-rotated.svg'), className='pb-2 img-fluid'),
            html.Dl(className='row border-top py-3',
                    children=[
                        html.Dt('Checkpoint', className='col-sm-2'),
                        html.Dd(config['checkpoint_file'], className='text-wrap col-sm-10'),
                        html.Dt('Config', className='col-sm-2'),
                        html.Dd(config['config_file'], className='text-wrap col-sm-10'),
                        html.Dt('Datum ID', className='col-sm-2'),
                        html.Dd(config['datum_id'], className='text-wrap col-sm-10'),
                        html.Dt('Label', className='col-sm-2'),
                        html.Dd(data_dict.datum.y, className='text-wrap col-sm-10'),
                        html.Dt('Prediction', className='col-sm-2'),
                        html.Dd(f'{data_dict.datum.pred}, score={data_dict.datum.pred_score:.4f}',
                                className='text-wrap col-sm-10'),
                        html.Dt('Input', className='col-sm-2'),
                        html.Dd(f'#chars={len(data_dict.datum.x)}, #tokens={len(data_dict.datum.input_ids)}',
                                className='text-wrap col-sm-10')
                    ])
        ]
    )
    input_text = html.Div(
        className='col-6',
        children=[
            dcc.Textarea(
                id='input_text',
                contentEditable=False,
                readOnly=True,
                wrap='soft',
                # cols=120,
                # rows=20,
                value=data_dict['datum']['x'],
                style={'width': '100%', 'height': '100%'}
            ),
            # html.Img(className='img-fluid', src=app.get_asset_url('details.svg'))
        ]
    )
    config_bar_row = html.Div(
        id='config-bar',
        className='row border border-1 p-4 rounded',
        children=[
            config_display,
            input_text
        ])
    return [config_bar_row]


def _tensor_controlbar_layout(num_enc_layers: int, app: dash.Dash) -> List:
    """
    Make global control-bar layout rows
    """
    colorscales = px.colors.named_colorscales()
    colorscales = sorted(colorscales + [c + '_r' for c in colorscales])

    def tensor_selector(name: str, SPEC: Any, TYPES: Any) -> html.Div:
        """Make Global Tensors Selector"""
        return html.Div(
            id=f'{name}-tensor-selector-grouper',
            # className=' col-md-3 col-lg-2 border-top border-bottom border-1',
            className='border-bottom border-top py-3',
            children=[
                html.Label(f'{name.capitalize()} Tensors', className='form-label d-block')] + [
                html.Div(className='row', children=[
                    html.Dt(_type, className='col-sm-1'),
                    html.Dd(className='col-sm-11', children=[
                        dcc.Checklist(
                            id=f'{name}-{T_name}-selector',
                            className='form-check d-inline-block',
                            labelClassName='form-check-label pe-3', inputClassName='form-check-input',
                            options=[{'label': label, 'value': T_name}], value=[]
                        )
                        for T_name, label in reversed([(k, d['htmlName'])
                                                       for k, d in SPEC.items() if d.type == _type])
                    ])])
                for _type in TYPES
            ]
        )

    def layer_selector(name: str) -> html.Div:
        """Make Global Layers Selector"""
        return html.Div(
            # className='col-md-2 col-lg-1 border-end border-1',
            className='py-3',
            id=f'{name}-layer-selector-grouper',
            children=[
                html.Label(f'{name.capitalize()} Layers', className='form-label d-block'),
                html.Div([
                    dcc.Checklist(
                        id=f'{name}-layer-selector-{i}',
                        options=[{'label': str(i), 'value': i}], value=[],
                        className='form-check d-inline',
                        labelClassName='form-check-label pe-3', inputClassName='form-check-input'
                    )
                    for i in range(1, (num_enc_layers) + 1)]),
            ]
        )

    encoder_control = html.Div(
        className='col-6 border-end border-1',
        children=[
            html.Img(className='img-fluid pb-3', src=app.get_asset_url('encoder-rotated.svg')),
            tensor_selector('encoder', ENCODER_LAYER_TENSOR_SPEC, ENCODER_TENSOR_TYPES),
            layer_selector('encoder'),
        ]
    )
    decoder_control = html.Div(
        className='col-6 border-end border-1',
        children=[
            html.Img(className='img-fluid pb-3', src=app.get_asset_url('decoder-rotated.svg')),
            tensor_selector('decoder', DECODER_LAYER_TENSOR_SPEC, DECODER_TENSOR_TYPES),
            layer_selector('decoder'),

        ]
    )

    # Global Options Selector
    # global_options_selector = html.Div(
    #     className='col-auto border-end border-1', id='global-options-grouper',
    #     children=[
    #         html.Label('Global Options', htmlFor='global-options-selector'),
    #         dcc.Checklist(
    #             id='global-options-selector',
    #             className='form-check', inputClassName='form-check-input', labelClassName='form-check-label',
    #             labelStyle={'display': 'block'},
    #             options=[{'label': 'Zero Mean', 'value': 'zero-mean'},
    #                      {'label': 'Unit Variance', 'value': 'unit-variance'},
    #                      {'label': 'Absolute Values', 'value': 'absolute'}
    #                      ],
    #             value=[]
    #         )]
    # )
    # colorscale_selector = html.Div(
    #     className='col-auto border-end border-1', id='colorscale-wrapper',
    #     children=[
    #         html.Label('Colorscale', htmlFor='global-options-selector'),
    #         dcc.Dropdown(
    #             id='colorscale-selector',  # className='form-check',
    #             options=[{"value": x, "label": x}
    #                      for x in colorscales],
    #             value='cividis_r',
    #             clearable=False
    #         )]
    # )

    submit_buttons = html.Div(
        className='row',
        children=[
            html.Div(
                className='col-6 mt-4',
                children=html.Button(
                    id='control-bar-submit',
                    className='btn btn-outline-primary',
                    style={'width': '100%'},
                    children='Update Graphs 1',
                    n_clicks=0
                )
            ),
            html.Div(
                className='col-6 mt-4',
                children=html.Button(
                    id='control-bar-submit2',
                    className='btn btn-outline-primary',
                    style={'width': '100%'},
                    children='Update Graphs 2',
                    n_clicks=0
                )
            )
        ]
    )
    control_bar_row = html.Div(
        id='control-bar',
        className='row border border-1 p-4 rounded',
        children=[
            encoder_control,
            decoder_control,
            # global_options_selector,
            # colorscale_selector,
            submit_buttons
        ])

    return [control_bar_row]


def _layout_transformer_stack(LAYER_TENSOR_SPEC: Dict, T_l: List[Mapping], stack: str) -> List:
    """
    Make layout rows
    """
    def sidebar_control(id: str, spec: NDict, className: Optional[str] = None) -> Component:  # pylint: disable=redefined-builtin
        """Return html component for sidebar control object"""
        if spec.type == dcc.Checklist:
            return html.Div(
                title=spec.label,
                className=className,

                children=dcc.Checklist(
                    className='form-check',
                    labelClassName='form-check-label d-block',
                    inputClassName='form-check-input',
                    labelStyle={'font-size': 'small'},
                    id=id,
                    options=[dict(label=spec.labels[i], value=spec.options[i])
                             for i in range(len(spec.options))],
                    value=spec.default,
                    persistence=spec.persist
                ))
        elif spec.type == dcc.RadioItems:
            return html.Div(
                title=spec.label,
                className=className,
                children=dcc.RadioItems(
                    id=id,
                    options=[dict(label=spec.labels[i], value=spec.options[i])
                             for i in range(len(spec.options))],
                    value=spec.default,
                    className='form-check',
                    labelClassName='form-check-label d-block',
                    inputClassName='form-check-input',
                    labelStyle={'font-size': 'small'},
                    persistence=spec.persist
                ))
        elif spec.type == dcc.Dropdown:
            return html.Div(
                title=spec.label,
                className=className,
                children=dcc.Dropdown(
                    id=id,
                    style={'font-size': 'small'},
                    options=[dict(label=value, value=value)
                             for value in spec.options],
                    value=spec.default,
                    clearable=False,
                    persistence=spec.persist
                ))
        else:
            raise ValueError(f'Invalid spec type ({spec.type})')

    def T_block(i: int, T_spec: TensorSpec) -> Any:
        """Return HTML BLock for displaying a Weight or activation Tensor"""
        T_name = T_spec.key
        return html.Div(id=f'{stack}-layer-{i}-{T_name}',
                        className='row justify-content-start border-top border-1',
                        hidden=True,
                        children=[
                            html.Label(T_name, id=f'{stack}-layer-{i}-{T_name}-name'),
                            html.Div(
                                id=f'{stack}-layer-{i}-{T_name}-0',
                                className='col-auto btn-group-vertical btn-group-sm py-2',
                                # {'margin-top': f'{PLOT_MARGINS["t"]}px'},
                                children=[
                                    html.Div(
                                        className='btn-toolbar',
                                        children=html.Div(
                                            className='btn-group btn-group-sm',
                                            children=[
                                                html.Button(
                                                    '1',
                                                    title='Update Graph 1',
                                                    id=f'{stack}-layer-{i}-{T_name}-0-submit',
                                                    className='btn btn-outline-primary',  # btn-sm d-block mt-1 mb-2 me-2',
                                                    n_clicks=0
                                                ),
                                                html.Button(
                                                    '2',
                                                    title='Update Graph 2',
                                                    id=f'{stack}-layer-{i}-{T_name}-0-submit2',
                                                    className='btn btn-outline-primary',  # btn-sm d-block mt-1 mb-2 me-2',
                                                    n_clicks=0
                                                )
                                            ]
                                        )
                                    ),
                                    html.Div(
                                        id=f'{stack}-layer-{i}-{T_name}-0-controls',
                                        hidden=True,
                                        style={'overflow': "scroll;", 'max-height': "512px;"},
                                        children=[
                                            sidebar_control(
                                                id=f'{stack}-layer-{i}-{T_name}-0-control-{key}',
                                                spec=control,
                                                className='border-bottom')
                                            for key, control in T_spec.sidebar_controls.items()])
                                ]
                            ),
                            html.Div(
                                id=f'{stack}-layer-{i}-{T_name}-1',
                                className='col-auto',
                                hidden=True,
                                children=empty_graph(id=f'{stack}-layer-{i}-{T_name}-1-g')
                            ),
                            html.Div(
                                id=f'{stack}-layer-{i}-{T_name}-2',
                                className='col-auto',
                                hidden=True,
                                children=empty_graph(id=f'{stack}-layer-{i}-{T_name}-2-g')
                            )
                        ]
                        )

    return [
        html.Div(
            id=f'{stack}-layer-{i}', className='row border border-1 p-4 rounded',
            hidden=True,
            children=[
                html.Div(
                    className='col-auto border-end justify-content-center', id=f'{stack}-layer-{i}-controls',
                    children=[
                        html.Div(
                            f'{stack.capitalize()} Layer {i}',
                            className='text-nowrap mx-0 mb-2', style={'writing-mode': 'vertical-rl'}
                        ),
                        # html.Div(
                        #     className='border-bottom pb-1',
                        #     children=[
                        #         html.Label('Tie Colorscales', htmlFor=f'{stack}-layer-{i}-layer-options'),
                        #         dcc.Checklist(
                        #             className='col form-check',
                        #             labelClassName='form-check-label pe-3 me-3', inputClassName='form-check-input',
                        #             id=f'{stack}-layer-{i}-layer-options',
                        #             options=[{'label': 'qk', 'value': 'tie-qk-colorscale'},
                        #                      {'label': 'vo', 'value': 'tie-vo-colorscale'},
                        #                      {'label': 'all', 'value': 'tie-all-colorscales'}],
                        #             value=[]
                        #         )
                        #     ]),
                        # html.Div(
                        #     className='border-bottom pb-1',
                        #     children=[
                        #         html.Label('Horizontal Scale',
                        #                    className='form-label',
                        #                    htmlFor=f'{stack}-layer-{i}-hscale'),
                        #         dcc.Slider(
                        #             id=f'{stack}-layer-{i}-hscale',
                        #             className='ps-0',
                        #             min=0.25, max=5, step=0.25,
                        #             included=False,
                        #             marks={i: str(i) for i in range(6)},
                        #             value=1
                        #         )
                        #     ]),
                        # html.Div(
                        #     className='border-bottom pb-1',
                        #     children=[
                        #         html.Label(
                        #             'Vertical Scale',
                        #             className='form-label',
                        #             htmlFor=f'{stack}-layer-{i}-vscale'),
                        #         dcc.Slider(
                        #             id=f'{stack}-layer-{i}-vscale',
                        #             className='ps-0',
                        #             min=.25, max=5, step=.25,
                        #             included=False,
                        #             marks={i: str(i) for i in range(6)},
                        #             value=1
                        #         )
                        #     ]),
                        # html.Div(
                        #     className='border-bottom pb-1',
                        #     children=[
                        #         html.Label('Tie Vector Scales', htmlFor=f'{stack}-layer-{i}-layer-options2'),
                        #         dcc.RadioItems(
                        #             className='col form-check',
                        #             labelClassName='form-check-label pe-3 me-3', inputClassName='form-check-input',
                        #             id=f'{stack}-layer-{i}-layer-options2',
                        #             options=[{'label': 'Col', 'value': 'col'},
                        #                      {'label': 'Head', 'value': 'head'},
                        #                      {'label': 'All Heads', 'value': 'all-heads'},
                        #                      {'label': 'Layer', 'value': 'layer'},
                        #                      ],
                        #             value='col'
                        #         )
                        #     ]),

                        html.Div(
                            className='btn-group-vertical',  # style={'width': '100%'},
                            children=[
                                html.Button(
                                    '1',
                                    id=f'{stack}-layer-{i}-submit',
                                    className='fs-6 btn btn-outline-primary',
                                    style={'width': '100%', 'writing-mode': 'vertical-rl'},
                                    # style={'transform': 'rotate(-90deg)'},
                                    n_clicks=0

                                ),
                                html.Button(
                                    '2',
                                    id=f'{stack}-layer-{i}-submit2',
                                    className='fs-6 btn btn-outline-primary',
                                    style={'width': '100%', 'writing-mode': 'vertical-rl'},
                                    # style={'transform': 'rotate(-90deg)'},
                                    n_clicks=0

                                )])
                    ]
                ),
                html.Div(
                    className='col-auto',
                    children=[
                        T_block(i, T_spec) for T_spec in LAYER_TENSOR_SPEC.values() if T_spec.key in T_l[i - 1]
                    ])
            ],
            **{'data-initialized': 'false'})
        for i in range(len(T_l), 0, -1)]


def _embeddings_visualizer() -> List:
    """Return list of dash elements comprising the embedding space visualizer"""
    return [dcc.Tabs(
        id='embedding-viz-section',
        value='embedding-viz-spec',
        children=[
            dcc.Tab(label='Embedding Viz Spec',
                    value='embedding-viz-spec',
                    children=dcc.Textarea(
                        id='embedding-viz-spec',
                        contentEditable=True,
                        className='px-3',
                        style={'width': '100%'},
                        rows=25,
                        wrap='soft',
                        value=DEFAULT_EMBEDDING_VIZ_SPEC
                    )
                    ),
            dcc.Tab(label='Embedding Viz', value='embedding-viz',
                    children=html.Div(className='row border-1 px-3',
                                      children=[
                                          html.Button(
                                              'Go',
                                              id='embedding-viz-button',
                                              className='col-auto fs-6 btn btn-outline-primary',
                                              n_clicks=0
                                          ),
                                          html.Span(id='embedding-viz-graphs', className='col-auto')
                                      ]),
                    )
        ]
    ),
    ]


def _get_cminmax(*,
                 W_layer: Mapping[str, torch.Tensor],
                 keys: Optional[Sequence[str]],
                 global_options: List[str],
                 tensor_checklist: List[str],
                 data_dict: NDict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return color min/max values based on selected tensors. Set cmin and cmax to equal values so that cmid
    can be at 0
    """
    if not keys or (W_layer[keys[0]] is None):
        vmin, vmax, vmid = None, None, None
    else:
        W_l = {key: preprocess(W=W_layer[key], global_options=global_options, tensor_options=tensor_checklist,
                               data_dict=data_dict) for key in keys}
        vmin, vmax = W_l[keys[0]].min().item(), W_l[keys[0]].max().item()
        vmid = None
        for key in keys[1:]:
            vmin = min(vmin, W_l[key].min().item())
            vmax = max(vmax, W_l[key].max().item())
        if (vmin != 0) and (vmax != 0) and (np.sign(vmin) != np.sign(vmax)):
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
            vmid = 0
    return vmin, vmax, vmid


def _compute_colorscale_range(*,
                              T_name: str,
                              tied_tensors: List[str],
                              T: Mapping[str, torch.Tensor],
                              global_options: List[str],
                              tensor_checklist: List[str],
                              data_dict: NDict
                              ) -> Tuple[Optional[float],
                                         Optional[float],
                                         Optional[float]]:
    """Compute colorscale range for plot"""
    if 'center-colorscale' not in tensor_checklist:
        return None, None, None
    else:
        colorscale_keys = [T_name]
        if T_name in ['Wq', 'Wk', 'Wv', 'Wo']:
            if 'tie-all-colorscales' in tied_tensors:
                colorscale_keys = ['Wq', 'Wk', 'Wv', 'Wo']
            elif T_name in ['Wv', 'Wo'] and 'tie-vo-colorscales' in tied_tensors:
                colorscale_keys = ['Wv', 'Wo']
            elif T_name in ['Wq', 'Wk'] and 'tie-qk-colorscales' in tied_tensors:
                colorscale_keys = ['Wq', 'Wk']
        assert T_name in colorscale_keys
        return _get_cminmax(W_layer=T,
                            keys=colorscale_keys,
                            global_options=global_options,
                            tensor_checklist=tensor_checklist,
                            data_dict=data_dict)


def _compute_axes_range(*,
                        W_layer: Mapping[str, torch.Tensor],
                        option: str,
                        matrix_name: str,
                        head: int
                        ) -> Tuple[Optional[int], Optional[int]]:
    """Return min/max values of tensors depending on selected option"""
    if option == 'head':
        W = W_layer[matrix_name][head]
        vmin, vmax = W.min().item(), W.max().item()
    elif option == 'all-heads':
        W = W_layer[matrix_name]
        vmin, vmax = W.min().item(), W.max().item()
    elif option == 'layer':
        # W = W_layer
        vmin = min([_W.min().item() for _W in W_layer.values()])
        vmax = min([_W.max().item() for _W in W_layer.values()])
    else:
        return None, None

    return math.floor(vmin), math.ceil(vmax)


def update_graph1(*,
                  tensor_selector: List,
                  layer_selector: List,
                  global_options: List,
                  colorscale: str,
                  layer_options: List,
                  hscale: int,
                  vscale: int,
                  tensor_options: Params,
                  k: int,
                  _l: int,
                  T_name: str,
                  T_spec: TensorSpec,
                  data_dict: NDict,
                  stack: str
                  ) -> Tuple[Union[dcc.Graph, List], bool]:
    """
    Update main plot
    """
    T_l: List[Mapping[str, torch.Tensor]] = data_dict[f'{stack}_T_l']
    if (len(layer_selector) == 0) or (len(tensor_selector) == 0) or (T_l[_l - 1][T_name] is None):
        return empty_graph(id=f'{stack}-layer-{_l}-{T_name}-1-g'), True

    cmin, cmax, cmid = _compute_colorscale_range(T_name=T_name,
                                                 tied_tensors=layer_options,
                                                 T=T_l[_l - 1],
                                                 global_options=global_options,
                                                 tensor_checklist=tensor_options.checklist,
                                                 data_dict=data_dict)

    W, stack_heads, W_name = T_l[_l - 1][T_name], ('stack-heads' in tensor_options.checklist), T_name
    if 'transpose' in tensor_options.checklist:
        W, stack_heads = W.transpose(2, 1), (not stack_heads)
        transposed = True
    else:
        transposed = False

    if W is None:
        _LOGGER.warning(
            f'Graph1: Skipping tensor {T_name}. Not found in layer {_l}')

    return dcc.Graph(id=f'{stack}-layer-{_l}-{T_name}-1-g',
                     figure=plot1(W=W,
                                  cmin=cmin, cmax=cmax, cmid=cmid,
                                  colorscale=colorscale,
                                  hscale=hscale,
                                  vscale=vscale,
                                  stack_heads=stack_heads,
                                  W_name=W_name,
                                  T_spec=T_spec,
                                  transposed=transposed,
                                  input_tokens=T_l[_l - 1]['input_tokens'],
                                  show_tokens='show-tokens' in tensor_options.checklist,
                                  show_y_tokens='show-y-tokens' in tensor_options.checklist,
                                  show_magnitude='show-magnitude' in tensor_options.checklist,
                                  knn='knn' in tensor_options.checklist,
                                  k=k,
                                  data_dict=data_dict,
                                  global_options=global_options,
                                  tensor_options=tensor_options
                                  ),
                     responsive=False,
                     config={'displayModeBar': 'hover', 'showAxisDragHandles': False}), False


def update_graph2(*,
                  clickData: Any,
                  tensor_selector: List,
                  layer_selector: List,
                  global_options: List,
                  colorscale: str,
                  layer_options: List,
                  axis_range_option: str,
                  hscale: int,
                  vscale: int,
                  _l: int,
                  T_name: str,
                  T_spec: TensorSpec,
                  data_dict: NDict,
                  tensor_options: Params,
                  plot_col: bool,
                  k: int,
                  stack: str,
                  ) -> Tuple[dcc.Graph, bool]:
    """Update secondary plot"""
    T_l: List[Mapping[str, torch.Tensor]] = data_dict[f'{stack}_T_l']
    if (len(layer_selector) == 0) or (len(tensor_selector) == 0) or ('draw-graph2' not in tensor_options.checklist):
        return empty_graph(id=f'{stack}-layer-{_l}-{T_name}-2-g'), True

    if not clickData:
        x, y, head = 0, 0, 0
    else:
        x, y, head = clickData['points'][0]['x'], clickData['points'][0]['y'], clickData['points'][0]['curveNumber']

    cmin, cmax, cmid = _compute_colorscale_range(T_name=T_name, tied_tensors=layer_options, T=T_l[_l - 1],
                                                 global_options=global_options,
                                                 tensor_checklist=tensor_options.checklist,
                                                 data_dict=data_dict)
    axis_min, axis_max = _compute_axes_range(
        W_layer=T_l[_l - 1], matrix_name=T_name, head=head, option=axis_range_option)

    W = T_l[_l - 1][T_name]
    Wa_stacked_heads = ('stack-heads' in tensor_options.checklist)
    if 'transpose' in tensor_options.checklist:
        W = W.transpose(2, 1)
        Wa_stacked_heads = (not Wa_stacked_heads)
    if W is None:
        _LOGGER.warning(
            f'Graph2: Skipping tensor {T_name} ({T_spec.htmlName}). Not found in layer {_l}')

    return plot2(W=W,
                 head=head,
                 x=x, y=y, plot_col=plot_col,
                 cmin=cmin, cmax=cmax, cmid=cmid,
                 axis_min=axis_min, axis_max=axis_max,
                 colorscale=colorscale,
                 iD=f'{stack}-layer-{_l}-{T_name}-2-g',
                 plot1_size=compute_plot1_size(W=W,
                                               T_name=T_name,
                                               vscale=vscale,
                                               hscale=hscale,
                                               stack_heads=Wa_stacked_heads,
                                               show_tokens='show-tokens' in tensor_options.checklist,
                                               show_y_tokens='show-y-tokens' in tensor_options.checklist,
                                               show_magnitude='show-magnitude' in tensor_options.checklist,
                                               knn='knn' in tensor_options.checklist,
                                               k=k,
                                               tensor_options=tensor_options,
                                               data_dict=data_dict,
                                               do_preprocess=False
                                               ),
                 W_name=T_name,
                 T_spec=T_spec,
                 visible='head' if not clickData else 'vec',
                 global_options=global_options,
                 tensor_options=tensor_options,
                 data_dict=data_dict
                 ), False


def init_layer_callbacks(*,
                         app: dash.Dash,
                         data_dict: NDict,
                         LAYER_TENSOR_SPEC: Dict,
                         stack: str,
                         i: int) -> None:
    """Lazy init of callbacks for one layer"""
    @ app.callback(Output(f'{stack}-layer-{i}', 'hidden'),
                   #   Output(f'stack-layer-{i}', 'data-initialized'),
                   Input(f'{stack}-layer-selector-{i}', 'value'),
                   State(f'{stack}-layer-{i}', 'data-initialized'))
    def _hide_layer(show_layer: List, _initialized: str, _layer: int = i) -> bool:
        return len(show_layer) == 0

    T_l: List[Mapping[str, torch.Tensor]] = data_dict[f'{stack}_T_l']
    for T_name, T_spec in LAYER_TENSOR_SPEC.items():
        if T_name not in T_l[i - 1]:
            continue

        @ app.callback(Output(f'{stack}-layer-{i}-{T_name}-0-controls', 'hidden'),
                       Input(f'{stack}-layer-{i}-{T_name}-name', 'n_clicks'),
                       prevent_initial_call=True)
        def _show_controls(n_clicks: int) -> bool:
            return n_clicks % 2 == 0

        @ app.callback(Output(f'{stack}-layer-{i}-{T_name}-1', 'children'),
                       Output(f'{stack}-layer-{i}-{T_name}-1', 'hidden'),
                       Input('control-bar-submit', 'n_clicks'),
                       Input(f'{stack}-layer-{i}-submit', 'n_clicks'),
                       Input(f'{stack}-layer-{i}-{T_name}-0-submit', 'n_clicks'),
                       State(f'{stack}-{T_name}-selector', 'value'),
                       State(f'{stack}-layer-selector-{i}', 'value'),
                       #    State('global-options-selector', 'value'),
                       #    State('colorscale-selector', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-colorscale', 'value'),
                       #    State(f'{stack}-layer-{i}-layer-options', 'value'),
                       #    State(f'{stack}-layer-{i}-hscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-hscale', 'value'),
                       #    State(f'{stack}-layer-{i}-vscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-vscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-tensor_ops_checklist', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-viz_opns_checklist', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-k', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-plot-type', 'value'),
                       prevent_initial_call=True
                       )
        def _update_m_graph1(_global_submit_button: int,
                             _layer_submit_button: int,
                             _tensor_submit_button: int,
                             T_selector: List,
                             layer_selector: List,
                             #  global_options: List,
                             colorscale: str,
                             #  layer_options: List,
                             hscale: int,
                             vscale: int,
                             tensor_ops_checklist: List,
                             viz_opns_checklist: List,
                             k: int,
                             plot_type: str,
                             _l: int = i,  # bind the current value of loop variable i
                             T_name: str = T_name,  # bind the current value of loop variable m
                             T_spec: TensorSpec = T_spec,
                             ) -> Tuple[Union[dcc.Graph, List], bool]:
            # _LOGGER.info(f'Graph1 callback invoked for tensor {T_name} layer {_l}')
            return update_graph1(
                tensor_selector=T_selector,
                layer_selector=layer_selector,
                global_options=[],
                colorscale=colorscale,
                layer_options=[],
                hscale=hscale,
                vscale=vscale,
                tensor_options=Params(checklist=tensor_ops_checklist + viz_opns_checklist, plot_type=plot_type),
                k=k,
                _l=_l,
                T_name=T_name,
                T_spec=T_spec,
                data_dict=data_dict,
                stack=stack)

        @ app.callback(Output(f'{stack}-layer-{i}-{T_name}-2', 'children'),
                       Output(f'{stack}-layer-{i}-{T_name}-2', 'hidden'),
                       Input('control-bar-submit2', 'n_clicks'),
                       Input(f'{stack}-layer-{i}-submit2', 'n_clicks'),
                       Input(f'{stack}-layer-{i}-{T_name}-0-submit2', 'n_clicks'),
                       Input(f'{stack}-layer-{i}-{T_name}-1-g', 'clickData'),
                       State(f'{stack}-{T_name}-selector', 'value'),
                       State(f'{stack}-layer-selector-{i}', 'value'),
                       #    State('global-options-selector', 'value'),
                       #    State('colorscale-selector', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-colorscale', 'value'),
                       #    State(f'{stack}-layer-{i}-layer-options', 'value'),
                       #    State(f'{stack}-layer-{i}-layer-options2', 'value'),
                       #    State(f'{stack}-layer-{i}-hscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-hscale', 'value'),
                       #    State(f'{stack}-layer-{i}-vscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-vscale', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-tensor_ops_checklist', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-viz_opns_checklist', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-row-or-col', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-k', 'value'),
                       State(f'{stack}-layer-{i}-{T_name}-0-control-plot-type', 'value'),
                       prevent_initial_call=True
                       )
        def _update_m_graph2(_global_submit_button: int,
                             _layer_submit_button: int,
                             _tensor_submit_button: int,
                             clickData: Any,
                             T_selector: List,
                             layer_selector: List,
                             #  global_options: List,
                             colorscale: str,
                             #  layer_options: List,
                             #  axis_range_option: str,
                             hscale: int,
                             vscale: int,
                             tensor_ops_checklist: List,
                             viz_opns_checklist: List,
                             row_or_col: str,
                             k: int,
                             plot_type: str,
                             _l: int = i,  # bind the current value of loop variable i
                             T_name: str = T_name,  # bind the current value of loop variable m
                             T_spec: TensorSpec = T_spec,
                             ) -> Tuple[dcc.Graph, bool]:
            return update_graph2(
                clickData=clickData,
                tensor_selector=T_selector,
                layer_selector=layer_selector,
                global_options=[],
                colorscale=colorscale,
                layer_options=[],
                axis_range_option='col',
                hscale=hscale,
                vscale=vscale,
                tensor_options=Params(checklist=tensor_ops_checklist + viz_opns_checklist, plot_type=plot_type),
                plot_col=(row_or_col == 'col'),
                k=k,
                _l=_l,
                T_name=T_name,
                T_spec=T_spec,
                data_dict=data_dict,
                stack=stack)

        @ app.callback(Output(f'{stack}-layer-{i}-{T_name}', 'hidden'),
                       Input(f'{stack}-{T_name}-selector', 'value'),
                       Input(f'{stack}-layer-selector-{i}', 'value'),
                       prevent_initial_call=True
                       )
        def _hide_tensor(T_selector: List,
                         layer_selector: List
                         ) -> bool:
            return (len(T_selector) == 0) or (len(layer_selector) == 0)


def _setup_embedding_callbacks(*, app: dash.Dash, data_dict: NDict):
    """Initialize callbacks in the embeddings section"""
    @ app.callback(Output('embedding-viz-graphs', 'children'),
                   State('embedding-viz-spec', 'value'),
                   Input('embedding-viz-button', 'n_clicks'))
    def _draw_graphs(spec: str, n_clicks: int):
        if not n_clicks > 0:
            return dash.no_update
        else:
            try:
                o = YAML(typ="safe", pure=True).load(spec.strip())
                _LOGGER.info(f'embedding-viz-button.n_clicks={n_clicks}')
                return plot_embeddings(o, data_dict)
            except ruamel.yaml.error.YAMLError as e:
                return dash.html.Pre(dash.html.Code(repr(e) + '\n' + spec.strip()))

    @ app.callback(Output('embedding-viz-spec', 'value'),
                   State('embedding-viz-spec', 'value'),
                   Input('embedding-viz-spec', 'n_blur'))
    def _verify_spec(spec: str, _n: int):
        ret_val: Any
        try:
            YAML(typ="safe", pure=True).load(spec.strip())
            # json.loads(spec.strip())
            return dash.no_update
        except ruamel.yaml.error.YAMLError as e:
            ret_val = repr(e) + '\n' + spec.strip()
            return ret_val


def layout_app(*, app: dash.Dash, data_dict: NDict, config: Dict, app_mode: str) -> None:
    """Populate dash app layout and setup callbacks"""
    T_l_enc: List[Mapping[str, torch.Tensor]] = data_dict['encoder_T_l']
    T_l_dec: List[Mapping[str, torch.Tensor]] = data_dict['decoder_T_l']
    num_enc_layers = len(T_l_enc)
    app.layout = html.Div(
        className='container-fluid',
        children=(
            _config_bar_layout(config, data_dict, app)
            + _tensor_controlbar_layout(num_enc_layers, app)
            + _embeddings_visualizer()
        ) if app_mode == 'view_embeddings' else (
            _config_bar_layout(config, data_dict, app)
            + _embeddings_visualizer()
            + _tensor_controlbar_layout(num_enc_layers, app)
            + _layout_transformer_stack(DECODER_LAYER_TENSOR_SPEC, T_l_dec, stack='decoder')
            + _layout_transformer_stack(ENCODER_LAYER_TENSOR_SPEC, T_l_enc, stack='encoder')
        )
    )

    if app_mode == 'view_embeddings':
        _setup_embedding_callbacks(app=app, data_dict=data_dict)
    else:
        _setup_embedding_callbacks(app=app, data_dict=data_dict)
        for i in range(1, num_enc_layers + 1):
            init_layer_callbacks(app=app, data_dict=data_dict, i=i,
                                 LAYER_TENSOR_SPEC=ENCODER_LAYER_TENSOR_SPEC, stack='encoder')

        for i in range(1, len(T_l_dec) + 1):
            init_layer_callbacks(app=app, data_dict=data_dict, i=i,
                                 LAYER_TENSOR_SPEC=DECODER_LAYER_TENSOR_SPEC, stack='decoder')
