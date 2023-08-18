"""Common imports across viz modules"""
from typing import Optional, Any, Union
from collections import OrderedDict
import math
import numpy as np
from ruamel.yaml import YAML
import plotly.express as px
from dash import dcc
from attrdict import Params, NDict

# UCS colormap is perceptually Uniform Color Space i.e. it has linear visual perception delta of which
# viridis is the gold standard (see https://arxiv.org/pdf/1712.01662.pdf)
# cividis is viridis adjusted for Color Vision Deficiency (red-green)

SEQUENTIAL_UCS_COLORMAPS = ['cividis_r', 'viridis_r', 'cividis', 'viridis']
# Use digerging colormaps for data that has a natural mid-point - like weight vectors and activations. Set cmid
# on the colorscale
# See https://plotly.com/python/builtin-colorscales/
DIVERGING_COLORMAPS = ['BrBG', 'icefire', 'icefire_r', 'portland']
PLOT_MARGINS = Params({'t': 50, 'b': 100, 'r': 100, 'l': 100, 'autoexpand': False})
COLORSCALES = px.colors.named_colorscales()
COLORSCALES = sorted(COLORSCALES + [c + '_r' for c in COLORSCALES])


def roundup(x: Union[float, int]) -> int:
    """Round up to the nearest integer away from zero"""
    return int(math.ceil(abs(x)) * np.sign(x))


class TensorSpec(NDict):
    """Base class of a tensor visualization specification"""

    def __init__(self,
                 *,
                 key: str,
                 type_: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 colorscale: str = 'icefire_r',
                 draw_graph2: bool = True,
                 xaxis_title: Optional[str] = None,
                 yaxis_title: Optional[str] = None,
                 vscale: Union[int, float] = 1,
                 hscale: Union[int, float] = 1,
                 show_tokens: bool = False,
                 show_y_tokens: bool = False,
                 knn: bool = True,
                 k: int = 2,
                 stack_heads: bool = False,
                 plot_type: str = 'heatmap',
                 transpose: bool = False,
                 normalize_cols: bool = False,
                 normalize_rows: bool = False,
                 prob_cols: bool = False,
                 prob_rows: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(
            NDict(
                key=key,
                type=type_,
                htmlName=htmlName or key,
                latexName=latexName or htmlName or key,
                colorscale=colorscale,
                draw_graph2=draw_graph2,
                sidebar_controls=NDict({
                    'hscale': NDict(
                        label='Horizontal Scale',
                        type=dcc.Dropdown,
                        options=[.125, .25, .5] + list(range(1, max(20, roundup(hscale)) + 1)),
                        default=hscale,
                        persist=True,
                    ),
                    'vscale': NDict(
                        label='Vertical Scale',
                        type=dcc.Dropdown, options=[.125, .25, .5] + list(range(1, max(100, roundup(vscale)) + 1)),
                        default=int(vscale),
                        persist=True
                    ),
                    'colorscale': NDict(
                        label='Colorscale', type=dcc.Dropdown, options=COLORSCALES,
                        default=colorscale,
                        persist=True
                    ),
                    'k': NDict(
                        label='K',
                        type=dcc.Dropdown, options=list(range(1, max(10, roundup(k)) + 1)),
                        default=int(k),
                        persist=True
                    ),
                    'plot-type': NDict(
                        type=dcc.RadioItems,
                        options=['heatmap', 'bar', 'surface'],
                        labels=['Heatmap', 'Bar', 'Surface'],
                        default=plot_type,
                        persist=False
                    ),
                    'tensor_ops_checklist': NDict(
                        type=dcc.Checklist,
                        options=['transpose',
                                 'dp-cols-w-op',
                                 'prob-rows', 'prob-cols',
                                 'pre-zeromean-cols',
                                 'pre-unitnorm-cols', 'pre-unitvar-rows', 'pre-unitvar-cols',
                                 'dot-product',
                                 # 'absolute',
                                 'log-scale',
                                 # 'unit-variance',
                                 'normalize-cols', 'normalize-rows',
                                 'exp', 'softmax-rows', 'softmax-cols',
                                 ],
                        labels=['Transpose (& invert Stacking)',
                                'DotProduct Cols w/ O/P',
                                'Prob Rows', 'Prob Cols',
                                'ZeroMean Cols',
                                'UnitNorm Cols', 'UnitVar Rows', 'UnitVar Cols',
                                'DotProduct Cols',
                                # 'Abs',
                                'Log',
                                # 'Unit Variance (Headwise)',
                                'UnitNorm Col Vecs', 'UnitNorm Row Vecs',
                                'Exponential', 'Softmax Rows', 'Softmax Cols',
                                ],
                        default=(
                            (['transpose'] if transpose else []) +
                            (['prob-cols'] if prob_cols else []) +
                            (['prob-rows'] if prob_rows else []) +
                            (['normalize-cols'] if normalize_cols else []) +
                            (['normalize-rows'] if normalize_rows else [])
                        ),
                        persist=True
                    ),
                    'viz_opns_checklist': NDict(
                        type=dcc.Checklist,
                        options=['show-tokens', 'show-y-tokens', 'show-magnitude', 'knn', 'stack-heads', 'draw-graph2',
                                 'center-colorscale'],
                        labels=['Show Tokens', 'Show Y Tokens', 'Show Magnitude', 'KNN', 'Stack Heads', 'Plot Graph2',
                                'Center Colorscale'],
                        default=(
                            # (['center-colorscale']) +
                            (['show-tokens'] if show_tokens else []) +
                            (['show-y-tokens'] if show_y_tokens else []) +
                            (['knn'] if knn else []) +
                            (['stack-heads'] if stack_heads else []) +
                            (['draw-graph2'] if draw_graph2 else [])
                        ),
                        persist=True
                    ),
                    'row-or-col': NDict(
                        type=dcc.RadioItems,
                        options=['col', 'row'],
                        labels=['Graph2: Plot Col', 'Graph2: Plot Row'],
                        default='col',
                        persist=False
                    )
                }),
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                # show_y_tokens=show_y_tokens,
                # stack_heads=stack_heads,
                **kwargs
            )
        )
        assert not (set(self.sidebar_controls['tensor_ops_checklist'].options)
                    & set(self.sidebar_controls['viz_opns_checklist'].options)), 'Overlapping options in two checklists'


class ActivationSpec(TensorSpec):
    """Activation tensor spec"""

    def __init__(self,
                 *,
                 key: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 xaxis_title: str = 'Input Pos',
                 yaxis_title: str = 'Vector Dims',
                 colorscale: str = 'cividis_r',
                 **kwargs: Any) -> None:
        super().__init__(
            key=key,
            type_='A',
            htmlName=htmlName or key,
            latexName=latexName or htmlName or key,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            colorscale=colorscale,
            **kwargs
        )


class WeightsSpec(TensorSpec):
    """Weights tensor spec"""

    def __init__(self,
                 *,
                 key: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 xaxis_title: str = 'Out Dims',
                 yaxis_title: str = 'Inp Dims',
                 colorscale: str = 'icefire_r',
                 **kwargs: Any) -> None:
        super().__init__(
            key=key,
            type_='W',
            htmlName=htmlName or key,
            latexName=latexName or htmlName or key,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            colorscale=colorscale,
            **kwargs)


class AttributionsSpec(TensorSpec):
    """Attributions tensor spec"""

    def __init__(self,
                 *,
                 key: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 xaxis_title: str = 'Inp Pos',
                 yaxis_title: str = 'Out Pos',
                 show_tokens: bool = True,
                 knn: bool = False,
                 colorscale: str = 'cividis_r',
                 prob_cols: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(
            key=key,
            type_='Att',
            htmlName=htmlName or key,
            latexName=(latexName or htmlName or key),
            colorscale=colorscale,
            draw_graph2=False,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            show_tokens=show_tokens,
            knn=knn,
            normalize_rows=True,
            prob_cols=prob_cols,
            **kwargs)
        self.sidebar_controls['row-or-col'].default = 'row'


class AttentionSpec(TensorSpec):
    """Attention tensor spec"""

    def __init__(self,
                 *,
                 key: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 xaxis_title: str = 'Key Pos',
                 yaxis_title: str = 'Query Pos',
                 show_tokens: bool = True,
                 knn: bool = False,
                 colorscale: str = 'cividis_r',
                 **kwargs: Any) -> None:
        super().__init__(
            key=key,
            type_='Attn',
            htmlName=htmlName or key,
            latexName=(latexName or htmlName or key),
            colorscale=colorscale,
            draw_graph2=False,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            show_tokens=show_tokens,
            knn=knn,
            **kwargs)
        self.sidebar_controls['row-or-col'].default = 'row'


class PosEncodingSpec(TensorSpec):
    """Position Encodings tensor spec"""

    def __init__(self,
                 *,
                 key: str,
                 latexName: Optional[str] = None,
                 htmlName: Optional[str] = None,
                 xaxis_title: str = 'Pos Encoding',
                 yaxis_title: str = 'Key Pos',
                 show_tokens: bool = False,
                 knn: bool = False,
                 colorscale: str = 'cividis_r',
                #  share_x_axis: bool = 'all',  # 'all' | 'columns' | False
                #  share_y_axis: bool = 'all',  # 'all' | 'rows' | False
                 **kwargs: Any) -> None:
        super().__init__(
            key=key,
            type_='Attn',
            htmlName=htmlName or key,
            latexName=(latexName or htmlName or key),
            colorscale=colorscale,
            draw_graph2=False,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            show_tokens=show_tokens,
            knn=knn,
            # share_x_axis=share_x_axis,
            # share_y_axis=share_y_axis,
            **kwargs)
        self.sidebar_controls['row-or-col'].default = 'row'


ENCODER_LAYER_TENSOR_SPEC = OrderedDict((spec.key, spec)  # type: ignore
                                        for spec in [
    # List them in the order you want them displayed
    AttributionsSpec(key='FinalDecOutAttribs', htmlName='AtrF', latexName=r'Atr_Dec', yaxis_title='Decoder Out Pos',
                     vscale=100.),
    ActivationSpec(key='FinalNorm', htmlName='Nfinal', latexName=r'N_{final}'),
    WeightsSpec(key='W_FinalNorm', latexName='W_{FinalNorm}', hscale=25),

    ActivationSpec(key='FF2+', latexName='FF_{2+}'),
    ActivationSpec(key='FF2', latexName='FF_2'),
    ActivationSpec(key='RELU', knn=False),
    WeightsSpec(key='Wff2', latexName=r'W_{ff2}'),
    ActivationSpec(key='FF1', latexName='FF_1'),
    WeightsSpec(key='Wff1', latexName=r'W_{ff1}'),
    ActivationSpec(key='NormFF', htmlName='Nff', latexName=r'N_{FF}'),
    WeightsSpec(key='W_NormFF', latexName='W_{Norm_{FF}}', hscale=25),

    ActivationSpec(key='O+'),
    ActivationSpec(key='O'),
    WeightsSpec(key='Wo', latexName='W_o', stack_heads=True, transpose=True),
    AttributionsSpec(key='SelfAttM', htmlName='SelfAtt.mean', latexName=r'Att_{self}^{mean}', show_y_tokens=True),
    AttributionsSpec(key='SelfAtt', htmlName='SelfAtt', latexName=r'Att_{self}', show_y_tokens=True),
    ActivationSpec(key='V', stack_heads=True),
    WeightsSpec(key='Wv', latexName='W_v'),
    AttributionsSpec(key='Pos', yaxis_title='Pos Encoding', xaxis_title='Relative Key Pos',
                     show_tokens=False, hscale=1, vscale=100, stack_heads=True, plot_type='bar'),
    ActivationSpec(key='K', stack_heads=True),
    WeightsSpec(key='Wk', latexName='W_k'),
    ActivationSpec(key='Q', stack_heads=True),
    WeightsSpec(key='Wq', latexName='W_q'),
    AttributionsSpec(key='DecOutAttribs', htmlName='AtrD', latexName=r'Atr_Dec', yaxis_title='Decoder Out Pos',
                     xaxis_title='Encoder Input Position', vscale=100.),
    AttributionsSpec(key='TopLayerAttribs', htmlName='AtrE', latexName=r'Atr_Enc', yaxis_title='Encoder Out Pos',
                     xaxis_title='Encoder Input Position', show_y_tokens=True),
    ActivationSpec(key='NormSA', latexName='N_{SA}', htmlName='N_SA'),
    WeightsSpec(key='W_NormSA', latexName='W_{Norm_{SA}}', hscale=25),
    ActivationSpec(key='E', show_tokens=True),
])

ENCODER_TENSOR_TYPES = sorted({spec.type  # type: ignore
                               for spec in ENCODER_LAYER_TENSOR_SPEC.values()})


DECODER_LAYER_TENSOR_SPEC = OrderedDict((spec.key, spec)  # type: ignore
                                        for spec in [
    # List them in the order you want them displayed
    ActivationSpec(key='FinalNorm', htmlName='Nfinal', latexName=r'N_{final}', hscale=25),
    WeightsSpec(key='W_FinalNorm', latexName='W_{FinalNorm}', hscale=25),

    ActivationSpec(key='FF2+', latexName='FF_{2+}', hscale=25),
    ActivationSpec(key='FF2', latexName='FF_2', hscale=25),
    ActivationSpec(key='RELU', knn=False, hscale=25),
    WeightsSpec(key='Wff2', latexName=r'W_{ff2}'),
    ActivationSpec(key='FF1', latexName='FF_1', hscale=25),
    WeightsSpec(key='Wff1', latexName=r'W_{ff1}'),
    ActivationSpec(key='NormFF', htmlName='Nff', latexName=r'N_{FF}', hscale=25),
    WeightsSpec(key='W_NormFF', latexName='W_{Norm_{FF}}', hscale=25),

    ActivationSpec(key='CO+', latexName='O_c', hscale=25),
    ActivationSpec(key='CO', latexName='O_c', hscale=25),
    WeightsSpec(key='CWo', latexName='W_o^c', stack_heads=True, transpose=True),
    AttributionsSpec(key='CrossAttM', htmlName='CrossAtt.mean', latexName=r'Att_{cross}^{mean}', show_y_tokens=False,
                     vscale=50, hscale=5),
    AttributionsSpec(key='CrossAtt', htmlName='CrossAtt', latexName=r'Att_{cross}', show_y_tokens=False,
                     stack_heads=True, vscale=25, hscale=5),
    ActivationSpec(key='CV', latexName='V_c', stack_heads=True, hscale=25),
    WeightsSpec(key='CWv', latexName='W_v^c'),
    # AttributionsSpec(key='CPos', yaxis_title='', xaxis_title='Key Pos', colorscale='icefire_r',
    #                  show_tokens=False, hscale=1, vscale=100, stack_heads=True, plot_type='bar'),
    ActivationSpec(key='CK', latexName='K_c', stack_heads=True, hscale=25),
    WeightsSpec(key='CWk', latexName='W_k^c'),
    ActivationSpec(key='CQ', latexName='Q_c', stack_heads=True, hscale=25),
    WeightsSpec(key='CWq', latexName='W_q^c'),
    AttributionsSpec(key='CrossAtr', htmlName='AtrC', latexName=r'Atr_Cross', yaxis_title='Decoder Out Pos',
                     xaxis_title='Encoder Input Position', vscale=100., show_y_tokens=True),
    ActivationSpec(key='NormCA', latexName='N_{CA}', htmlName='N_CA', hscale=25),
    WeightsSpec(key='W_NormCA', latexName='W_{Norm_{CA}}', hscale=25),

    ActivationSpec(key='O+', hscale=25),
    ActivationSpec(key='O', hscale=25),
    WeightsSpec(key='Wo', latexName='W_o', stack_heads=True, transpose=True),
    # AttributionsSpec(key='SelfAtt', htmlName='Self', latexName=r'Att_{self}', show_y_tokens=True),
    ActivationSpec(key='V', stack_heads=True, hscale=25),
    WeightsSpec(key='Wv', latexName='W_v'),
    # AttributionsSpec(key='Pos', yaxis_title='Query Pos', xaxis_title='Key Pos', colorscale='icefire_r',
    #                  show_tokens=False, hscale=1, vscale=100, stack_heads=True, plot_type='bar'),
    ActivationSpec(key='K', stack_heads=True, hscale=25),
    WeightsSpec(key='Wk', latexName='W_k'),
    ActivationSpec(key='Q', stack_heads=True, hscale=25),
    WeightsSpec(key='Wq', latexName='W_q'),
    ActivationSpec(key='NormSA', latexName='N_{SA}', htmlName='N_SA', hscale=25),
    WeightsSpec(key='W_NormSA', latexName='W_{Norm_{SA}}', hscale=25),
    # AttributionsSpec(key='TopLayerAttribs', htmlName='AtrD', latexName=r'Atr_Dec', yaxis_title='Decoder Out Pos',
    #                  show_y_tokens=True),
    ActivationSpec(key='E', show_tokens=True, hscale=25),
])
DECODER_TENSOR_TYPES = sorted({spec.type  # type: ignore
                               for spec in DECODER_LAYER_TENSOR_SPEC.values()})


# EMPTY_FIGURE = {
#     'data': [],
#     'layout': go.Layout(
#         xaxis={
#             'showticklabels': False,
#             'ticks': '',
#             'showgrid': False,
#             'zeroline': False
#         },
#         yaxis={
#             'showticklabels': False,
#             'ticks': '',
#             'showgrid': False,
#             'zeroline': False
#         },
#         # width=10,
#         # height=10,
#         showlegend=False,
#         autosize=True
#     )
# }


def empty_graph(id: str) -> dcc.Graph:  # pylint: disable=redefined-builtin
    """Return an empty graph object that stays invisible"""
    return dcc.Graph(id=id,
                     #  figure=EMPTY_FIGURE, responsive=True, config={'displayModeBar': False}
                     )


DEFAULT_EMBEDDING_VIZ_SPEC = """
[

            # Best config is: metric=nip, n_neighbors=1000 w/o densmap
            # Intersting config is: metric=nip, out_metric=nip, no densmap
            # Pretraining the map based on word-vecs alone produces a flat (2 dimensional map) except for the Final Norm
            # which alone occupies the third dimension - intersting. However better to train everything together.

    {
        "preprocess": {"unit_norm": false},  # chart level preprocessing items
        "plot_size": {"width": 1024, "height": 1024},
        "umap": {
            "metric": "nip",
            "out_metric": null,
            "seed": null,  # Setting to null enables multi-threaded but fully stochastic behavior
            "num_embeddings_to_train": null,  # null => all
            "num_embeddings_to_plot": 1000,
            "n_neighbors": 1000,  # null => # points
            "num_dims": 3,
            "cache": True,  # default is True
            "from_cache_only": True,  # Do not recompute if not cached
            "n_epochs": 500,  # 200-500
            "dens_map": False,
            "dens_lambda": 2.0
        },
        "support_set": {"color": "lightgray", "size": 2},
        "paths": [
            {
                "stack": "encoder",
                "positions": [202, 223, 261, 344, 429, 532],  # ([, L:000, L:000A, 7, L:1,  </s>)
                # "positions": [0, 202, 223, 261, 344, 429, 531, 532],  # (Grade, [, L:000, L:000A, 7, L:1, L:, </s>)

                "layers": [0,1,2,3,4,5],
                "points": [
                    {"tensor_key": "NormSA"},
                    # {"tensor_key": "O", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "O+", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "FF2", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "FF2+", "preprocess": {"unit_norm": true}},
                    {"tensor_key": "NormFF"}
                ]
            },
            {
                "stack": "decoder",
                "positions": [0],
                "layers": [0,1,2,3,4,5],
                "points": [
                    {"tensor_key": "NormSA"},
                    # {"tensor_key": "O", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "O+", "preprocess": {"unit_norm": true}},
                    {"tensor_key": "NormCA"},
                    # {"tensor_key": "CO+", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "FF2", "preprocess": {"unit_norm": true}},
                    # {"tensor_key": "FF2+", "preprocess": {"unit_norm": true}},
                    {"tensor_key": "NormFF"},
                    {"tensor_key": "FinalNorm"}
                ]
            }
        ],
    },


]
"""
