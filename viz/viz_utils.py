from typing import Optional
import os
from pathlib import Path
import tempfile
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from viz.viz_plot import plot1
from viz.viz_spec import AttentionSpec, PosEncodingSpec
from viz.viz_commons import mid_point
from params import Params


def show_fig(fig: go.Figure, renderer: Optional[str] = 'iframe connected', figdir=None) -> str:
    if renderer.startswith('iframe'):
        figdir = tempfile.mkdtemp(dir=figdir)
        fig.show(renderer, html_directory=figdir)
        print(f'Saving plot to dir {figdir}')
        logging.warning('This will not display in vscode')
    elif renderer == 'offline':
        figfile = write_iframe(fig)
        print(f'Saving plot to file {figfile}')
        logging.warning('This will not display in vscode')
        display.display(display.IFrame(src=figfile, width=1000, height=600))
    else:
        fig.show(renderer)


def write_fig(fig: go.Figure, format: Optional[str] = 'png', figdir=None) -> str:
    _, figfile = tempfile.mkstemp(dir=figdir, suffix=f'.{format}')
    figfile = Path(figfile).relative_to(Path(figdir).resolve())
    figfile = '/'.join([figdir, str(figfile)])
    fig.write_image(file=figfile)
    return figfile


def get_html(fig: go.Figure, figdir=None) -> str:
    _, figfile = tempfile.mkstemp(dir=figdir, suffix=f'.txt')
    figfile = Path(figfile).relative_to(Path(figdir).resolve())
    figfile = '/'.join([figdir, str(figfile)])
    fig.write_html(file=figfile, full_html=False, auto_open=False, include_plotlyjs='cdn')
    with open(figfile, 'rt') as f:
        return f.read()
#     return figfile


def write_iframe(fig: go.Figure, figdir=None) -> str:
    _, figfile = tempfile.mkstemp(dir=figdir, suffix=f'.html')
    figfile = Path(figfile).relative_to(Path(figdir).resolve())
    figfile = '/'.join([figdir, str(figfile)])
    fig.write_html(file=figfile, full_html=True, auto_open=False, include_plotlyjs='cdn', include_mathjax='cdn')
    return figfile


def plot_attention_maps(*,
                        stack, head, layer,
                        attention_score=None, position_bias=None, show_kernel_only=False, attention_weights=None,
                        ModelName, show_tokens=False, datum_id, data_dict,
                        fig_format='iframe_connected', tempdir=None, stack_heads=False,
                        hscale=None):
    if position_bias is not None:
        if stack == 'decoder':
            show_row = position_bias.shape[1] - 10
        else:
            show_row = mid_point(position_bias.shape[1])
        kernel_x_ticks = np.arange(position_bias.shape[1]) - show_row
        position_bias_kernel = position_bias[:, show_row:show_row+1]  # (H, 1, S) = (1, 1, S)
        if show_kernel_only:
            position_bias = None

    OutputPathPrefix = f"figures/{ModelName.replace('/', '-')}/dataum_id={datum_id.replace('/', '-')}-stack={stack}-layer={layer}-head={head}"
    titles = [f'$Q_{{{head}}}.K_{{{head}}}^T$',
              f'Position Bias: {stack} layer {layer}, head {head}',
              f'Position Bias Kernel: {stack} layer {layer}, head {head}',
              f'Combined Attention Score: {stack} layer {layer}, head {head}',
              f'Final Attention Weights (after softmax): {stack} layer {layer}, head {head}'
              ]
    input_tokens = data_dict.input_tokens_unstripped if show_tokens else None

    os.makedirs('figures', exist_ok=True)
    if attention_score is not None and position_bias is not None:
        fig1 = plot1(W=attention_score - position_bias,
                     W_name=titles[0],
                     T_spec=AttentionSpec(key=None, xaxis_title='Key Position', yaxis_title='Query Position'),
                     colorscale='cividis_r',
                     vscale=1, hscale=hscale or 1,
                     stack_heads=stack_heads,
                     input_tokens=input_tokens,
                     knn=False,
                     k=1,
                     cmid=None,
                     data_dict=data_dict,
                     tensor_options=Params(checklist=[], plot_type='heatmap'),
                     show_tokens=show_tokens,
                     show_y_tokens=show_tokens,
                     _is_standalone=True
                     )
        OutputFileName = OutputPathPrefix + f"-{titles[0]}"
        if fig_format is not None:
            show_fig(fig1, fig_format, figdir=tempdir)
        else:
            print(f'Saving to {OutputFileName}')
        fig1.write_image(file=OutputFileName + '.pdf')
    if position_bias is not None:
        fig2 = plot1(W=position_bias,
                     W_name=titles[1],
                     T_spec=PosEncodingSpec(key=None, xaxis_title='Key Position', yaxis_title='Query Position'),
                     colorscale='cividis_r',
                     vscale=1, hscale=hscale or 1,
                     stack_heads=stack_heads,
                     input_tokens=input_tokens,
                     knn=False,
                     k=1,
                     cmid=None,
                     data_dict=data_dict,
                     tensor_options=Params(checklist=[], plot_type='heatmap'),
                     show_tokens=show_tokens,
                     show_y_tokens=show_tokens,
                     _is_standalone=True
                     )
        OutputFileName = OutputPathPrefix + f"-{titles[1]}"
        if fig_format is not None:
            show_fig(fig2, fig_format, figdir=tempdir)
        else:
            print(f'Saving to {OutputFileName}')
        fig2.write_image(file=OutputFileName + '.pdf')

    if position_bias_kernel is not None:
        fig3 = plot1(W=position_bias_kernel,
                     W_name=titles[2],
                     T_spec=PosEncodingSpec(key=None, yaxis_title='Attention Bias', xaxis_title='Relative Key Pos'),
                     colorscale='cividis_r',
                     vscale=100, hscale=hscale or 1,
                     stack_heads=stack_heads,
                     input_tokens=input_tokens,
                     knn=False,
                     k=1,
                     cmid=None,
                     data_dict=data_dict,
                     tensor_options=Params(checklist=[], plot_type='bar'),
                     show_tokens=show_tokens,
                     show_y_tokens=show_tokens,
                     _is_standalone=True,
                     _bar_x_ticks=kernel_x_ticks
                     )
        OutputFileName = OutputPathPrefix + f"-{titles[2]}"
        if fig_format is not None:
            show_fig(fig3, fig_format, figdir=tempdir)
        else:
            print(f'Saving to {OutputFileName}')
        fig3.write_image(file=OutputFileName + '.pdf')
    if attention_score is not None:
        fig4 = plot1(W=attention_score,
                     W_name=titles[3],
                     T_spec=AttentionSpec(key=None, xaxis_title='Key Position', yaxis_title='Query Position'),
                     colorscale='cividis_r',
                     vscale=1, hscale=hscale or 1,
                     stack_heads=stack_heads,
                     input_tokens=input_tokens,
                     knn=False,
                     k=1,
                     cmid=None,
                     data_dict=data_dict,
                     tensor_options=Params(checklist=[], plot_type='heatmap'),
                     show_tokens=show_tokens,
                     show_y_tokens=show_tokens,
                     _is_standalone=True
                     )
        OutputFileName = OutputPathPrefix + f"-{titles[3]}"
        if fig_format is not None:
            show_fig(fig4, fig_format, figdir=tempdir)
        else:
            print(f'Saving to {OutputFileName}')
        fig4.write_image(file=OutputFileName + '.pdf')

    if attention_weights is not None:
        fig5 = plot1(W=attention_weights,
                     W_name=titles[4],
                     T_spec=AttentionSpec(key=None, xaxis_title='Key Position', yaxis_title='Query Position'),
                     colorscale='cividis_r',
                     vscale=1, hscale=hscale or 1,
                     stack_heads=stack_heads,
                     input_tokens=input_tokens,
                     knn=False,
                     k=1,
                     cmid=None,
                     data_dict=data_dict,
                     tensor_options=Params(checklist=[], plot_type='heatmap'),
                     show_tokens=show_tokens,
                     show_y_tokens=show_tokens,
                     _is_standalone=True
                     )
        OutputFileName = OutputPathPrefix + f"-{titles[4]}"
        if fig_format is not None:
            show_fig(fig5, fig_format, figdir=tempdir)
        else:
            print(f'Saving to {OutputFileName}')
        fig5.write_image(file=OutputFileName + '.pdf')
        #  fig.write_image(file=OutputFileName + '.svg')
        #  fig.write_image(file=OutputFileName + '.png')
