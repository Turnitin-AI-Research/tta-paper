"""
Misc utils for inferencing reference implementation
"""
from typing import Mapping, Union, Optional, Dict
import os
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig

PathName = Union[str, Path]


def load_checkpoint(ckpt_path: str) -> Dict:
    """Load Torch checkpoint"""
    return torch.load(ckpt_path, map_location='cpu')


def pretrained_state_dir(pretrained_cachedir: str, model_name: str) -> str:
    """Return pretrained state root dir"""
    ver = transformers.__version__
    return f'{pretrained_cachedir}/transformers=={ver}/{model_name}'


def pretrained_tokenizer_state_dir(pretrained_cachedir: str, model_name: str) -> str:
    """Return pretrained transformers tokenizer state dir"""
    return f'{pretrained_state_dir(pretrained_cachedir, model_name)}/tokenizer-fast'


def pretrained_model_state_dir(pretrained_cachedir: str, model_name: str) -> str:
    """Return pretrained transformers model state dir"""
    return f'{pretrained_state_dir(pretrained_cachedir, model_name)}/model'


def pretrained_config_state_dir(pretrained_cachedir: str, model_name: str) -> str:
    """Return pretrained transformers config state dir"""
    return f'{pretrained_state_dir(pretrained_cachedir, model_name)}/config'


def save_pretrained_model(pretrained_cachedir: str, model_name) -> None:
    """
    Save state of pretrained models so that they may be instantiated later without access to huggingface.
    """
    if not os.path.exists(pretrained_state_dir(pretrained_cachedir, model_name)):
        t5 = AutoModel.from_pretrained(model_name)
        t5.save_pretrained(pretrained_model_state_dir(pretrained_cachedir, model_name))
        # torch.save(t5.encoder.embed_tokens,
        #            f'{pretrained_model_state_dir(pretrained_cachedir, model_name)}/encoder.embed_tokens.pt')
        # torch.save(t5.encoder.embed_tokens.weight,
        #            f'{pretrained_model_state_dir(pretrained_cachedir, model_name)}/encoder.embed_tokens.weight.pt')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(pretrained_tokenizer_state_dir(pretrained_cachedir, model_name))

        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(pretrained_config_state_dir(pretrained_cachedir, model_name))


def pretrained_tokenizer(model_name: Optional[str] = None, pretrained_cachedir: Optional[str] = None, **kwargs) -> PreTrainedTokenizerFast:
    """
    Initialize and return a pretrained T5TokenizerFast object given the t5 model config name.

    Parameters
    ----------
        model_name: (str): Huggingface model name 't5-small', 't5-large' etc.
        pretrained_cachedir: (Optional[str]): Path of pretrained cachedir e.g. data/_pretrained_cachedir
        kwargs: Args to be passed on to the 'from_pretrained' method. e.g. return_tensors='np'

    Return
    ------
        T5TokenizerFast object.
    """
    state_dir = None if pretrained_cachedir is None else pretrained_tokenizer_state_dir(pretrained_cachedir, model_name)
    # Default kwargs
    def_kwargs = dict(return_tensors='np')
    def_kwargs.update(kwargs)
    return AutoTokenizer.from_pretrained(state_dir or model_name, use_fast=True, **def_kwargs)


def pretrained_config(model_name: Optional[str] = None, pretrained_cachedir: Optional[str] = None, **kwargs) -> PretrainedConfig:
    """
    Initialize and return a pretrained T5Config object given the t5 model config name.

    Parameters
    ----------
        model_name: (str): Huggingface model name 't5-small', 't5-large' etc.
        pretrained_cachedir: (Optional[str]): Path of pretrained cachedir e.g. data/_pretrained_cachedir
        kwargs: Args to be passed on to the 'from_pretrained' method. e.g.  return_tensors='np', return_dict=True

    Return
    ------
        T5Config object.
    """
    state_dir = None if pretrained_cachedir is None else pretrained_config_state_dir(pretrained_cachedir, model_name)
    # Default kwargs
    def_kwargs = dict(return_tensors='np', return_dict=True)
    def_kwargs.update(kwargs)
    return AutoConfig.from_pretrained(state_dir or model_name, **def_kwargs)


def pretrained_model(model_name: Optional[str] = None, pretrained_cachedir: Optional[str] = None, **kwargs) -> PreTrainedModel:
    """
    Initialize and return a pretrained T5 model object given the t5 model config name.

    Parameters
    ----------
        model_name: (str): Huggingface model name 't5-small', 't5-large' etc.
        pretrained_cachedir: (Optional[str]): Path of pretrained cachedir e.g. data/_pretrained_cachedir
        kwargs: Args to be passed on to the 'from_pretrained' method. e.g. return_dict=True

    Return
    ------
        T5 model object.
    """
    state_dir = None if pretrained_cachedir is None else pretrained_model_state_dir(pretrained_cachedir, model_name)
    # Default kwargs
    def_kwargs = dict(return_dict=True)
    def_kwargs.update(kwargs)
    return AutoModel.from_pretrained(state_dir or model_name, **def_kwargs)


def load_pretrained_embeddings(model_name: Optional[str] = None,
                               pretrained_cachedir: Optional[str] = None,
                               **kwargs) -> torch.nn.Embedding:
    """
    Initialize and return a pretrained Embedding object given the t5 model config name.

    Parameters
    ----------
        model_name: (str): Huggingface model name 't5-small', 't5-large' etc.
        pretrained_cachedir: (Optional[str]): Path of pretrained cachedir e.g. data/_pretrained_cachedir
        kwargs: any extra keyword args for torch.load
    Return
    ------
        nn.Embedding: Input embeddings of the named pretrained model.
    """
    return torch.load(f'{pretrained_model_state_dir(pretrained_cachedir, model_name)}/encoder.embed_tokens.pt',
                      **kwargs)
