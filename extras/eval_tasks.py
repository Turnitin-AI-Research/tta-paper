"""
Task Registry
"""
from lm_eval.tasks import hellaswag, webqs, triviaqa, TASK_REGISTRY

EVAL_TASKS = {
    "lambada": {'context-parts': [], 'hints': []},
    "hellaswag": {'context-parts': [], 'hints': []},
    "webqs":  {'context-parts': [], 'hints': []},
    "triviaqa":  {'context-parts': [], 'hints': []},
    "wsc273":   {'context-parts': [], 'hints': []},
    "winogrande":   {'context-parts': [], 'hints': []},
    "piqa":   {'context-parts': [], 'hints': []},
    "arc_easy":   {'context-parts': [], 'hints': []},
    "arc_challenge":   {'context-parts': [], 'hints': []},
    "openbookqa":   {'context-parts': [], 'hints': []},
    "coqa":   {'context-parts': [], 'hints': []},
    "drop":   {'context-parts': [], 'hints': []},
    "squad2":   {'context-parts': [], 'hints': []},
    "race":   {'context-parts': [], 'hints': []},
    # SuperGLUE
    "wsc":   {'context-parts': [], 'hints': []},
    "cb":   {'context-parts': [], 'hints': []},
    "wic":   {'context-parts': [], 'hints': []},
    "boolq":   {'context-parts': [], 'hints': []},
    "copa":   {'context-parts': [], 'hints': []},
    "rte":   {'context-parts': [], 'hints': []},
    "multirc":   {'context-parts': [], 'hints': []},
    "record":   {'context-parts': [], 'hints': []}
}
