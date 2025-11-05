import torch
import torch.nn as nn

def safe_setattr(obj, attr, value, mode=None):
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    else:
        if mode == 'overwrite':
            setattr(obj, attr, value)
        elif mode == 'ignore':
            pass
        else:
            raise ValueError(f'Attribute {attr} already exists in {obj}!')


def foward_wrapper(orig_forward, note: str):

    def forward(input: torch.Tensor, *args, **kwargs):
        print(note, input.shape)
        return orig_forward(input, *args, **kwargs)

    return forward
