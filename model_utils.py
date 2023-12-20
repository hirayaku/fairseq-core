import os, sys, argparse, re
from enum import Enum
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Optional, Tuple, Any, Dict, List
import torch
import torch.nn as nn
from torch import Tensor
import fairseq.utils as utils
from fairseq.dataclass import FairseqDataclass
from fairseq.data.dictionary import Dictionary, TruncatedDictionary
from fairseq.data.gpt2_bpe import GPT2BPE
from fairseq.file_io import PathManager
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
import fairseq.models
from fairseq.modules.moe import MOELayer

import logging
logger = logging.getLogger(__name__)

def extract_param(model: torch.nn.Module, pattern:Optional[str]=None):
    '''
    extract all parameters whose name matches the regex `pattern`
    examples:
    sum([p.numel() for p in extract_param(model, '\.experts\.')])
    '''
    matched = {}
    for n, m in model.named_parameters():
        if pattern is None or re.search(pattern, n) is not None:
            matched[n] = m
    return matched

@dataclass
class LMTaskConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    model_placement: str = field(
        default='cpu', metadata={"help": "placement policy of the model"}
    )
    bpe: Any = field(
        default=None, metadata={"help": "BPE tokenizer"}
    )
    dictionary: Optional[Dictionary] = field(default=None)
    source_dictionary: Optional[Dictionary] = field(default=None)
    target_dictionary: Optional[Dictionary] = field(default=None)
    output_dict_size: int = field(default=-1, metadata={"help": "translation task"})
    unk_penalty: float = field(default=0.0)

    min_tokens: int = field(
        default=100, metadata={"help": "min number of tokens in prefix and generation"}
    )
    max_tokens: int = field(
        default=120, metadata={"help": "max number of tokens in prefix and generation"}
    )
    topp: Optional[float] = field(
        default=None, metadata={"help": "topp value for nucleus sampling"}
    )
    topk: Optional[int] = field(
        default=8, metadata={"help": "topk value for topk sampling"}
    )
    beam_size: Optional[int] = field(
        default=None, metadata={"help": "beam value for beam search"}
    )
    temperature: float = field(default=1)

def setup_task(args: argparse.Namespace) -> LMTaskConfig:
    '''
    setup a customized simple task object for our case (fairseq MoE models)
    '''
    task = LMTaskConfig(data=args.data, bpe=GPT2BPE(), model_placement=args.placement)
    # load dict.txt into `task` from the given path
    dictionary = None
    paths = utils.split_paths(task.data)
    assert len(paths) > 0
    dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
    logger.info("dictionary: {} types".format(len(dictionary)))
    output_dictionary = dictionary
    if task.output_dict_size >= 0:
        # hth NOTE: what is output_dictionary used for?
        # ans: it's used for machine translation tasks.
        # in conventional langauge modeling tasks they are the same
        output_dictionary = TruncatedDictionary(
            dictionary, task.output_dict_size
        )
    task.dictionary = dictionary
    task.source_dictionary = dictionary
    task.target_dictionary = output_dictionary
    task.topk = args.topk
    task.temperature = args.temperature
    return task

def load_pretrain(
    args: argparse.Namespace,
    cfg_overrides: dict = None
) -> Tuple[fairseq.models.BaseFairseqModel, DictConfig, LMTaskConfig]:
    filename = args.path
    # hth: what does num_shards do?
    num_shards = 1
    if cfg_overrides is None:
        cfg_overrides = {'checkpoint_activations': False}
    is_moe = not args.dense
    rank = 0
    task = setup_task(args)

    for shard_idx in range(num_shards):
        if is_moe:
            suffix = f'-rank-{rank}'
            filename = filename.replace(".pt", suffix + ".pt")
        # else:
        #     filename = filename[:-3] + f"_part{shard_idx}.pt"
        if not PathManager.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        # hth TODO: we might want to override the expert capacity as well for inference
        state = load_checkpoint_to_cpu(filename, cfg_overrides, is_moe=is_moe)
        # hth NOTE: cfg.model keeps the model architecture and (hyper-)parameters
        # state['model'] keeps the model weights
        cfg = state["cfg"]
        model: torch.nn.Module = fairseq.models.build_model(cfg.model, task)
        model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    
    fp16 = cfg.model.get('fp16', False)
    if not fp16:
        for param in state['model'].values():
            if param.dtype == torch.float16:
                fp16 = True
                break
    if fp16:
        model = model.half()

    return model, cfg, task

def map_moe_layer(model: torch.nn.Module, fn):
    for n, m in model.named_modules():
        if isinstance(m, MOELayer):
            yield fn(n, m)

class Placement(Enum):
    CPU = 0
    CUDA = 1
    MIXED = 2

    @classmethod
    def create(cls, policy: str):
        return cls[policy.upper()]

def place(model: torch.nn.Module, policy: Placement, cache_cfg:Optional[dict]=None) -> torch.nn.Module:
    '''
    policies:
    * CPU: everything on cpu
    * CUDA: everything on cuda
    * MIXED: shared weights on cuda, expert weights on CPU's pin_memory
    \t* 'cache_cfg': the ratio of cached experts on GPU
    '''
    if policy == Placement.CPU:
        return model.cpu()
    elif policy == Placement.CUDA:
        return model.cuda()
    elif policy == Placement.MIXED:
        model = place(model, Placement.CPU)
        model = model._apply(lambda p: p.pin_memory() if getattr(p, 'expert', False) else p.cuda())
        # special handling of MOELayer
        for n, m in model.named_modules():
            if isinstance(m, MOELayer):
                m.cpu_offloaded = True
                m.init_expert_cache_(cache_cfg)
        return model
    else:
        raise ValueError(f"Unknown placement policy: {policy}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--dense', action='store_true', help='load dense model')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--placement', type=str, default='cuda',
                        help='placement of the model: cuda, cpu or mixed')
    parser.add_argument('-c', '--cache-size', type=float, default=0.1,
                        help='expert cache ratio for each MoE layer')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--topk', type=int, default=8, help="topk sampling")
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    if args.data is None:
        args.data = os.path.dirname(args.path)
    args.placement = Placement.create(args.placement)
    return args

class LanguageModelWrapper(nn.Module):
    """A wrapper around a language model with (possible) encoder and (necessary) decoder"""

    def __init__(self, model: fairseq.models.BaseFairseqModel):
        super().__init__()
        self.model = model
        assert hasattr(model, "decoder")
        self.has_incremental: bool = hasattr(model, "decoder") and \
            isinstance(model.decoder, fairseq.models.FairseqIncrementalDecoder)

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return self.model.max_decoder_positions()

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return self.model.encoder.forward_torchscript(net_input)

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_out: Dict[str, List[Tensor]],
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        temperature: float = 1.0,
    ):
        model = self.model
        if self.has_incremental_states():
            decoder_out = model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        else:
            decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]["attn"]
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )

        probs = model.get_normalized_probs(
            decoder_out_tuple, log_probs=True, sample=None
        )
        probs = probs[:, -1, :]
        return probs, attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_out: Optional[Dict[str, List[Tensor]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if not self.has_encoder():
            return None
        assert encoder_out is not None
        return self.model.encoder.reorder_encoder_out(encoder_out, new_order)

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order,
    ):
        '''
        Reorders the kv cache so that it matches the sequence order in the current iteration.

        When the token generation terminates for some sequences, we should exclude their
        incremental states from the kv cache by excluding their indices from `new_order`.
        '''
        if not self.has_incremental_states():
            return
        self.model.decoder.reorder_incremental_state_scripting(
            incremental_state, new_order
        )
 