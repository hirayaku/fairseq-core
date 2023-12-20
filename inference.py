import os, sys, math
from typing import List, Dict, Optional, Any
import torch
from torch import Tensor
from fairseq.search import Search, Sampling, BeamSearch
from fairseq.utils import move_to_cuda
from model_utils import (
    get_args, load_pretrain, place,
    LMTaskConfig, LanguageModelWrapper
)
import logging
logger = logging.getLogger(__name__)

def make_search(task: LMTaskConfig) -> Search:
    if task.beam_size:
        logger.info(f"Using beam search with size={task.beam_size}")
        logger.error("Beam search not supported yet")
    elif task.topp:
        logger.info(f"Using nucleus sampling with topp={task.topp}")
        search = Sampling(task.target_dictionary, sampling_topp=task.topp)
    elif task.topk:
        logger.info(f"Using topk sampling with k={task.topk}")
        search = Sampling(task.target_dictionary, sampling_topk=task.topk)
    return search

def make_batch(sequences: List[str], task: LMTaskConfig):
    '''
    encode input sequences into tokens and assemble a batch from them
    '''
    tokens = [
        task.source_dictionary.encode_line(
            task.bpe.encode(src_str), add_if_not_exist=False, append_eos=False,
        ).long()
        for src_str in sequences
    ]
    lengths = [t.numel() for t in tokens]
    src_tokens = torch.zeros((len(tokens), task.max_tokens)).to(tokens[0])
    src_tokens.fill_(task.dictionary.pad())
    prefix_tokens = torch.zeros((len(tokens), max(lengths))).to(tokens[0])
    prefix_tokens.fill_(task.dictionary.pad())
    for i, tokens_i in enumerate(tokens):
        size_i = tokens_i.size(0)
        src_tokens[i][:size_i] = tokens_i[:]
        prefix_tokens[i][:size_i] = tokens_i[:]
    src_lengths = torch.tensor(lengths)
    return {
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prefix": prefix_tokens,
        },
    }

def decode_batch(generation: List[Dict[str, Tensor]], task: LMTaskConfig):
    outputs = [
        task.bpe.decode(task.target_dictionary.string(output['tokens']))
        for output in generation
    ]
    return outputs
    

def generate_incremental(
    task: LMTaskConfig,
    model: LanguageModelWrapper,
    sample: dict,
    prefix_tokens=None,
):
    '''
    A simplified version of incremental generation:
    * no beam search
    * no search constraints as the result
    * batch size = 1 (?)

    Generate tokens based on `sample` inputs incrementally.
    Requires the `model.decoder` to be `FairseqIncrementalDecoder`
    '''

    incremental_states = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})

    sample = move_to_cuda(sample)
    net_input = sample["net_input"]
    if prefix_tokens is None and 'prefix' in net_input:
        prefix_tokens = net_input['prefix']
    bos, eos = task.dictionary.bos(), task.dictionary.eos()
    pad, unk = task.dictionary.pad(), task.dictionary.unk()
    vocab_size = len(task.dictionary)

    def _prefix_tokens(
        step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    if "src_tokens" in net_input:
        src_tokens = net_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = net_input.get("src_lengths",
            (src_tokens.ne(eos) & src_tokens.ne(pad)).long().sum(dim=1)
        )
    elif "source" in net_input:
        src_tokens = net_input["source"]
        src_lengths = (
            net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
            if net_input["padding_mask"] is not None
            else torch.tensor(src_tokens.size(-1)).to(src_tokens)
        )
    else:
        raise Exception("expected src_tokens or source in net input")

    # bsz: total number of sentences in beam
    # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
    bsz, src_len = src_tokens.size()[:2]
    assert task.beam_size is None, "beam search not supported"
    search = make_search(task)
    beam_size = 1

    max_len: int = task.max_tokens
    min_len: int = task.min_tokens
    pre_len: int = src_lengths.min().item()

    # hth NOTE: none for decoder only models including fairseq MoE
    encoder_out = model.forward_encoder(net_input)
    # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
    new_order = new_order.to(src_tokens.device).long()
    encoder_out = model.reorder_encoder_out(encoder_out, new_order)

    # initialize buffers
    scores = (
        torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
    )  # +1 for eos; pad is never chosen for scoring
    tokens = (
        torch.zeros(bsz * beam_size, max_len + 2)
        .to(src_tokens)
        .long()
        .fill_(pad)
    )  # +2 for eos and pad
    tokens[:, 0] = eos  # hth: fairseq adds eos to the begining of sentence. weird.
    attn: Optional[Tensor] = None

    # A list that indicates candidates that should be ignored.
    # For example, suppose we're sampling and have already finalized 2/5
    # samples. Then cands_to_ignore would mark 2 positions as being ignored,
    # so that we only finalize the remaining 3 samples.
    cands_to_ignore = (
        torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
    )  # forward and backward-compatible False mask

    # list of completed sentences
    finalized = torch.jit.annotate(
        List[Dict[str, Tensor]],
        [torch.jit.annotate(Dict[str, Tensor], {}) for i in range(bsz)],
    )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

    finished = [
        False for i in range(bsz)
    ]  # a boolean array indicating if the sentence at the index is finished or not
    num_remaining_sent = bsz  # number of sentences remaining

    # # number of candidate hypos per step
    # cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

    # offset arrays for converting between different indexing schemes
    bbsz_offsets = (
        (torch.arange(0, bsz) * beam_size)
        .unsqueeze(1)
        .type_as(tokens)
        .to(src_tokens.device)
    )
    # cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

    reorder_state: Optional[Tensor] = None
    reorder_state_last: Optional[Tensor] = None
    batch_idxs: Optional[Tensor] = None

    original_batch_idxs: Optional[Tensor] = None
    if "id" in sample and isinstance(sample["id"], Tensor):
        original_batch_idxs = sample["id"]
    else:
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

    for step in range(max_len + 1):  # one extra step for EOS marker
        logger.info(f"Step {step}: batch size = {bsz}")
        # reorder decoder internal states based on the prev choice of beams
        if reorder_state is not None:
            if batch_idxs is not None:
                # hth NOTE: track idx in the original batch
                original_batch_idxs = original_batch_idxs[batch_idxs]
            if reorder_state_last is None or not torch.equal(reorder_state, reorder_state_last):
                reorder_state_last = reorder_state
                model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_out = model.reorder_encoder_out(
                    encoder_out, reorder_state
                )

        # hth: what is avg_attn_scores? what is it used for?
        lprobs, avg_attn_scores = model.forward_decoder(
            tokens[:, : step + 1],
            encoder_out,
            incremental_states,
            task.temperature,
        )

        # hth: what is this?
        # if self.lm_model is not None:
        #     lm_out = self.lm_model(tokens[:, : step + 1])
        #     probs = self.lm_model.get_normalized_probs(
        #         lm_out, log_probs=True, sample=None
        #     )
        #     probs = probs[:, -1, :] * self.lm_weight
        #     lprobs += probs

        # remove nan
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
        lprobs[:, pad] = -math.inf  # never select pad
        lprobs[:, unk] -= task.unk_penalty  # apply unk penalty

        # handle max length constraint
        if step >= max_len:
            lprobs[:, : eos] = -math.inf
            lprobs[:, eos + 1 :] = -math.inf

        # handle prefix tokens (possibly with different lengths)
        if (
            prefix_tokens is not None
            and step < prefix_tokens.size(1)
            and step < max_len
        ):
            lprobs, tokens, scores = _prefix_tokens(
                step, lprobs, scores, tokens, prefix_tokens, beam_size
            )

        if step < min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            lprobs[:, eos] = -math.inf

        # Record attention scores, only support avg_attn_scores is a Tensor
        if avg_attn_scores is not None:
            if attn is None:
                attn = torch.empty(
                    bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                ).to(scores)
            attn[:, :, step + 1].copy_(avg_attn_scores)

        scores = scores.type_as(lprobs)
        eos_bbsz_idx = torch.empty(0).to(
            tokens
        )  # indices of hypothesis ending with eos (finished sentences)
        eos_scores = torch.empty(0).to(
            scores
        )  # scores of hypothesis ending with eos (finished sentences)

        # if self.should_set_src_lengths:
        #     self.search.set_src_lengths(src_lengths)
        # if self.repeat_ngram_blocker is not None:
        #     lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

        # Shape: (batch, cand_size)
        # hth NOTE: for non beam search, cand_size = 1
        cand_scores, cand_indices, cand_beams = search.step(
            step,
            lprobs.view(bsz, -1, vocab_size),
            scores.view(bsz, beam_size, -1)[:, :, :step],
            tokens[:, : step + 1],
            original_batch_idxs,    # hth: useless for topk/topp
        )

        # cand_bbsz_idx contains beam indices for the top candidate
        # hypotheses, with a range of values: [0, bsz*beam_size),
        # and dimensions: [bsz, cand_size]
        cand_bbsz_idx = cand_beams.add(bbsz_offsets)

        # finalize hypotheses that end in eos
        # Shape of eos_mask: (batch size, beam size)
        eos_mask = cand_indices.eq(eos) & cand_scores.ne(-math.inf)
        eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

        # only consider eos when it's among the top beam_size indices
        # Now we know what beam item(s) to finish
        # Shape: 1d list of absolute-numbered
        # hth NOTE: this is the index of all finished sequences, flattened
        eos_bbsz_idx = torch.masked_select(
            cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
        )
        active_bbsz_idx = torch.masked_select(
            cand_bbsz_idx[:, :beam_size], mask=~eos_mask[:, :beam_size]
        )

        finalized_sents: List[int] = []
        if eos_bbsz_idx.numel() > 0:
            eos_scores = torch.masked_select(
                cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
            )
            finalized_sents = eos_bbsz_idx.tolist()
            tokens_clone = tokens.index_select(0, eos_bbsz_idx)[:, 1:step+1]
            for i, sent in enumerate(finalized_sents):
                # hth TODO: attach lm scores and attention scores
                finalized[original_batch_idxs[sent]] = {"tokens": tokens_clone[i].cpu()}
            num_remaining_sent -= len(finalized_sents)

        assert num_remaining_sent >= 0
        if num_remaining_sent == 0:
            break
        if search.stop_on_max_len and step >= max_len:
            break
        assert step < max_len, f"{step} < {max_len}"

        # Remove finalized sentences (ones for which {beam_size}
        # finished hypotheses have been generated) from the batch.
        if len(finalized_sents) > 0:
            new_bsz = bsz - len(finalized_sents)

            # construct batch_idxs which holds indices of batches to keep for the next pass
            batch_mask = torch.ones(
                bsz, dtype=torch.bool, device=cand_indices.device
            )
            batch_mask[eos_bbsz_idx] = False
            batch_idxs = torch.arange(
                bsz, device=cand_indices.device
            ).masked_select(batch_mask)

            eos_mask = eos_mask[batch_idxs]
            cand_beams = cand_beams[batch_idxs]
            bbsz_offsets.resize_(new_bsz, 1)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            cand_scores = cand_scores[batch_idxs]
            cand_indices = cand_indices[batch_idxs]

            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[batch_idxs]
            src_lengths = src_lengths[batch_idxs]
            cands_to_ignore = cands_to_ignore[batch_idxs]

            scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            bsz = new_bsz
        else:
            batch_idxs = None

        # Select the next token for each of them
        tokens.view(bsz, beam_size, -1)[:, :, step + 1] = cand_indices
        # reorder incremental state in decoder
        reorder_state = active_bbsz_idx

    return finalized

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout
    )
    args = get_args()

    model, cfg, task = load_pretrain(args)
    model = model.eval()
    model = place(model, args.placement)
    model.prepare_for_inference_(cfg)

    with torch.no_grad():
        sequences = [
            "Boston is a harbor city in the northeast of United States",
            "I love to walk my dog during the afternoon after I leave my work",
            "The railway station was built in the last century and has been",
        ]
        sample = make_batch(sequences, task)
        wrapped_model = LanguageModelWrapper(model)
        generation = generate_incremental(
            task, wrapped_model, sample,
        )
        print(decode_batch(generation, task))
