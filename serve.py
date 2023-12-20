import os, sys, fileinput
import torch
from model_utils import (
    get_args, map_moe_layer, load_pretrain, place, LanguageModelWrapper
)
from inference import make_batch, decode_batch, generate_incremental
import logging
logger = logging.getLogger(__name__)

def get_input(bsz=1):
    sequences = [
        "Boston is a harbor city in the northeast of United States. It's known for its higher education",
        "The railway station was built in the last century and has been",
        # "Paris is the capital and most populous city of France. With an official estimated population of over 2 million residents in an area of more than 105 km2 (41 sq mi), Paris is the fifth-most populated city in the",
    ]
    idx = 0
    seq_len = len(sequences)
    while idx < seq_len:
        yield sequences[idx:min(idx+bsz, seq_len)]
        idx += bsz
    buffers = []
    logger.info("Type more inputs from stdin:")
    with fileinput.input(files="-", encoding="utf-8") as h:
        for src_str in h:
            buffers.append(src_str.strip())
            if len(buffers) >= bsz:
                yield buffers
        if len(buffers) > 0:
            yield buffers

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
    model = place(model, args.placement, cache_cfg={
        'size': args.cache_size,
        'init': True,
    })
    model.prepare_for_inference_(cfg)   # will invoke model.eval()
    wrapped_model = LanguageModelWrapper(model)

    from viztracer import VizTracer
    seq_no = 0
    for sequences in get_input(args.batch_size):
        with torch.no_grad():
            sample = make_batch(sequences, task)
            with VizTracer(
                tracer_entries=int(5e6), max_stack_depth=16,
                ignore_frozen=True,
                output_file=f"result.{seq_no}.json",
            ):
                generation = generate_incremental(
                    task, wrapped_model, sample,
                )
            cache_list = list(map_moe_layer(model, lambda n, m: m.cache))
            cache_stats = [(c._hits, c._lookups) for c in cache_list]
            hits, lookups = list(zip(*cache_stats))
            logger.info(f"hits={hits}, {lookups[0]} lookups, hit rate {sum(hits)/sum(lookups)*100:.2f}%")
            print(decode_batch(generation, task))
        seq_no += 1
