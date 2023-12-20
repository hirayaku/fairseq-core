from fairseq.distributed import utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar  # noqa

# initialize hydra
from fairseq.dataclass.initialize import hydra_init
hydra_init()

import fairseq.distributed  # noqa
import fairseq.models  # noqa
import fairseq.modules  # noqa
# import fairseq.model_parallel  # noqa
