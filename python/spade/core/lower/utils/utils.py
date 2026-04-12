from typing import Iterable, List


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def get_num_blocks(seqlen_tuple: Iterable[int],
                   block_size_tuple: Iterable[int]) -> List[int]:
    return [
        ceil_div(seqlen, block)
        for seqlen, block in zip(seqlen_tuple, block_size_tuple)
    ]


# Backwards-compatible names.
def ceilDiv(x, y):
    return ceil_div(x, y)


def getNumBlocks(seqlenTuple, blockSizeTuple):
    return get_num_blocks(seqlenTuple, blockSizeTuple)
