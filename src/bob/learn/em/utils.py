import logging

import dask
import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def check_and_persist_dask_input(data, persist=True):
    # check if input is a dask array. If so, persist and rebalance data
    input_is_dask = False
    if isinstance(data, da.Array):
        if persist:
            data: da.Array = data.persist()
        input_is_dask = True
        # if there is a dask distributed client, rebalance data
        try:
            client = dask.distributed.Client.current()
            client.rebalance()
        except ValueError:
            pass

    else:
        data = np.asarray(data)
    return input_is_dask, data


def array_to_delayed_list(data, input_is_dask):
    # If input is a dask array, convert to delayed chunks
    if input_is_dask:
        data = data.to_delayed().ravel().tolist()
        logger.debug(f"Got {len(data)} chunks.")
    return data
