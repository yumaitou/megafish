import importlib
import dask

if importlib.util.find_spec("cupy") is not None:
    USE_GPU = True
else:
    USE_GPU = False


def set_resource(gpu=True, scheduler=None):
    """
    Sets the resource configuration for GPU usage and concurrent future scheduling.

    Args:
        gpu (bool, optional): Whether to use the GPU. This option is only effective in a CUDA environment. Defaults to True.
        scheduler (str or None, optional): The type of scheduler to use as a string. If None, uses the current scheduler. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If an invalid scheduler type is provided.
    """
    use_gpu(gpu)
    if scheduler is not None:
        set_scheduler(scheduler)


def use_gpu(usage):
    """
    Configures the usage of GPU for processing.
    This function reloads the modules to reflect the changes in the resource configuration.

    Args:
        usage (bool): A flag indicating whether to enable GPU usage. 

    Returns:
        None

    """
    global USE_GPU
    USE_GPU = usage

    from . import decode, process, register, segment, seqfish, seqif, view

    importlib.reload(decode)
    importlib.reload(process)
    importlib.reload(register)
    importlib.reload(segment)
    importlib.reload(seqfish)
    importlib.reload(seqif)
    importlib.reload(view)


def set_scheduler(scheduler):
    """
    Sets the scheduler for Dask.

    Args:
        scheduler (str): The type of scheduler to use (e.g., 'threads', 'processes', etc.).

    Returns:
        None

    Raises:
        ValueError: If the provided scheduler type is invalid.
    """
    dask.config.set(scheduler=scheduler)


def show_resource():
    """
    Shows the current resource configuration.

    This function is inserted in the processing functions to show the current resource configuration.

    Returns:
        str: A string representation of the current resource configuration.

    """
    output = " ["
    if USE_GPU:
        output += "GPU/"
    else:
        output += "CPU/"
    output += str(dask.config.get("scheduler")) + "]"

    return output
