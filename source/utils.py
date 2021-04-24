import logging
import sys
import warnings

def disable_warnings():
    """
    Disable printing of warnings

    Returns
    -------
    None
    """
    warnings.simplefilter("ignore")

def get_logger(name):
    """
    Return a logger for current module
    Returns
    -------

    logger : logger instance

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger