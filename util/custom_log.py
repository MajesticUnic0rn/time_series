import logging
import logging.config
import codecs
import yaml

CONFIG_FILE="../config/log_config.yml" # replace config file with constant

def get_custom_logger():

    logger = logging.getLogger()
    logger.info("config file: %s", CONFIG_FILE)
    # We use codecs.open because it is equivalent to Python 3 open()
    with codecs.open(CONFIG_FILE, "r", encoding="utf-8") as fd:
        log_config = yaml.full_load(fd.read())
    logging.config.dictConfig(log_config)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    ## add green for good stuff
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)