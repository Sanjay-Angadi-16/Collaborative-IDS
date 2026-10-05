import logging
from pathlib import Path

def setup_logging(log_dir='logs', level='INFO'):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('ids')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh=logging.FileHandler(Path(log_dir)/'ids.log', encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
