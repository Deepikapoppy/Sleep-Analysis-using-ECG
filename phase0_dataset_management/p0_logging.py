import os, sys, logging
from datetime import datetime

def setup_logger(name, log_dir, log_file="pipeline.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s  [%(levelname)-8s]  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def log_phase_header(logger, phase, dataset, record):
    logger.info("=" * 60)
    logger.info(f"  {phase}  |  Dataset: {dataset.upper()}  |  Record: {record}")
    logger.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

def setup_environment(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    log_path = os.path.join(config["output_dir"], config.get("log_file", "pipeline.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger()
    log_phase_header(logger, "SLEEP ECG PIPELINE",
                     config.get("dataset","slpdb"),
                     str(config.get("record_name","unknown")))
    return logger