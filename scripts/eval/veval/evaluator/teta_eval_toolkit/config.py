# fmt: off
# flake8: noqa

"""Config."""
import argparse
import os


def parse_configs():
    """Parse command line."""
    default_eval_config = get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_dataset_config = get_default_dataset_config()
    default_metrics_config = {"METRICS": ["TETA"]}
    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        f"Command line parameter {setting} must be True/False"
                    )
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }

    return eval_config, dataset_config, metrics_config


def get_default_eval_config():
    """Returns the default config values for evaluation."""
    code_path = get_code_path()
    default_config = {
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": os.path.join(code_path, "error_log.txt"),
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": True,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_TEM_RAW_DATA": True,
        "OUTPUT_PER_SEQ_RES": True,
    }
    return default_config


def get_default_dataset_config():
    """Default class config values"""
    code_path = get_code_path()
    default_config = {
        "GT_FOLDER": os.path.join(
            code_path, "data/gt/tao/tao_training"
        ),  # Location of GT data
        "TRACKERS_FOLDER": os.path.join(
            code_path, "data/trackers/tao/tao_training"
        ),  # Trackers location
        "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        "TRACKERS_TO_EVAL": ['TETer'],  # Filenames of trackers to eval (if None, all in folder)
        "CLASSES_TO_EVAL": None,  # Classes to eval (if None, all classes)
        "SPLIT_TO_EVAL": "training",  # Valid: 'training', 'val'
        "PRINT_CONFIG": True,  # Whether to print current config
        "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        "MAX_DETECTIONS": 0,  # Number of maximal allowed detections per image (0 for unlimited)
        "USE_MASK": False,  # Whether to use mask data for evaluation
    }
    return default_config


def init_config(config, default_config, name=None):
    """Initialize non-given config values with defaults."""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config["PRINT_CONFIG"]:
        print("\n%s Config:" % name)
        for c in config.keys():
            print("%-20s : %-30s" % (c, config[c]))
    return config


def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        "Command line parameter " + setting + "must be True or False"
                    )
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
