from utils import get_args, merge_args_with_config, load_config
from  train import  train_swa,train_base,train_kd

if __name__ == "__main__":
    args = get_args()
    if args.config:
        cfg = load_config(args.config)
        config = merge_args_with_config(args, cfg)
    else:
        config = vars(args)
    if config.get("train_method") in ["base","basels"]:
        train_base(config)
    elif config.get("train_method")in ["swa","swals"]:
        train_swa(config)
    elif config.get("train_method")in ["kd","kdwa"]:
        train_kd(config)
    else:
        raise NotImplementedError
