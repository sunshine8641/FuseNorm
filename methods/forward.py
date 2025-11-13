import torch
import os
from .get_threshold import compute_threshold_react
from .get_threshold import compute_activation_stats
from accelerate import Accelerator

def forward_base(inputs,model,config):
    features = model.forward_features(inputs)
    outputs = model.forward_head(features)
    return features,outputs

def forward_all_features(inputs,model,config):
    features = model.forward_features_blockwise(inputs)
    return features,None



def forward_react(inputs,model,config):
    features = model.forward_features(inputs)
    if config['react_threshold'] is None:
        raise ValueError("react_threshold cannot be None")
    react_threshold=config['react_threshold']
    features = torch.where(features < react_threshold, features, react_threshold)
    logits = model.forward_head(features)
    return features,logits



def get_forward(name="base",config=None):
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    if name == "base":
        return forward_base
    elif name == "react":

        load_path = os.path.join(model_save_dir, config["test_model"], "react_threshold.pt")
        if not os.path.exists(load_path):
            compute_threshold_react(config)

        config['react_threshold']= torch.load(load_path)
        return forward_react
    elif name == "bats":
        load_path = os.path.join(model_save_dir, config["test_model"] ,"feature_stats.pt")
        if not os.path.exists(load_path):
            compute_activation_stats(config)

        feature_stats=torch.load(load_path)
        config['feature_std']=   feature_stats['feature_std']
        config['feature_mean']= feature_stats['feature_mean']
        config['feature_std']=config['feature_std'].to(config["accelerator"].device)
        config['feature_mean']=config['feature_mean'].to(config["accelerator"].device)
        return forward_bats
    elif name == "laps":
        if config.get('feature_std',None) is None:
            load_path = os.path.join(model_save_dir, config["test_model"],"feature_stats.pt")
            if not os.path.exists(load_path):
                compute_activation_stats(config)
            feature_stats = torch.load(load_path)
            config['feature_std'] = feature_stats['feature_std']
            config['feature_mean'] = feature_stats['feature_mean']
            config['feature_std'] = config['feature_std'].to(config["accelerator"].device)
            config['feature_mean'] = config['feature_mean'].to(config["accelerator"].device)
        lam = config['laps_lam']
        feature_mean = config['feature_mean']
        feature_std = config['feature_std']
        config['laps_lam1'] = lam + (torch.mean(feature_mean) - feature_mean) * config['laps_m'] \
               + (torch.mean(feature_std) - feature_std) * config['laps_n']
        config['laps_lam2'] = lam - (torch.mean(feature_mean) - feature_mean) * config['laps_m'] \
               + (torch.mean(feature_std) - feature_std) * config['laps_n']
        return forward_laps
    elif name == "forward_features":
        return forward_all_features
    else:
        raise ValueError(f"Unknown forward method {name}")




if __name__ == "__main__":

    pass