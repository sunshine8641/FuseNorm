from .train_base import  train_baseline


def train_swa(config):
    assert config["use_swa"] ==True
    assert config["swa_window"]>1
    assert config["swa_start"]>=0
    train_baseline(config)