import hydra
from omegaconf import DictConfig

from smartphone_classifier.train import train


@hydra.main(config_path="configs", config_name="hydra", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)

#test
if __name__ == "__main__":
    main()
