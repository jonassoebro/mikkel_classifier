import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data.data import get_data#, download_data
from src.models.engine import EngineModule
from src.models.trainer import get_trainer
#from project_1_1.src.utils import print_class_dist

wandb.init(project='mkl_classifier', entity='jonassoebro')

@hydra.main(config_path='../../config', config_name="config")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    #ownload_data(cfg.data.path)
    train_dataloader, val_dataloader = get_data(cfg.training.batch_size)
    #print_class_dist(train_dataloader, title='Train set'), print_class_dist(test_dataloader, title='Test set')
    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    

if __name__ == '__main__':
    run_training()
