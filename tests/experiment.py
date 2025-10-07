from memories.memory_stream import StreamManager

from moviad.datasets.bmad.bmad_dataset import BMAD

from memories.replay_strategy import ReplayModel
from trainers.models import STFPMModel
from trainers.continual_trainer import ContinualTrainer

import wandb

wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")

if __name__ == "__main__":
    continual_dataset = StreamManager(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", random_seed=42)
    # Create model and strategy
    replay_strategy = ReplayModel(
        model_conf={'stfpm': STFPMModel("cuda:0", 'resnet18', ['layer1', 'layer2', 'layer3'])},
        buffer_size=1000
    )
    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=True
        )
    trainer.train(
        continual_dataset,
        epochs_per_task=1
        )

