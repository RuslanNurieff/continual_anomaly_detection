from memories.memory_stream import StreamManager

from moviad.datasets.bmad.bmad_dataset import BMAD

from memories.replay_strategy import ReplayModel
from trainers.models import STFPMModel
from trainers.continual_trainer import ContinualTrainer
import wandb

import json

# wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")

def main():
    continual_dataset = StreamManager(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", random_seed=21)

    replay_strategy = ReplayModel(
        model_conf=STFPMModel("cuda:0", 'wide_resnet50_2', ['layer1', 'layer2', 'layer3']),
        buffer_size=1000
    )

    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=False
        )

    avg_metrics = trainer.train(
        continual_dataset,
        epochs_per_task=50
    )

    with open('/home/ruslan/thesis/tests/file.txt', 'w') as file:
        file.write(json.dumps(avg_metrics))

if __name__ == "__main__":
    main()