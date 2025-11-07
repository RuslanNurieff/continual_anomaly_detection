from memories.memory_stream import StreamManager

from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES

from memories.replay_strategy import ReplayModel
from trainers.models import RD4ADModel
from trainers.continual_trainer import ContinualTrainer
import wandb

import json

wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")

def main():
    continual_dataset = StreamManager(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", categories=list(CATEGORIES))

    replay_strategy = ReplayModel(
        model_conf=RD4ADModel("cuda:0", "wide_resnet50_2", (224, 224)),
        buffer_size=1000
    )

    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=True
        )

    avg_metrics = trainer.train(
        continual_dataset,
        epochs_per_task=50
    )

    with open('/home/ruslan/thesis/tests/file_rd4ad.txt', 'w') as file:
        file.write(json.dumps(avg_metrics))

if __name__ == "__main__":
    main()