from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.models.draem.augmentation import DRAEMTrain
bmad = BMAD("segmentation", "/mnt/disk1/ruslan_nuriev/bmad", "liver", "train")
draem = DRAEMTrain(bmad, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", resize_shape=[224, 224])

import torch
adasd = torch.utils.data.DataLoader(draem, batch_size=32)

from moviad.models.draem.draem import DRAEM
from moviad.trainers.trainer_draem import TrainerDRAEM

model = DRAEM("cuda:0")
trainer = TrainerDRAEM(model, adasd, adasd, "cuda:0")
trainer.train(5)