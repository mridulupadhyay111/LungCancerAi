from datasets.lung_dataset import LungDataset
from transforms.ct_transforms import Resize3D


dataset=LungDataset(

    root_dir=
    r"D:\LungCancerAI\data\processed",

    transform=Resize3D(),

    use_masks=True

)


sample=dataset[0]


print(sample["patient_id"])

print(
    "Image:",
    sample["image"].shape
)


print(
    "Mask:",
    sample["mask"].shape
)