import torch
import torch.nn as nn
import copy
from utils.utils import get_pretrained_model
from utils.data import DATASET_TO_NUM_CLASSES
from utils.curvature_tuning import replace_module_dynamic, TrainableCTU


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    model_list = ['resnet18', 'resnet50', 'resnet152', 'swin_t', 'swin_s']
    dataset_list = [
        "arabic-characters",
        "arabic-digits",
        "beans",
        "cub200",
        "dtd",
        "fashion-mnist",
        "fgvc-aircraft",
        "flowers102",
        "food101",
        "medmnist/dermamnist",
        "medmnist/octmnist",
        "medmnist/pathmnist",
    ]
    method_list = ['ct']
    seeds = [42, 43, 44]

    count = 0
    for model in model_list:
        pretrained_ds = 'imagenette' if 'swin' in model else 'imagenet'

        for transfer_ds in dataset_list:
            # Check completeness
            for method in method_list:
                for seed in seeds:
                    file_path = f'./ckpts/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.pth'

                    model = get_pretrained_model(pretrained_ds, model)

                    if 'swin' not in model:
                        model.fc = nn.Linear(in_features=model.fc.in_features,
                                             out_features=DATASET_TO_NUM_CLASSES[transfer_ds]).to(device)
                    else:
                        model.head = nn.Linear(in_features=model.head.in_features,
                                               out_features=DATASET_TO_NUM_CLASSES[transfer_ds]).to(device)

                    dummy_input_shape = (1, 3, 224, 224)
                    ct_model = replace_module_dynamic(copy.deepcopy(model), dummy_input_shape, old_module=nn.ReLU,
                                                      new_module=TrainableCTU).to(device)
                    ct_model.load_state_dict(torch.load(file_path, map_location=device))

