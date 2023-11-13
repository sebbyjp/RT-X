import io
import torch
from torch import nn, optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from rtx import RTX2
import datasets
from pprint import pprint
from torch.utils.data import DataLoader
from PIL import Image


def describe(dic, prefix=""):
    """Useful to print out the structure of TF Record. ds.info can also be used
        but it does not show lengths of lists and dicts.

    Args:
        dic (dict): Input
        prefix (str, optional): Prefix used for nested indentation. Defaults to "".
    """
    if not isinstance(dic, dict):
        return

    def describe_img(img: bytes):
        img = Image.open(io.BytesIO(img))
        return f"{img.__class__.__name__} sz: { img.size}"

    for k, v in dic.items():
        if isinstance(v, list):
            list_type = ""
            if len(v) > 0:
                v_description = ""
                if isinstance(v[0], torch.Tensor):
                    v_description = f"({tuple(v[0].size())}, {v[0].dtype})"
                elif isinstance(v[0], bytes):
                    v_description = describe_img(v[0])
                list_type = f"({v[0].__class__.__name__ }{v_description})"
            print(f"{prefix} {k}, {v.__class__.__name__}{list_type} sz: {len(v)}")
            if len(v) > 0:
                describe(v[0], prefix + "  ")
        elif isinstance(v, dict):
            print(f"{prefix} {k}, {v.__class__.__name__} sz: {len(v.items())}")
            describe(v, prefix + "  ")
        elif isinstance(v, bytes):
          print(f'{prefix} {k}, {describe_img( v)}')
        elif isinstance(v, str):
            print(f"{prefix} {k}, {v.__class__.__name__} v: {v}")

        else:
            tensor_type = ""
            if isinstance(v, torch.Tensor):
                tensor_type = f"({tuple(v[0].size())}, {v[0].dtype})"
            print(f"{prefix} {k}, {v.__class__.__name__} {tensor_type} ")


def remove_nonetypes(dic: any):
    """Remove nonetypes from a dict and returns it back. If Input is not a dict, it is returned as is.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    """
    if not isinstance(dic, dict):
        return dic
    to_remove = []
    for k, v in dic.items():
        if isinstance(v, list):
            for vv in v:
                remove_nonetypes(vv)
        elif v is None:
            to_remove.append(k)
        else:
            remove_nonetypes(v)
    for k in to_remove:
        del dic[k]

    return dic

def format_imgs(dic: any, sz: int):
    """Resizes images to sz as a numpy array.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    """
    if isinstance(dic, bytes):
        img = Image.open(io.BytesIO(dic))
        return np.array(img.resize(img, (sz,sz)))

    if not isinstance(dic, dict):
        return dic

    for k, v in dic.items():
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = format_imgs(v, sz)
        else:
            dic[k] = format_imgs(v, sz)
    return dic


ds = datasets.load_dataset(
    "jxu124/OpenX-Embodiment",
    "bridge",
    split="train",
    cache_dir="datasets_cache",
    streaming=True,  # Comment this out to save dataset to disk.
)  # IterDataset


# pprint(ds.info)
# describe(next(iter(ds)))

loader = DataLoader(
    ds.map(lambda x: remove_nonetypes(x["data.pickle"]), 256), batch_size=1, num_workers=0
)

for batch in loader:
    describe(batch)
    break


model = RTX2()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(10):
    running_loss = 0.0
    for i, batch in enumerate(loader):
        inputs = batch["image"]
        labels = batch["label"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/10:.3f}")
            running_loss = 0.0
    scheduler.step()


# usage
# img = torch.randn(1, 3, 256, 256)
# text = torch.randint(0, 20000, (1, 1024))

# model = RTX2()
# output = model(img, text)
# print(output)
