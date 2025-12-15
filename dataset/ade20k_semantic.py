import json
from pathlib import Path

import datasets
import numpy as np
from PIL import Image


class ADE20KSemantic(datasets.GeneratorBasedBuilder):
    """ADE20K dataset with captions and annotations."""

    VERSION = datasets.Version("1.0.0")

    @property
    def data_root(self):
        return Path("../dataset/ADE20K")

    @property
    def object_info(self):
        info = open(self.data_root / "objectInfo150.txt", "r").readlines()
        idx, _, _, _, name = zip(*[line.strip().split("\t") for line in info[1:]])
        return [{"id": int(i), "name": n.split(',')[0]} for i, n in zip(idx, name)]

    @property
    def image_info(self):
        info = json.load(open(self.data_root / "imgCatIds.json", "r"))["images"]
        return {i["file_name"]: i for i in info}

    @property
    def category_names(self):
        return ["background"] + [k["name"] for k in self.object_info]

    @property
    def ignore_label(self):
        return 0

    @property
    def num_classes(self):
        return len(self.object_info)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "semantic": datasets.Image(),
                "category_name_caption": datasets.Value("string"),
                "blip2_caption": datasets.Value("string"),
                "meta": {
                    "file_name": datasets.Value("string"),
                    "image_id": datasets.Value("string"),
                    "height": datasets.Value("int32"),
                    "width": datasets.Value("int32"),
                }
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_path": dl_manager.iter_files(str(self.data_root / "images/training")),
                    "semantic_path": dl_manager.iter_files(str(self.data_root / "annotations/training")),
                    "caption_json": self.data_root / "annotations_caption/training.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_path": dl_manager.iter_files(str(self.data_root / "images/validation")),
                    "semantic_path": dl_manager.iter_files(str(self.data_root / "annotations/validation")),
                    "caption_json": self.data_root / "annotations_caption/validation.json",
                }
            ),
        ]

    def load_semantic(self, semantic_path):
        semantic = np.array(Image.open(semantic_path))
        classes = np.unique(semantic)
        classes = classes[classes != self.ignore_label]
        class_names = [self.category_names[c] for c in classes]
        return semantic, class_names

    def _generate_examples(self, image_path, semantic_path, caption_json):
        counter = 0

        blip2_caption = json.load(open(caption_json, "r"))
        for image, semantic in zip(image_path, semantic_path):
            sample = {
                "image": image,
                "meta": {
                    "file_name": Path(image).name,
                    "image_id": self.image_info[Path(image).name]["id"],
                    "height": self.image_info[Path(image).name]["height"],
                    "width": self.image_info[Path(image).name]["width"],
                }
            }
            sample["semantic"], class_names = self.load_semantic(semantic)
            sample["category_name_caption"] = ", ".join(cls.split(",")[0] for cls in class_names)
            sample["blip2_caption"] = blip2_caption[Path(image).name]
            yield counter, sample
            counter += 1