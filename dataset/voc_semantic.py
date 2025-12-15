import json
from pathlib import Path

import datasets
import numpy as np
from PIL import Image


class VOCSemantic(datasets.GeneratorBasedBuilder):
    """CityScapes dataset with captions and annotations."""

    VERSION = datasets.Version("1.0.0")

    @property
    def data_root(self):
        return Path("../dataset/VOC2012")

    @property
    def image_root(self):
        return self.data_root / "JPEGImages"

    @property
    def semantic_root(self):
        return [self.data_root / "SegmentationClass", self.data_root / "SegmentationClassAug"]

    @property
    def category_names(self):
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep','sofa', 'train', 'tvmonitor']

    @property
    def ignore_label(self):
        return 0

    @property
    def num_classes(self):
        return len(self.category_names) - 1

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "semantic": datasets.Image(),
                "category_name_caption": datasets.Value("string"),
                "blip2_caption": datasets.Value("string"),
                "meta": {
                    "file_name": datasets.Value("string"),
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
                    "image_list": self.data_root / "ImageSets/Segmentation/trainaug.txt",
                    "caption_json": self.data_root / "ImageSets/Caption/trainaug.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_list": self.data_root / "ImageSets/Segmentation/val.txt",
                    "caption_json": self.data_root / "ImageSets/Caption/val.json",
                }
            ),
        ]

    def load_semantic(self, image_name):
        semantic_path = [self.semantic_root[0] / f"{image_name}.png", self.semantic_root[1] / f"{image_name}.png"]
        semantic_path = semantic_path[1] if semantic_path[1].exists() else semantic_path[0]
        assert semantic_path.exists(), f"Semantic map not found: {semantic_path}"
        semantic = np.array(Image.open(semantic_path))
        semantic[semantic == 255] = self.ignore_label
        classes = np.unique(semantic)
        classes = classes[classes != self.ignore_label]
        class_names = [self.category_names[c] for c in classes]
        return semantic, class_names

    def _generate_examples(self, image_list, caption_json):
        counter = 0

        blip2_caption = json.load(open(caption_json, "r"))
        for image_name in open(image_list, "r").readlines():
            image_path = self.image_root / f"{image_name.strip()}.jpg"
            _tmp_img = Image.open(image_path)
            sample = {
                "image": str(image_path),
                "meta": {
                    "file_name": image_path.name,
                    "height": _tmp_img.height,
                    "width": _tmp_img.width,
                }
            }
            sample["semantic"], class_names = self.load_semantic(image_name.strip())
            sample["category_name_caption"] = ", ".join(cls.split(",")[0] for cls in class_names)
            sample["blip2_caption"] = blip2_caption[image_name.strip()]
            yield counter, sample
            counter += 1