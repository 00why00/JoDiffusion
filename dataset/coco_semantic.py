import json
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
from pycocotools.coco import COCO


class COCOSemantic(datasets.GeneratorBasedBuilder):
    """COCO dataset with captions and annotations."""

    VERSION = datasets.Version("1.0.0")

    @property
    def data_root(self):
        return Path("../dataset/COCO")

    @property
    def category_info(self):
        return [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
            {"id": 4, "name": "motorcycle"},
            {"id": 5, "name": "airplane"},
            {"id": 6, "name": "bus"},
            {"id": 7, "name": "train"},
            {"id": 8, "name": "truck"},
            {"id": 9, "name": "boat"},
            {"id": 10, "name": "traffic light"},
            {"id": 11, "name": "fire hydrant"},
            {"id": 13, "name": "stop sign"},
            {"id": 14, "name": "parking meter"},
            {"id": 15, "name": "bench"},
            {"id": 16, "name": "bird"},
            {"id": 17, "name": "cat"},
            {"id": 18, "name": "dog"},
            {"id": 19, "name": "horse"},
            {"id": 20, "name": "sheep"},
            {"id": 21, "name": "cow"},
            {"id": 22, "name": "elephant"},
            {"id": 23, "name": "bear"},
            {"id": 24, "name": "zebra"},
            {"id": 25, "name": "giraffe"},
            {"id": 27, "name": "backpack"},
            {"id": 28, "name": "umbrella"},
            {"id": 31, "name": "handbag"},
            {"id": 32, "name": "tie"},
            {"id": 33, "name": "suitcase"},
            {"id": 34, "name": "frisbee"},
            {"id": 35, "name": "skis"},
            {"id": 36, "name": "snowboard"},
            {"id": 37, "name": "sports ball"},
            {"id": 38, "name": "kite"},
            {"id": 39, "name": "baseball bat"},
            {"id": 40, "name": "baseball glove"},
            {"id": 41, "name": "skateboard"},
            {"id": 42, "name": "surfboard"},
            {"id": 43, "name": "tennis racket"},
            {"id": 44, "name": "bottle"},
            {"id": 46, "name": "wine glass"},
            {"id": 47, "name": "cup"},
            {"id": 48, "name": "fork"},
            {"id": 49, "name": "knife"},
            {"id": 50, "name": "spoon"},
            {"id": 51, "name": "bowl"},
            {"id": 52, "name": "banana"},
            {"id": 53, "name": "apple"},
            {"id": 54, "name": "sandwich"},
            {"id": 55, "name": "orange"},
            {"id": 56, "name": "broccoli"},
            {"id": 57, "name": "carrot"},
            {"id": 58, "name": "hot dog"},
            {"id": 59, "name": "pizza"},
            {"id": 60, "name": "donut"},
            {"id": 61, "name": "cake"},
            {"id": 62, "name": "chair"},
            {"id": 63, "name": "couch"},
            {"id": 64, "name": "potted plant"},
            {"id": 65, "name": "bed"},
            {"id": 67, "name": "dining table"},
            {"id": 70, "name": "toilet"},
            {"id": 72, "name": "tv"},
            {"id": 73, "name": "laptop"},
            {"id": 74, "name": "mouse"},
            {"id": 75, "name": "remote"},
            {"id": 76, "name": "keyboard"},
            {"id": 77, "name": "cell phone"},
            {"id": 78, "name": "microwave"},
            {"id": 79, "name": "oven"},
            {"id": 80, "name": "toaster"},
            {"id": 81, "name": "sink"},
            {"id": 82, "name": "refrigerator"},
            {"id": 84, "name": "book"},
            {"id": 85, "name": "clock"},
            {"id": 86, "name": "vase"},
            {"id": 87, "name": "scissors"},
            {"id": 88, "name": "teddy bear"},
            {"id": 89, "name": "hair drier"},
            {"id": 90, "name": "toothbrush"},
        ]

    @property
    def category_id_to_contiguous_id(self):
        return {k["id"]: i for i, k in enumerate(self.category_info)}

    @property
    def category_names(self):
        return [k["name"] for k in self.category_info]

    @property
    def ignore_label(self):
        return 0

    @property
    def num_classes(self):
        return len(self.category_info) - 1

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "semantic": datasets.Image(),
                "category_name_caption": datasets.Value("string"),
                "coco_caption": [datasets.Value("string")],
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
                    "image_path": self.data_root / "train2017",
                    "annotations": self.data_root / "annotations/instances_train2017.json",
                    "caption_json": self.data_root / "annotations/captions_train2017.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_path": self.data_root / "val2017",
                    "annotations": self.data_root / "annotations/instances_val2017.json",
                    "caption_json": self.data_root / "annotations/captions_val2017.json",
                }
            ),
        ]

    def _generate_examples(self, image_path, annotations, caption_json):
        counter = 0

        coco = COCO(annotations)
        captions = json.load(open(caption_json, "r"))
        image_info = {i["file_name"]: i for i in captions["images"]}
        caption_dict = defaultdict(list)
        for cap in captions["annotations"]:
            caption_dict[cap['image_id']].append(cap['caption'])
        for file_path in image_path.glob("*.jpg"):
            sample = {
                "image": str(file_path),
                "coco_caption": caption_dict[image_info[file_path.name]["id"]],
                "meta": {
                    "file_name": file_path.name,
                    "image_id": image_info[file_path.name]["id"],
                    "height": image_info[file_path.name]["height"],
                    "width": image_info[file_path.name]["width"],
                }
            }

            # parse semantic segmentation labels
            img_ann_ids = coco.getAnnIds(imgIds=[sample["meta"]["image_id"]])
            img_anns = coco.loadAnns(img_ann_ids)
            class_names = []
            if len(img_anns) == 0:
                sample["semantic"] = np.zeros((sample["meta"]["height"], sample["meta"]["width"]), dtype=np.uint8)
            else:
                semantic_mask = np.zeros((sample["meta"]["height"], sample["meta"]["width"]), dtype=np.uint8)
                for ann in img_anns:
                    mask = coco.annToMask(ann)
                    assert len(np.unique(mask).tolist()) <= 2
                    semantic_mask[mask != 0] = self.category_id_to_contiguous_id[ann['category_id']]
                    class_names.append(self.category_names[self.category_id_to_contiguous_id[ann['category_id']]])
                assert max(np.unique(semantic_mask)) <= 80
                sample["semantic"] = semantic_mask

            sample["category_name_caption"] = ", ".join(cls.split(",")[0] for cls in class_names)
            yield counter, sample
            counter += 1
