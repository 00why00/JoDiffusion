from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


dataset_path = "../dataset/COCO"

for split, annotations in zip(["train2017", "val2017"], ["annotations/instances_train2017.json", "annotations/instances_val2017.json"]):
    annotation_path = Path(dataset_path) / annotations
    save_path = Path(dataset_path) / f"annotations/semantic_{split}_80"
    save_path.mkdir(parents=True, exist_ok=True)
    coco = COCO(annotation_path)

    images = coco.loadImgs(coco.getImgIds())
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_contiguous_id = {0: 0}
    category_id_to_contiguous_id.update({k["id"]: i + 1 for i, k in enumerate(categories)})

    for img in tqdm(images):
        img_ann_ids = coco.getAnnIds(imgIds=[img["id"]])
        img_anns = coco.loadAnns(img_ann_ids)
        semantic_mask = np.zeros((img["height"], img["width"]), dtype=np.uint8)

        for ann in img_anns:
            mask = coco.annToMask(ann)
            assert len(np.unique(mask).tolist()) <= 2
            semantic_mask[mask != 0] = category_id_to_contiguous_id[ann['category_id']]
        assert max(np.unique(semantic_mask)) <= 80
        Image.fromarray(semantic_mask).save(save_path / img["file_name"].replace("jpg", "png"))
