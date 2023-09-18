import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from ensemble_boxes import *


def load_json(filename):
    """
    json 文件的加载
    """
    file = open(filename, 'r')
    json_dict = json.load(file)
    file.close()
    return json_dict


def generate_image_list(image_dir):
    image_list = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    image_list = sorted(image_list, key=lambda a: int(os.path.basename(a.split(".")[0])))
    return image_list


def dump_json(save_path, json_dict):
    file = open(save_path, "w")
    json.dump(json_dict, file)
    file.close()


def im_index(json_dict, image_list, threshold=0.01):
    """
    将json转化成以图片为索引的列表，方便检测框的融合
    """
    image_dict = {}
    # print(len(json_dict), len(image_list))
    for idx, obj in enumerate(tqdm(json_dict)):
        if obj["score"] < threshold:
            continue
        if obj["image_id"] not in image_dict.keys():
            # image = cv2.imread(image_list[obj["image_id"]])
            image = Image.open(image_list[obj["image_id"]]).convert('RGB')
            image = np.asarray(image, dtype=np.uint8)

            w, h = image.shape[:2]
            image_dict[obj["image_id"]] = {"bbox": [], "score": [], "category_id": [], "shape": [h, w]}
        image_dict[obj["image_id"]]["bbox"].append([
            obj["bbox"][0] / h,
            obj["bbox"][1] / w,
            (obj["bbox"][0] + obj["bbox"][2]) / h,
            (obj["bbox"][1] + obj["bbox"][3]) / w,
        ])
        image_dict[obj["image_id"]]["score"].append(obj["score"])
        image_dict[obj["image_id"]]["category_id"].append(obj["category_id"])
    return image_dict


def generate_category_id():
    return [
        {
            "supercategory": "none", 
            "id": 1, 
            "name": "aeroplane"
        }, 
        {
            "supercategory": "none", 
            "id": 2, 
            "name": "bicycle"
        }, 
        {
            "supercategory": "none", 
            "id": 3, 
            "name": "boat"
        }, 
        {
            "supercategory": "none", 
            "id": 4, 
            "name": "bus"
        }, 
        {
            "supercategory": "none", 
            "id": 5, 
            "name": "car"
        }, 
        {
            "supercategory": "none", 
            "id": 6, 
            "name": "chair"
        }, 
        {
            "supercategory": "none", 
            "id": 7, 
            "name": "diningtable"
        }, 
        {
            "supercategory": "none", 
            "id": 8, 
            "name": "motorbike"
        }, 
        {
            "supercategory": "none", 
            "id": 9, 
            "name": "sofa"
        }, 
        {
            "supercategory": "none", 
            "id": 10, 
            "name": "train"
        }
    ]


def split_json(im_dict, pigeonhole):
    pigeonhole = load_json(pigeonhole)
    new_pigeonhole = {}
    for key, value in pigeonhole.items():
        new_pigeonhole[key] = {}
        for item in value:
            ids = int(item.split(".")[0])
            new_pigeonhole[key][ids] = im_dict[ids]
    return new_pigeonhole


def im2coco(im_dict, image_path_list, threshold=0.01):
    obj_dict = []
    image_dict = []
    # ids = 1
    print(len(image_path_list), len(im_dict))
    for im_key, im_value in im_dict.items():
        image_dict.append({
            "file_name": os.path.basename(image_path_list[im_key]),
            "height": im_value["shape"][1],
            "width": im_value["shape"][0],
            "id": im_key, 
        })
        for i in range(len(im_value["category_id"])):
            if im_value["score"][i] >= threshold:
                # h = (im_value["bbox"][i][2] - im_value["bbox"][i][0]) * im_value["shape"][0]
                # w = (im_value["bbox"][i][3] - im_value["bbox"][i][1]) * im_value["shape"][1]
                obj = {
                    # "image_id": im_key,
                    "image_id": os.path.basename(image_path_list[im_key]).split(".")[0],
                    "category_id": im_value["category_id"][i],
                    "bbox": [
                        im_value["bbox"][i][0] * im_value["shape"][0],
                        im_value["bbox"][i][1] * im_value["shape"][1],
                        (im_value["bbox"][i][2] - im_value["bbox"][i][0]) * im_value["shape"][0],
                        (im_value["bbox"][i][3] - im_value["bbox"][i][1]) * im_value["shape"][1]
                    ],
                    "score": im_value["score"][i],
                    # "iscrowd": 0,
                    # "id": ids,
                    # "segmentation": [],
                    # "area": int(h * w)
                }
                obj_dict.append(obj)
    return obj_dict
    # return {
    #     "images": image_dict,
    #     "annotations": obj_dict,
    #     "categories": generate_category_id()
    # }


def wbf(dir_path, save_path, bbox_paths, image_dir, wbf_bool=False, threshold=0.01):
    if not os.path.exists(os.path.join(dir_path, save_path)):
        os.makedirs(os.path.join(dir_path, save_path))

    image_results = {}
    image_lists = []
    results_list = []
    image_path_list = generate_image_list(image_dir)

    # json 2 index
    for bbox_path in bbox_paths:
        image_dict = im_index(load_json(os.path.join(bbox_path, "bbox.json")), image_path_list, threshold)
        image_list = list(image_dict.keys())
        image_lists += image_list
        results_list.append(image_dict)
    
    image_lists = sorted(list(set(image_lists)))

    if wbf_bool:
        for i in image_lists:
            bbox, score, category_id = [], [], []
            for results in results_list:
                if i in results.keys():
                    bbox.append(results[i]["bbox"])
                    score.append(results[i]["score"])
                    category_id.append(results[i]["category_id"])
            bbox, score, category_id = weighted_boxes_fusion(bbox, score, category_id, iou_thr=0.8, weights=[1, 1, 1, 1, 2, 2, 2, 2], skip_box_thr=0.0001)
            # bbox, score, category_id = weighted_boxes_fusion(bbox, score, category_id, weights=[2, 3, 3, 2], iou_thr=0.775, skip_box_thr=0.001)
            image_results[i] = {
                "bbox": bbox.tolist(),
                "score": score.tolist(),
                "category_id": category_id.tolist(),
                "shape": results[i]["shape"]
            }
    else:
        image_results = results_list[0]
    results = split_json(image_results, "nuisance2filenames.json")
    for key, value in results.items():
        results_coco = im2coco(value, image_path_list, 0)
        dump_json(os.path.join(dir_path, save_path, key+".json"), results_coco)


if __name__ == "__main__":
    # coco 投票
    config_path = [
        "configs/OOD/PPYoloEPlus/main_448.yml",
        "configs/OOD/PPYoloEPlus/main_512.yml",
    ]
    # base_list = [
    #     "output_final/ppyoloe_plus-semi-base/main_512",
    #     "output_final/ppyoloe_plus-semi-base/main_448",
    #     "output_final/strong_ppyoloe_plus/main_448",
    #     "output_final/strong_ppyoloe_plus/main_512",
    #     "output_final/strong_ppyoloe_plus_semi_128/main_448",
    #     "output_final/strong_ppyoloe_plus_semi_128/main_512",
    #     "output_final/strong_ppyoloe_plus_semi_96/main_448",
    #     "output_final/strong_ppyoloe_plus_semi_96/main_512",
    # ]
    dir_path = "output_final"
    save_path = "end_model-unWBF"
    model_list = [
        "strong_ppyoloe_plus_semi_128",
        "strong_ppyoloe_plus_semi_96",
        "ppyoloe_plus-semi-base",
        "strong_ppyoloe_plus"
    ]
    image_dir = "dataset/coco/test_images/images"
    for cfg in config_path:
        for model in model_list:
            os.system("python tools/infer.py -c %s --infer_dir=%s -o weights=model/%s/best_model.pdparams --visualize=False --save_results=True --output_dir %s" % (cfg, image_dir, model, os.path.join(dir_path, model, os.path.basename(cfg).split(".")[0])))
        base_list.append(os.path.join(dir_path, model, os.path.basename(cfg).split(".")[0]))
    wbf(
        dir_path,
        save_path,
        base_list,
        image_dir,
        wbf_bool=True,
        threshold=0.,
    )

