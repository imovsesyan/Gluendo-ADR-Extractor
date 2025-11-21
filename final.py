import os
import json
import cv2
import csv
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import easyocr

# === CONFIG ===
INPUT_DIR            = r"C:/Pi/stuff"
OUTPUT_DIR           = r"C:/Pi/desktop"
ANNOTATION_PATH      = r"C:/Pi/annotations.json"
MODEL_WEIGHTS        = r"C:/Pi/model_final.pth"
SCORE_THRESH         = 0.1    # lower to catch faint arrows
OCR_CONF_THRESH      = 0.1    # lower to catch light text
ARROW_CLUSTER_DIST   = 150    # cluster head/body/tail within px

# load class names
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    ann = json.load(f)
categories  = sorted(ann["categories"], key=lambda x: x["id"])
class_names = [c["name"] for c in categories]

# split into arrow vs box labels
arrow_labels = [n for n in class_names if "_arrow_" in n]
box_labels   = [n for n in class_names if n.endswith("_box")]

# setup detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES      = len(class_names)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.MODEL.WEIGHTS                    = MODEL_WEIGHTS
predictor = DefaultPredictor(cfg)

# setup OCR
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = boxA[2] * boxA[3]
    return interArea / areaA if areaA>0 else 0

def get_center(box):
    return (box[0] + box[2]/2, box[1] + box[3]/2)

def cluster_arrow_parts(parts):
    groups = []
    used = set()
    for i,p in enumerate(parts):
        if i in used: continue
        grp = [p]
        for j,q in enumerate(parts):
            if j<=i or q["type"]!=p["type"]: continue
            d = np.linalg.norm(np.array(p["center"]) - np.array(q["center"]))
            if d < ARROW_CLUSTER_DIST:
                grp.append(q)
                used.add(j)
        groups.append(grp)
    return groups

def process_image(img_path):
    name,_ = os.path.splitext(os.path.basename(img_path))
    img = cv2.imread(img_path)
    if img is None:
        print(f"[❌ ERROR] Cannot load {img_path}")
        return

    # Detectron2 inference
    outputs = predictor(img)
    inst    = outputs["instances"].to("cpu")
    boxes   = inst.pred_boxes.tensor.numpy()
    scores  = inst.scores.numpy()
    classes = inst.pred_classes.numpy()

    arrow_parts    = []
    box_detections = []
    for box,score,cls in zip(boxes,scores,classes):
        if score < SCORE_THRESH: continue
        x1,y1,x2,y2 = box
        w,h = x2-x1, y2-y1
        label = class_names[cls]
        bbox  = [int(x1),int(y1),int(w),int(h)]
        ctr   = get_center(bbox)
        if label in box_labels:
            box_detections.append({"label":label,"bbox":bbox})
        elif label in arrow_labels:
            base,part = label.split("_arrow_")
            arrow_parts.append({
                "type":base,
                "part":part,
                "bbox":bbox,
                "center":ctr
            })

    print(f"[INFO] {len(arrow_parts)} arrow parts, {len(box_detections)} boxes")

    # cluster into arrows
    arrow_groups = cluster_arrow_parts(arrow_parts)

    # OCR
    ocr_res = reader.readtext(img)
    words = []
    for pts,text,conf in ocr_res:
        if conf < OCR_CONF_THRESH: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x,y = int(min(xs)), int(min(ys))
        w,h = int(max(xs)-x), int(max(ys)-y)
        cx,cy = x+w/2, y+h/2
        words.append({"text":text.strip(),"bbox":[x,y,w,h],"center":[cx,cy]})
    print(f"[INFO] {len(words)} OCR words")

    # match words→boxes
    word_box_map = {}
    for w in words:
        best_i, best_lbl = 0,None
        for bd in box_detections:
            i = compute_iou(w["bbox"], bd["bbox"])
            if i>best_i:
                best_i, best_lbl = i, bd["label"]
        if best_lbl:
            word_box_map[w["text"]] = best_lbl.replace("_box","")

    # build relations with fallbacks
    relations = []
    for grp in arrow_groups:
        parts      = {p["part"]:p for p in grp}
        arrow_type = grp[0]["type"]
        head = parts.get("head")
        tail = parts.get("tail")
        body = parts.get("body")

        # normal
        start_pt = tail["center"] if tail else None
        end_pt   = head["center"] if head else None

        # A) body→head if no tail
        if not start_pt and body and head:
            start_pt = body["center"]

        # B) head-only fallback
        if head and not (tail or body):
            cand = min(box_detections,
                       key=lambda b: np.linalg.norm(np.array(get_center(b["bbox"]))-np.array(head["center"])))
            start_pt = get_center(cand["bbox"])

        # C) body-only fallback
        if body and not head and not tail:
            start_pt = end_pt = body["center"]

        # skip incomplete
        if not start_pt or not end_pt:
            continue

        # find nearest words
        sw = min(words, key=lambda w: np.linalg.norm(np.array(w["center"])-np.array(start_pt)))
        ew = min(words, key=lambda w: np.linalg.norm(np.array(w["center"])-np.array(end_pt)))

        if sw and ew:
            st,et = sw["text"], ew["text"]
            stype = word_box_map.get(st,"unknown")
            etype = word_box_map.get(et,"unknown")
            relations.append([st,stype,arrow_type,et,etype])
            print(f"[📡 RELATION] {st} ({stype}) → {et} ({etype}) via {arrow_type}")

    # export CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,f"{name}_words.csv"),"w",newline="",encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["text","x","y","w","h"])
        for w in words:
            wr.writerow([w["text"]]+w["bbox"])

    with open(os.path.join(OUTPUT_DIR,f"{name}_relations.csv"),"w",newline="",encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source_text","source_type","relation_type","target_text","target_type"])
        wr.writerows(relations)

    print(f"[✅ DONE] {name}: {len(relations)} relations\n")

if __name__=="__main__":
    for fn in os.listdir(INPUT_DIR):
        if fn.lower().endswith((".png",".jpg",".jpeg")):
            print(f"Processing {fn}…")
            process_image(os.path.join(INPUT_DIR,fn))
