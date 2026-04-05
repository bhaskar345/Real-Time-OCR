import cv2
import numpy as np
import MNN
from sklearn.cluster import DBSCAN
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ------------------ LOAD MNN MODELS ------------------

# YOLO
yolo_interpreter = MNN.Interpreter("best.mnn")
yolo_session = yolo_interpreter.createSession({
    "backend": "CPU",
    "precision": "low",
    "numThread": 4
})
yolo_input = yolo_interpreter.getSessionInput(yolo_session)

# CRNN
rec_interpreter = MNN.Interpreter("rec_crnn.mnn")
rec_session = rec_interpreter.createSession({
    "backend": "CPU",
    "precision": "low",
    "numThread": 4
})
rec_input = rec_interpreter.getSessionInput(rec_session)

# Charset
with open("ppocr_keys.txt", "r", encoding="utf-8") as f:
    CHARS = [""] + [line.strip() for line in f.readlines()]

# ------------------ TRACKING ------------------

tracked_objects = {}
frame_count = 0
MAX_OCR = 3
OCR_INTERVAL = 2 # FPS/2
next_track_id = 0

# ------------------ IOU ------------------

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2-x1)*(y2-y1) + (x2g-x1g)*(y2g-y1g) - inter

    return inter/union if union > 0 else 0

# ------------------ TRACK MATCH ------------------

def match_boxes(detections):
    global tracked_objects, frame_count, next_track_id

    new_tracks = {}
    used_ids = set()

    for det in detections:
        best_iou = 0
        best_id = None

        for obj_id, obj in tracked_objects.items():
            score = iou(det, obj["bbox"])
            if score > best_iou:
                best_iou = score
                best_id = obj_id

        if best_iou > 0.5 and best_id not in used_ids:
            new_tracks[best_id] = {
                "bbox": det,
                "text": tracked_objects[best_id]["text"],
                "count": tracked_objects[best_id].get("count", 0),
                "last_seen": frame_count
            }
            used_ids.add(best_id)
        else:
            new_tracks[next_track_id] = {
                "bbox": det,
                "text": "",
                "count": 0,
                "last_seen": frame_count
            }
            next_track_id += 1

    tracked_objects.clear()
    tracked_objects.update(new_tracks)

# ------------------ YOLO ------------------

def letterbox(img, new_size=416):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)

    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top+nh, left:left+nw] = img_resized

    return canvas, scale, left, top

def preprocess(frame):
    img, scale, pad_x, pad_y = letterbox(frame, 416)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, scale, pad_x, pad_y

def run_yolo_mnn(inp):
    inp = np.ascontiguousarray(inp)

    # Resize
    yolo_interpreter.resizeTensor(yolo_input, (1, 3, 416, 416))
    yolo_interpreter.resizeSession(yolo_session)

    tmp = MNN.Tensor(
        (1, 3, 416, 416),
        MNN.Halide_Type_Float,
        inp,
        MNN.Tensor_DimensionType_Caffe
    )

    yolo_input.copyFrom(tmp)
    yolo_interpreter.runSession(yolo_session)

    output = yolo_interpreter.getSessionOutput(yolo_session)

    out = np.array(output.getData(), dtype=np.float32)
    out = out.reshape(output.getShape())

    return [out]

def postprocess(outputs, scale, pad_x, pad_y, orig_shape):
    h, w = orig_shape

    outputs = np.squeeze(outputs)

    xs = outputs[0]
    ys = outputs[1]
    ws = outputs[2]
    hs = outputs[3]
    confs = outputs[4]

    boxes = []

    for i in range(len(confs)):

        conf = float(confs[i])

        if conf < 0.2:
            continue

        x = xs[i]
        y = ys[i]
        w_box = ws[i]
        h_box = hs[i]

        x1 = x - w_box / 2
        y1 = y - h_box / 2
        x2 = x + w_box / 2
        y2 = y + h_box / 2

        x1 = int((x1 - pad_x) / scale)
        y1 = int((y1 - pad_y) / scale)
        x2 = int((x2 - pad_x) / scale)
        y2 = int((y2 - pad_y) / scale)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2, conf])

    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    final = []
    while boxes:
        best = boxes.pop(0)
        final.append(best)
        boxes = [
            b for b in boxes
            if iou(best[:4], b[:4]) < 0.4
        ]

    return [b[:4] for b in final]

# ------------------ OCR ------------------

def preprocess_paddle(img, max_w=320):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h = 48
    ratio = img.shape[1] / img.shape[0]
    new_w = int(h * ratio)
    new_w = min(new_w, max_w)

    img = cv2.resize(img, (new_w, h))

    img = img.astype(np.float32) / 255.0
    img = img * 2 - 1

    padded = np.zeros((h, max_w, 3), dtype=np.float32)
    padded[:, :new_w, :] = img

    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)

    return padded

def decode_paddle(preds, conf_thresh=0.5):
    preds_prob = np.max(preds, axis=2)[0]
    preds_idx = np.argmax(preds, axis=2)[0]

    text = ""
    prev = 0
    valid = []

    for i, idx in enumerate(preds_idx):
        prob = preds_prob[i]

        if idx != prev and idx != 0 and idx < len(CHARS):
            if prob > conf_thresh:
                text += CHARS[idx]
                valid.append(prob)

        prev = idx

    # reject weak OCR
    if len(valid) == 0 or np.mean(valid) < 0.6:
        return ""

    return text

def run_crnn_mnn(batch):
    batch = np.ascontiguousarray(batch)

    # RESIZE SESSION INPUT
    rec_interpreter.resizeTensor(rec_input, batch.shape)
    rec_interpreter.resizeSession(rec_session)

    tmp = MNN.Tensor(
        batch.shape,
        MNN.Halide_Type_Float,
        batch,
        MNN.Tensor_DimensionType_Caffe
    )

    rec_input.copyFrom(tmp)
    rec_interpreter.runSession(rec_session)

    output = rec_interpreter.getSessionOutput(rec_session)

    shape = output.getShape()
    data = np.array(output.getData(), dtype=np.float32)

    if len(data) == 0:
        print("EMPTY OCR OUTPUT")
        return np.zeros((1, 1, len(CHARS)))

    out = data.reshape(shape)

    # (B, C, T) → (B, T, C)
    if out.shape[1] == len(CHARS):
        out = np.transpose(out, (0, 2, 1))

    return out

def run_ocr_batch(crops):
    if not crops:
        return []

    batch = np.zeros((len(crops), 3, 48, 160), dtype=np.float32)

    for i, crop in enumerate(crops):
        batch[i] = preprocess_paddle(crop, 160)[0]

    preds = run_crnn_mnn(batch)

    return [decode_paddle(preds[i:i+1]) for i in range(len(crops))]

# ------------------ GROUPING ------------------

def group_text_dbscan(objects):
    if not objects:
        return []

    # center Y for line clustering
    ys = np.array([[(o["bbox"][1] + o["bbox"][3]) / 2] for o in objects])

    # adaptive eps
    heights = [o["bbox"][3]-o["bbox"][1] for o in objects]
    eps = np.mean(heights) * 0.8

    # cluster into lines
    clustering = DBSCAN(eps=eps, min_samples=1).fit(ys)

    lines = {}
    for label, obj in zip(clustering.labels_, objects):
        lines.setdefault(label, []).append(obj)

    # sort lines top → bottom
    sorted_lines = sorted(lines.values(), key=lambda l: np.mean([(o["bbox"][1]+o["bbox"][3])/2 for o in l]))

    results = []

    for line in sorted_lines:
        line.sort(key=lambda x: x["bbox"][0])

        text = " ".join([o["text"] for o in line])
        bboxes = [o["bbox"] for o in line]

        results.append({
            "text": text,
            "bboxes": bboxes
        })

    return results

def merge_lines_to_blocks(objects):
    if not objects:
        return []

    for obj in objects:
        x1,y1,x2,y2 = obj["bbox"]
        obj["cx"] = (x1 + x2) / 2
        obj["cy"] = (y1 + y2) / 2
        obj["h"] = y2 - y1
        obj["w"] = x2 - x1

    objects.sort(key=lambda x: x["cy"])

    blocks = []
    current_block = [objects[0]]

    for obj in objects[1:]:
        prev = current_block[-1]

        vgap = obj["bbox"][1] - prev["bbox"][3]
        x_align = abs(obj["cx"] - prev["cx"])

        # stricter conditions
        same_block = (
            vgap < 0.8 * prev["h"] and
            x_align < 0.8 * prev["h"] and
            abs(prev["w"] - obj["w"]) < 0.7 * max(prev["w"], obj["w"])
        )

        if same_block:
            current_block.append(obj)
        else:
            blocks.append(current_block)
            current_block = [obj]

    blocks.append(current_block)

    final_blocks = []

    for block in blocks:
        block.sort(key=lambda x: x["cy"])

        text = " ".join([o["text"] for o in block])
        bboxes = [o["bbox"] for o in block]

        final_blocks.append({
            "text": text,
            "bboxes": bboxes
        })

    return final_blocks

def get_text(objects):
    blocks = merge_lines_to_blocks(objects)

    if len(blocks) > 5:
        return group_text_dbscan(objects)

    return blocks

# ------------------ WEBSOCKET ------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global frame_count
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()

            data = msg.get("bytes", None)
            if data is None:
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1
            h, w, _ = frame.shape

            inp, scale, pad_x, pad_y = preprocess(frame)
            outputs = run_yolo_mnn(inp)
            detections = postprocess(outputs, scale, pad_x, pad_y, (h, w))

            match_boxes(detections)

            crops, ids = [], []

            for obj_id, obj in tracked_objects.items():
                x1, y1, x2, y2 = obj["bbox"]

                if (x2 - x1) < 40 or (y2 - y1) < 20:
                    continue

                if obj["text"] == "" or frame_count % OCR_INTERVAL == 0:
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        crops.append(crop)
                        ids.append(obj_id)

            if crops:
                texts = run_ocr_batch(crops[:MAX_OCR])

                for i, obj_id in enumerate(ids[:MAX_OCR]):
                    # SAME TEXT → increase stability
                    new_text = texts[i].strip()
                    obj = tracked_objects[obj_id]
                    if obj["text"] == new_text:
                        obj["count"] += 1
                    else:
                        obj["text"] = new_text
                        obj["count"] = 1

            objects = []
            for obj in tracked_objects.values():
                if obj["count"] >= 2 and obj["text"].strip() != "":
                    objects.append({
                        "bbox": obj["bbox"],
                        "text": obj["text"]
                    })

            grouped_texts = get_text(objects)
            await websocket.send_json({
                "words": objects,
                "texts": grouped_texts
            })
    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("Error:", str(e))

# ------------------ API ------------------

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"detections": []}

        h, w, _ = frame.shape

        inp, scale, pad_x, pad_y = preprocess(frame)
        outputs = run_yolo_mnn(inp)
        detections = postprocess(outputs, scale, pad_x, pad_y, (h, w))

        crops, boxes = [], []

        for x1, y1, x2, y2 in detections:
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                crops.append(crop)
                boxes.append((x1, y1, x2, y2))

        texts = run_ocr_batch(crops) if crops else []

        objects = []
        for i, bbox in enumerate(boxes):
            txt = texts[i] if i < len(texts) else ""
            if txt.strip() != "":
                objects.append({
                    "bbox": bbox,
                    "text": txt.strip()
                })

        grouped_texts = get_text(objects)

        return {
            "words": objects,
            "texts": grouped_texts
        }   
    except Exception as e:
        print("Error:", str(e))

# ------------------ HOME ------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------ RUN ------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)