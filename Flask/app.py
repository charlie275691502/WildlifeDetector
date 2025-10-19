from flask import Flask, request, jsonify
from PIL import Image
import torch
import numpy as np
import tempfile, os
from yolov5 import YOLOv5  # if you're using yolov5 module (not ultralytics)

import pathlib
from pathlib import Path

#RF-DETR imports from rfdetr and supervision
try:
    from rfdetr import RFDETRBase
    import supervision as sv
    RF_DETR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RF-DETR dependencies not available: {e}")
    RF_DETR_AVAILABLE = False

# Monkey-patch PosixPath on Windows
if hasattr(pathlib, 'PosixPath'):
    pathlib.PosixPath = Path

app = Flask(__name__)
# Resolve weights paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
YOLOV5_WEIGHTS = BASE_DIR / "YOLOv5_best.pt"  # renamed from best.pt
if not YOLOV5_WEIGHTS.exists():
    # Fallback to original name for backward compatibility
    YOLOV5_WEIGHTS = BASE_DIR / "best.pt"
FASTER_RCNN_WEIGHTS = BASE_DIR / "FasterRCNN.pth"

# Resolve YOLOv8 weights with fallbacks
def _find_yolov8_weights() -> Path:
    candidates = [
        BASE_DIR / "yolov8.pth",
        BASE_DIR / "yolov8.pt",
    ]
    # Also consider any yolov8*.pt / .pth files if present
    try:
        for p in sorted(BASE_DIR.glob("yolov8*.pt*")):
            if p not in candidates:
                candidates.append(p)
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return p
    # If none found, return the preferred default (will be validated later)
    return candidates[0]

YOLOV8_WEIGHTS = _find_yolov8_weights()
RF_DETR_WEIGHTS = BASE_DIR / "RF-DETR.pth"

# Load YOLOv5 once
yolov5_model = YOLOv5(str(YOLOV5_WEIGHTS), device="cpu")  # change to 'cuda' if available

_faster_rcnn_model = None
_yolov8_model = None
_rf_detr_model = None

# Optional label map for Faster-RCNN dataset
# Edit this dict to map numeric label ids to human-readable names
FASTER_RCNN_LABEL_MAP = {
    1: "Koala",
    # 2: "OtherClass",
}

# RF-DETR class names (matching your working example)
RF_DETR_CLASS_NAMES = ["Koala", "Non-koala"]

def get_faster_rcnn_model():
    """Load Faster-RCNN model lazily and cache it in memory."""
    global _faster_rcnn_model
    if _faster_rcnn_model is not None:
        return _faster_rcnn_model
    # Try loading either a full model or a state_dict
    weights_obj = torch.load(str(FASTER_RCNN_WEIGHTS), map_location="cpu")
    try:
        # Case 1: full model object saved via torch.save(model)
        if hasattr(weights_obj, 'eval'):
            model = weights_obj
            model.eval()
            _faster_rcnn_model = model
            return _faster_rcnn_model

        # Case 2: checkpoint dict possibly with 'model_state_dict'
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        state_dict = weights_obj
        if isinstance(weights_obj, dict) and 'model_state_dict' in weights_obj:
            state_dict = weights_obj['model_state_dict']

        # Try to infer num_classes from predictor shapes to avoid size mismatch
        inferred_num_classes = None
        cls_w = state_dict.get('roi_heads.box_predictor.cls_score.weight')
        bbox_w = state_dict.get('roi_heads.box_predictor.bbox_pred.weight')
        if cls_w is not None:
            inferred_num_classes = cls_w.shape[0]
        elif bbox_w is not None:
            # bbox weights are [num_classes*4, 1024]
            inferred_num_classes = int(bbox_w.shape[0] // 4)

        if inferred_num_classes is None or inferred_num_classes <= 0:
            # Fallback: create default then load with strict=False
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        else:
            # Create model with matching head
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=inferred_num_classes)

        # Load weights; allow missing/unexpected due to minor naming differences
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        _faster_rcnn_model = model
        return _faster_rcnn_model
    except Exception as e:
        raise RuntimeError(f"Failed to load Faster-RCNN weights: {e}")
    return _faster_rcnn_model

def predict_with_yolov5(pil_image: Image.Image):
    """Run inference using YOLOv5 and return a list of dicts."""
    results = yolov5_model.predict(pil_image)
    # Normalized output: list of {xmin, ymin, xmax, ymax, confidence, label/name}
    records = results.pandas().xyxy[0].to_dict(orient="records")
    normalized = []
    for r in records:
        normalized.append({
            "xmin": float(r.get("xmin", 0.0)),
            "ymin": float(r.get("ymin", 0.0)),
            "xmax": float(r.get("xmax", 0.0)),
            "ymax": float(r.get("ymax", 0.0)),
            "confidence": float(r.get("confidence", 0.0)),
            "name": str(r.get("name", ""))
        })
    return normalized

def predict_with_faster_rcnn(pil_image: Image.Image):
    """Run inference using Faster-RCNN loaded from .pth and return a list of dicts."""
    model = get_faster_rcnn_model()
    image = pil_image.convert("RGB")
    # Basic preprocessing to tensor in [0,1]
    arr = np.array(image)
    if arr.ndim == 2:
        # Grayscale to 3-channels
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).to(torch.uint8)  # HWC, uint8
    tensor = tensor.permute(2, 0, 1).to(torch.float32) / 255.0  # CHW float32
    with torch.no_grad():
        outputs = model([tensor])[0]
    boxes = outputs.get("boxes", [])
    scores = outputs.get("scores", [])
    labels = outputs.get("labels", [])
    result = []
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        score = float(scores[i].item()) if hasattr(scores[i], 'item') else float(scores[i])
        label_id = int(labels[i].item()) if hasattr(labels[i], 'item') else int(labels[i])
        # Map numeric label id to name if available
        mapped_label = FASTER_RCNN_LABEL_MAP.get(label_id, str(label_id))
        result.append({
            "xmin": float(box[0]),
            "ymin": float(box[1]),
            "xmax": float(box[2]),
            "ymax": float(box[3]),
            "score": score,
            # Frontend supports 'label' or 'name'
            "label": mapped_label
        })
    return result

def get_yolov8_model():
    """Load YOLOv8 model via ultralytics if available."""
    global _yolov8_model
    if _yolov8_model is not None:
        return _yolov8_model
    try:
        from ultralytics import YOLO  # lazy import
    except Exception as e:
        raise RuntimeError("ultralytics is not installed for YOLOv8: " + str(e))
    if not YOLOV8_WEIGHTS.exists():
        # Give a clearer error listing candidates we searched
        searched = [str(YOLOV8_WEIGHTS)]
        try:
            searched.extend([str(p) for p in sorted(BASE_DIR.glob("yolov8*.pt*"))])
        except Exception:
            pass
        raise RuntimeError("YOLOv8 weights not found. Tried: " + ", ".join(searched))
    _yolov8_model = YOLO(str(YOLOV8_WEIGHTS))
    return _yolov8_model

def predict_with_yolov8(pil_image: Image.Image):
    """Run inference using YOLOv8 and normalize output format."""
    model = get_yolov8_model()
    # Try prediction with verbose disabled for broader compatibility
    try:
        results_list = model.predict(pil_image, verbose=False)  # returns list
    except TypeError as e:
        # Handle older BaseModel.fuse signature without 'verbose'
        if "fuse() got an unexpected keyword argument 'verbose'" in str(e):
            base = getattr(model, 'model', None)
            if base is not None and hasattr(base, 'fuse'):
                original_fuse = base.fuse
                def fuse_compat(*args, **kwargs):
                    # Drop unsupported kw and delegate
                    kwargs.pop('verbose', None)
                    return original_fuse(*args, **kwargs)
                base.fuse = fuse_compat
            # Retry prediction without tripping on verbose kw
            results_list = model.predict(pil_image, verbose=False)
        elif "unexpected keyword argument 'embed'" in str(e):
            # Workaround some ultralytics versions: run via temporary file path
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    tmp_path = f.name
                    pil_image.save(tmp_path, format='JPEG')
                results_list = model.predict(tmp_path, verbose=False)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        else:
            raise
    normalized = []
    if not results_list:
        return normalized
    res = results_list[0]
    # Iterate over boxes
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        score = float(b.conf[0].item()) if hasattr(b, 'conf') else 0.0
        cls_id = int(b.cls[0].item()) if hasattr(b, 'cls') else -1
        name = str(res.names.get(cls_id, str(cls_id))) if hasattr(res, 'names') else str(cls_id)
        normalized.append({
            "xmin": float(xyxy[0]),
            "ymin": float(xyxy[1]),
            "xmax": float(xyxy[2]),
            "ymax": float(xyxy[3]),
            "confidence": score,
            "name": name,
        })
    return normalized

def get_rf_detr():
    """Load RF-DETR model using RFDETRBase class."""
    global _rf_detr_model
    if _rf_detr_model is not None:
        return _rf_detr_model
    
    if not RF_DETR_AVAILABLE:
        raise RuntimeError("RF-DETR dependencies not available. Please install rfdetr and supervision packages.")
    
    if not RF_DETR_WEIGHTS.exists():
        raise RuntimeError(f"RF-DETR weights not found at: {RF_DETR_WEIGHTS}")
    
    try:
        # Load the RF-DETR model using RFDETRBase class
        # This matches your working example code
        _rf_detr_model = RFDETRBase(pretrain_weights=str(RF_DETR_WEIGHTS))
        return _rf_detr_model
    except Exception as e:
        raise RuntimeError(f"Failed to load RF-DETR model: {e}")

def predict_with_rf_detr(pil_image: Image.Image):
    """Run inference for RF-DETR using the correct API."""
    model = get_rf_detr()
    
    # Convert PIL image to RGB if needed
    image = pil_image.convert("RGB")
    
    try:
        # Use the RF-DETR model's predict method (matches your working example)
        detections = model.predict(image)
        
        # Convert supervision detections to our normalized format
        result = []
        if detections is not None and len(detections) > 0:
            for i in range(len(detections.xyxy)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = detections.xyxy[i]
                confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.0
                class_id = detections.class_id[i] if hasattr(detections, 'class_id') else 0
                
                # Map class_id to class name
                if 0 <= class_id < len(RF_DETR_CLASS_NAMES):
                    class_name = RF_DETR_CLASS_NAMES[class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                result.append({
                    "xmin": float(x1),
                    "ymin": float(y1),
                    "xmax": float(x2),
                    "ymax": float(y2),
                    "confidence": float(confidence),
                    "name": class_name,
                })
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"RF-DETR prediction failed: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    # Validate file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Parse model key (default to yolov5 for backward compatibility)
    model_key = request.form.get("model", "yolov5").strip().lower()

    file = request.files['file']
    pil_image = Image.open(file.stream)

    try:
        if model_key == "yolov5":
            output = predict_with_yolov5(pil_image)
        elif model_key == "fasterrcnn":
            output = predict_with_faster_rcnn(pil_image)
        elif model_key == "yolov8":
            output = predict_with_yolov8(pil_image)
        elif model_key in ("rf-detr", "rf_detr", "rfdet r", "rf detr", "rfdetr", "lastmodel"):
            # Accept several variants and the old 'lastmodel' for compatibility
            # if not RF_DETR_AVAILABLE:
            #     return jsonify({"error": "RF-DETR dependencies not available. Please install rfdetr and supervision packages."}), 500
            output = predict_with_rf_detr(pil_image)
        else:
            return jsonify({"error": f"Unknown model '{model_key}'."}), 400
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Return normalized list with 200 OK
    return jsonify(output), 200

@app.route("/", methods=["GET"])
def home():
    return "Model API is up and running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
