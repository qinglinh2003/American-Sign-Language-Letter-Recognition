import torch

def get_topk_candidates_from_yolo(frame, yolo_model, k=3):
    results = yolo_model(frame)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        confidences = boxes.conf  # Tensor
        k = min(k, len(confidences))
        topk = torch.topk(confidences, k=k)
        candidates = []
        for i in range(k):
            idx = topk.indices[i]
            letter = results[0].names[int(boxes.cls[idx])]
            conf = float(topk.values[i])
            candidates.append((letter, conf))
        return candidates
    else:
        return [("Unknown", 1.0)]