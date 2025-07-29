import cv2
from collections import Counter, deque
import torch
from ultralytics import YOLO
from models.gpt_model import GPTModel
from utils.vocabulary import get_vocab, stoi, itos
from utils.inference import get_lm_probs
from utils.visual_utils import get_topk_candidates_from_yolo
from utils.fusion import fuse_vision_and_lm


def main():
    chars, vocab_size, stoi, itos = get_vocab()

    stable_threshold = 3
    last_predictions = deque(maxlen=stable_threshold)
    confirmed_prediction = "Unknown"
    last_confirmed = None
    reset_required = True
    text_buffer = ""

    main_model = YOLO("../models/vision/baseline.pt")
    fine_tune_model1 =  YOLO("../models/vision/beu.pt")
    fine_tune_model2 =  YOLO("../models/vision/amnst.pt")
    fine_tune_model3 =  YOLO("../models/vision/krx.pt")
    fine_tune_model4 =  YOLO("../models/vision/other.pt")

    hard_classes1 = {"B", "E", "U"}
    hard_classes2 = {"A", "M", "N", "S", "T"}
    hard_classes3 = {"K", "R", "X"}

    # Load language model
    lm_model = GPTModel(vocab_size=vocab_size)
    lm_checkpoint = torch.load( "../models/text/char_GPT.pth", map_location="cpu")
    lm_model.load_state_dict(lm_checkpoint['model_state_dict'])
    lm_model.eval()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vision_candidates = get_topk_candidates_from_yolo(frame, main_model, k=3)
        top_letter = vision_candidates[0][0]

        if top_letter in hard_classes1:
            vision_candidates = get_topk_candidates_from_yolo(frame, fine_tune_model1, k=3)
        elif top_letter in hard_classes2:
            vision_candidates = get_topk_candidates_from_yolo(frame, fine_tune_model2, k=3)
        elif top_letter in hard_classes3:
            vision_candidates = get_topk_candidates_from_yolo(frame, fine_tune_model3, k=3)
        else:
            vision_candidates = get_topk_candidates_from_yolo(frame, fine_tune_model4, k=3)

        context = text_buffer[-10:] if len(text_buffer) >= 10 else text_buffer
        lm_probs = get_lm_probs(context, lm_model)

        best_letter, fused_scores = fuse_vision_and_lm(vision_candidates, lm_probs)

        last_predictions.append(best_letter)

        if len(last_predictions) == stable_threshold:
            most_common = Counter(last_predictions).most_common(1)[0]
            if most_common[1] >= stable_threshold * 0.8:
                confirmed_prediction = most_common[0]

                if confirmed_prediction == "del":
                    if text_buffer:
                        text_buffer = text_buffer[:-1]
                    last_confirmed = "del"
                    reset_required = False
                elif confirmed_prediction == "space":
                    if last_confirmed != "space" or reset_required:
                        text_buffer += " "
                        last_confirmed = "space"
                        reset_required = False
                elif confirmed_prediction not in {"Unknown", "del", "space"}:
                    if confirmed_prediction != last_confirmed or reset_required:
                        text_buffer += confirmed_prediction
                        last_confirmed = confirmed_prediction
                        reset_required = False

                if confirmed_prediction == "Unknown":
                    reset_required = True
                    last_confirmed = None

        base_x, base_y = 50, 50
        line_spacing = 30

        cv2.putText(frame, f"Text: {text_buffer}", (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Prediction: {confirmed_prediction}", (base_x, base_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        vision_topk_text = "Vision TopK: " + ", ".join([f"{l}:{c:.2f}" for l, c in vision_candidates])
        cv2.putText(frame, vision_topk_text, (base_x, base_y + 2 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        sorted_lm = sorted(lm_probs.items(), key=lambda x: x[1], reverse=True)
        lm_top3_text = "LM Top3: " + ", ".join([f"{l}:{p:.2f}" for l, p in sorted_lm[:3]])
        cv2.putText(frame, lm_top3_text, (base_x, base_y + 3 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        fused_top_text = "Fused: " + ", ".join(
            [f"{l}:{p:.2f}" for l, p in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]])
        cv2.putText(frame, fused_top_text, (base_x, base_y + 4 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        cv2.imshow("Hand Gesture Typing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
