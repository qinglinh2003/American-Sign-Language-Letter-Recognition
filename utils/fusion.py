import math

def fuse_vision_and_lm(vision_candidates, lm_probs, alpha=0.5):
    fused_scores = {}
    for letter, vconf in vision_candidates:
        lconf = lm_probs.get(letter, 1e-6)
        fused_score = alpha * math.log(vconf + 1e-9) + (1 - alpha) * math.log(lconf + 1e-9)
        fused_scores[letter] = fused_score
    best_letter = max(fused_scores, key=fused_scores.get)
    return best_letter, fused_scores
