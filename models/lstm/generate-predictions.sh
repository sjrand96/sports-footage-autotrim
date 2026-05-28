CKPT="models/lstm/checkpoints/2026-05-27-14:47-cnn-lstm-3sec-context/best.pt"
OUT="models/lstm/predictions/2026-05-27-14:47-cnn-lstm-3sec-context"
mkdir -p "$OUT"

while IFS= read -r clip_id || [[ -n "$clip_id" ]]; do
  [[ -z "$clip_id" ]] && continue
  python models/lstm/export_clip_predictions.py \
    --clip-id "$clip_id" \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --device mps
done < data/test_clips.csv