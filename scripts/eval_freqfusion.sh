#!/usr/bin/env bash
set -e
echo "[eval] Evaluación simulada..." | tee logs/freqfusion/eval.log
python - <<'PY'
import json, os
metrics = {
  "AP": 0.192,
  "AP50": 0.334,
  "AP75": 0.201,
  "APs": 0.070,
  "APm": 0.190,
  "APl": 0.310
}
os.makedirs("results/freqfusion", exist_ok=True)
with open("results/freqfusion/coco_metrics.json","w") as f:
  json.dump(metrics, f, indent=2)
print("Métricas dummy guardadas en results/freqfusion/coco_metrics.json")
PY
echo "[eval] Fin." | tee -a logs/freqfusion/eval.log