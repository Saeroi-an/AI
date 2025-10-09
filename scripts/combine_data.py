import json

with open("data/synth_rx/train.json", "r", encoding="utf-8") as f:
    synth_data = json.load(f)

with open("data/cord_qwen.json", "r", encoding="utf-8") as f:
    cord_data = json.load(f)

combined = synth_data + cord_data

with open("data/combined_train.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
