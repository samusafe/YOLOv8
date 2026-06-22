import os
import json
from ultralytics import YOLO

def main():
    print("📊 A iniciar a avaliação do modelo YOLOv8 Nano (Shelves)...")
    
    weights_path = "weights.pt"
    data_path = "../../dataset/data.yaml"
    out_json = "eval.json"
    
    if not os.path.exists(weights_path):
        print(f"❌ Erro: O modelo '{weights_path}' não foi encontrado.")
        return
        
    model = YOLO(weights_path)
    
    print(f"🔍 A avaliar usando o dataset: {data_path}")
    metrics = model.val(data=data_path, split="test", verbose=False)
    
    results = {
        "model": "YOLOv8 Nano (Shelves)",
        "dataset": data_path,
        "mAP50": round(metrics.box.map50, 4),
        "mAP50_95": round(metrics.box.map, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4)
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("\n✅ --- Resultados da Avaliação (Test Set) ---")
    print(f"mAP@50:      {results['mAP50']}")
    print(f"mAP@50-95:   {results['mAP50_95']}")
    print(f"Precision:   {results['precision']}")
    print(f"Recall:      {results['recall']}")
    print(f"\n📁 Resultados guardados em '{out_json}'.")
    
if __name__ == "__main__":
    main()
