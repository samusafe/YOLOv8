"""
Script de Inferência — Ponto de Entrada CLI (D3)
==================================================
Uso:
  python infer.py --model modelos/yolov8n_sku/weights.pt --image caminho/para/imagem.jpg
  python infer.py --model modelos/yolov8_shelves/weights.pt --image caminho/para/imagem.jpg --conf 0.6
"""
import argparse
import json
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Inferência YOLOv8 — Deteção de Produtos em Prateleiras")
    parser.add_argument("--model", type=str, required=True,
                        help="Caminho para os pesos do modelo (ex: modelos/yolov8n_sku/weights.pt)")
    parser.add_argument("--image", type=str, required=True,
                        help="Caminho para a imagem de teste")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Limiar de confiança mínima (default: 0.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Pasta onde guardar a imagem anotada (default: ao lado da imagem original)")
    args = parser.parse_args()

    # Carregar modelo
    print(f"A carregar modelo: {args.model}")
    model = YOLO(args.model)

    # Inferência
    print(f"A processar: {args.image} (confiança >= {args.conf})")
    results = model.predict(source=args.image, conf=args.conf, save=True,
                            project=args.output or os.path.dirname(args.image),
                            name="resultado", exist_ok=True)

    # Saída estruturada (JSON)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": results[0].names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 1),
                "y1": round(float(box.xyxy[0][1]), 1),
                "x2": round(float(box.xyxy[0][2]), 1),
                "y2": round(float(box.xyxy[0][3]), 1),
            }
        })

    output_json = {"total_detections": len(detections), "detections": detections}

    # Imprimir resultado
    print(f"\n{'='*50}")
    print(f"  RESULTADO: {len(detections)} produtos detetados")
    print(f"{'='*50}")
    print(json.dumps(output_json, indent=2, ensure_ascii=False))

    # Guardar JSON ao lado da imagem
    json_path = os.path.splitext(args.image)[0] + "_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados em: {json_path}")


if __name__ == "__main__":
    main()
