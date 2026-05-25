import argparse
import json
import os
import sys
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Ponto de entrada CLI para inferência com YOLOv8 na linha de comandos.")
    parser.add_argument("--model", type=str, required=True, help="Caminho para os pesos do modelo (ex: modelos/yolov8n_sku/weights.pt)")
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem de teste a analisar")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança para filtrar deteções (default: 0.5)")
    
    args = parser.parse_args()
    
    # Validações iniciais
    if not os.path.exists(args.model):
        print(f"Erro: O modelo especificado não foi encontrado em '{args.model}'", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.image):
        print(f"Erro: A imagem especificada não foi encontrada em '{args.image}'", file=sys.stderr)
        sys.exit(1)
        
    print(f"🔄 A carregar o modelo YOLOv8 a partir de: {args.model}")
    model = YOLO(args.model)
    
    print(f"🧠 A executar inferência na imagem: {args.image} (conf={args.conf})")
    results = model.predict(source=args.image, conf=args.conf, verbose=False)
    
    result = results[0]
    
    # Determinar nomes de ficheiros de saída baseados na imagem original
    base_name, ext = os.path.splitext(args.image)
    out_img_path = f"{base_name}_annotated{ext}"
    out_json_path = f"{base_name}_detections.json"
    
    # 1. Guardar Imagem Anotada com Bounding Boxes
    annotated_img = result.plot()
    cv2.imwrite(out_img_path, annotated_img)
    print(f"✅ Imagem anotada guardada com sucesso em: {out_img_path}")
    
    # 2. Extrair dados para JSON Estruturado
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [round(float(c), 1) for c in box.xyxy[0]]
        
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "bbox": {
                "x1": x1, 
                "y1": y1, 
                "x2": x2, 
                "y2": y2
            }
        })
        
    json_data = {
        "file": args.image,
        "model_used": args.model,
        "confidence_threshold": args.conf,
        "total_detections": len(detections),
        "detections": detections
    }
    
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    
    # 3. Guardar Ficheiro JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"✅ Resultados guardados com sucesso em: {out_json_path}\n")
    
    # 4. Imprimir no Terminal (Requisito do Enunciado)
    print("📋 --- Saída JSON Estruturada ---")
    print(json_str)
    
if __name__ == "__main__":
    main()
