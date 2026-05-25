"""
Script Universal de Treino YOLOv8
=================================
Uso:
  python train.py --model yolov8n.pt --data submissao/dataset/data.yaml --epochs 50 --name yolov8_shelves
  python train.py --model yolov8s.pt --data submissao/dataset/data.yaml --epochs 150 --name yolov8s_shelves
  python train.py --model yolov8n.pt --data submissao/dataset_sku110k/data.yaml --epochs 50 --name yolov8n_sku
"""
import argparse
import os
import sys
from ultralytics import YOLO


class Logger:
    """Duplica stdout para um ficheiro de texto, permitindo rever os logs do treino."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Treino YOLOv8 — Script Universal")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Modelo base (ex: yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--data", type=str, default="submissao/dataset/data.yaml",
                        help="Caminho para o data.yaml do dataset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas de treino")
    parser.add_argument("--batch", type=int, default=16,
                        help="Tamanho do batch")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Resolução das imagens")
    parser.add_argument("--name", type=str, required=True,
                        help="Nome da pasta de saída (ex: yolov8_shelves)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join("submissao", "modelos", args.name)
    os.makedirs(output_dir, exist_ok=True)

    # Ativar logging para ficheiro
    log_path = os.path.join(output_dir, "training_log.txt")
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout

    print(f"{'='*60}")
    print(f"  TREINO YOLOv8 — {args.name}")
    print(f"{'='*60}")
    print(f"  Modelo base:  {args.model}")
    print(f"  Dataset:      {args.data}")
    print(f"  Épocas:       {args.epochs}")
    print(f"  Batch:        {args.batch}")
    print(f"  Resolução:    {args.imgsz}px")
    print(f"  Saída:        {output_dir}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="submissao/modelos",
        name=args.name,
        exist_ok=True
    )

    print(f"\n{'='*60}")
    print(f"  TREINO CONCLUÍDO COM SUCESSO!")
    print(f"  Resultados guardados em: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
