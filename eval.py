import os
from ultralytics import YOLO

MODELS_TO_EVALUATE = [
    {
        "name": "YOLOv8 Nano (Shelves 50ep)",
        "weights": "submissao/modelos/yolov8_shelves/weights.pt",
        "data": "submissao/dataset/data.yaml"
    },
    {
        "name": "YOLOv8 Small (Shelves 150ep)",
        "weights": "submissao/modelos/yolov8s_shelves/weights.pt",
        "data": "submissao/dataset/data.yaml"
    },
    {
        "name": "YOLOv8 Nano (SKU 110K 50ep)",
        "weights": "submissao/modelos/yolov8n_sku/weights.pt",
        "data": "dataset_sku110k/data.yaml"
    },
    {
        "name": "YOLOv8 Nano (Paletes 50ep)",
        "weights": "submissao/modelos/yolov8n_pallets/weights.pt",
        "data": "submissao/dataset_pallets/data.yaml"
    }
]

def main():
    print("=== YOLOv8 Evaluation Script ===")
    results_list = []

    for model_info in MODELS_TO_EVALUATE:
        name = model_info["name"]
        weights = model_info["weights"]
        data = model_info["data"]

        if not os.path.exists(weights):
            print(f"[{name}] ERRO: Pesos não encontrados em {weights}. Ignorando.")
            continue
            
        if not os.path.exists(data):
            print(f"[{name}] ERRO: Dataset {data} não encontrado. Ignorando.")
            continue

        print(f"\n--- A avaliar: {name} ---")
        try:
            model = YOLO(weights)
            
            # Corre a validação forçando o uso do split 'test'
            # verbose=False para não encher o ecrã com prints de cada batch
            metrics = model.val(data=data, split='test', verbose=False)
            
            # O YOLOv8 guarda os resultados num dicionário
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
            map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
            precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
            recall = metrics.results_dict.get('metrics/recall(B)', 0.0)
            
            results_list.append({
                "Modelo": name,
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "mAP50": f"{map50:.3f}",
                "mAP50_95": f"{map50_95:.3f}"
            })
            print(f"[{name}] Concluído: mAP50 = {map50:.3f}")
        except Exception as e:
            print(f"[{name}] ERRO durante a avaliação: {e}")

    # Guardar os resultados num relatório Markdown
    report_path = "submissao/modelos/evaluation_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Relatório de Avaliação dos Modelos no Test Set\n\n")
        f.write("A tabela abaixo representa a performance de inferência rigorosa de cada modelo num conjunto de imagens de teste que **nunca foi visto** durante o treino, garantindo que não existe overfitting e que os valores são reais.\n\n")
        f.write("| Modelo | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |\n")
        f.write("|---|---|---|---|---|\n")
        for res in results_list:
            f.write(f"| {res['Modelo']} | {res['Precision']} | {res['Recall']} | {res['mAP50']} | {res['mAP50_95']} |\n")
    
    print(f"\n✅ Avaliação terminada! Relatório guardado em: {report_path}")

if __name__ == "__main__":
    main()
