# Relatório de Avaliação dos Modelos no Test Set

A tabela abaixo representa a performance de inferência rigorosa de cada modelo num conjunto de imagens de teste que **nunca foi visto** durante o treino, garantindo que não existe overfitting e que os valores são reais.

| Modelo | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8 Nano (Shelves 50ep) | 0.882 | 0.844 | 0.890 | 0.534 |
| YOLOv8 Small (Shelves 150ep) | 0.884 | 0.858 | 0.893 | 0.540 |
| YOLOv8 Nano (SKU 110K 50ep) | 0.898 | 0.844 | 0.900 | 0.555 |
| YOLOv8 Nano (Paletes 50ep) | 0.960 | 0.976 | 0.987 | 0.758 |
