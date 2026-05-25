# Model Card: YOLOv8 Nano — Deteção de Paletes (Armazém)

## Detalhes do Modelo
- **Arquitetura Base:** YOLOv8 Nano (Fine-tuned a partir do modelo SKU-110K)
- **Tarefa:** Deteção de Objetos (Object Detection)
- **Versão:** 4.0
- **Data:** Maio de 2026
- **Contexto:** Projeto Universitário de Inteligência Artificial (Engenharia Informática)

## Uso Pretendido
Este modelo destina-se a sistemas de gestão logística e auditoria de armazéns industriais. Foi treinado para identificar grandes unidades logísticas (paletes/caixotes) em estantes (racks) de alta densidade, diferenciando-se assim de modelos de retalho puro.

## Detalhes do Treino
- **Framework:** Ultralytics YOLOv8
- **Hardware:** NVIDIA GeForce RTX 4060 (8GB VRAM)
- **Épocas:** 50
- **Estratégia:** Transfer Learning (Pesos iniciais provenientes do modelo treinado com 11,000 imagens do SKU-110K, permitindo uma convergência extremamente rápida e precisa no novo domínio).

## Dataset
- **Nome:** Pallet Detection
- **Origem:** [Roboflow Universe](https://universe.roboflow.com/pallet-detection-qy9uv/pallet-detection-hcw8s)
- **Tamanho:** ~972 imagens (augmentações incluídas)
- **Classes:** 1 (`item` — representando uma unidade de palete/caixote)

## Métricas de Desempenho (Test Set Independente)
- **mAP@0.5:** 0.987 (98.7%)
- **mAP@0.5:0.95:** 0.758 (75.8%)
- **Precision:** 0.960
- **Recall:** 0.976
- **Inference Speed:** ~3.2ms por imagem na GPU

## Análise Qualitativa
O modelo apresentou uma "sabedoria semântica" excelente face ao modelo SKU original. Enquanto o modelo original sofre de alucinações perante tábuas verticais num armazém (confundindo-as com pacotes pequenos), este modelo consegue encapsular um caixote gigante numa única "bounding box", percebendo o conceito de "unidade de carga paletizada". 

## Limitações
- Devido à especialização (Catastrophic Forgetting positivo para o caso de uso), o modelo perdeu a capacidade de detetar caixas individuais pequenas numa prateleira de retalho (como num supermercado).
- O dataset é relativamente pequeno, o que significa que paletes com formatos muito atípicos ou envolvimentos plásticos altamente refletores poderão, em cenários raros, não ser detetados.
