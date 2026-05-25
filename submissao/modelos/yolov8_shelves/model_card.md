# Model Card — YOLOv8 Shelves Detector (v1.0)

## 1. Overview

- **Goal:** Detetar e isolar produtos em prateleiras de armazém/supermercado para potenciar a gestão de inventário e deteção de falhas de stock.
- **Task:** Object Detection
- **Classes:** 1 (Produto/Item)
- **Intended users:** Operadores de armazém, engenheiros de controlo de qualidade e gestão de stock.

## 2. Intended Use and Scope

- **Intended environment:** Armazéns logísticos, supermercados, arquivos industriais.
- **Assumptions:** Câmara com visão frontal ou ligeiramente angular para as prateleiras. Boa a razoável condição de iluminação artificial.
- **Out-of-scope uses:** Este modelo não foi treinado para identificar a marca ou tipo específico de produto (apenas a sua presença e localização na prateleira). Não deve ser usado ao ar livre.

## 3. Training Data

- **Data source:** Dataset "shelves" importado do Roboflow (recolha baseada em imagens públicas/curadoria do grupo).
- **Dataset size:** ~1673 imagens geradas. (Treino: 1462 imagens; Validação: 211 imagens).
- **Class distribution:** Milhares de instâncias equilibradas em torno de uma única classe.
- **Labeling guidelines:** As caixas delimitadoras (bounding boxes) devem abranger as extremidades visíveis de cada produto individual na prateleira.
- **Augmentations:**
  - Resize: Stretch to 640x640
  - Horizontal Flip
  - Brightness: -15% a +15%
  - Blur: Até 1.25px
  - Multiplicador: 3x

## 4. Evaluation

- **Test set description:** Conjunto de validação isolado de 211 imagens (cerca de 12.6% dos dados).
- **Metrics (Após 50 Epochs):**
  - mAP@0.5: **0.867**
  - mAP@0.5:0.95: **0.506**
  - Precision: **0.879**
  - Recall: **0.811**
- **Qualitative analysis:** O modelo demonstrou uma elevada capacidade de separar itens muito próximos uns dos outros.
- **Recommended confidence threshold(s):** 0.50. Reduzir para 0.35 se for detetada a falta de produtos em prateleiras muito escuras.

## 5. Limitations and Failure Modes

- Known failure conditions: Produtos empilhados de forma irregular podem ser fundidos numa só "caixa" (bounding box). Reflexos muito intensos na embalagem podem gerar falsos negativos.
- Typical false positives / false negatives: Pode detetar etiquetas de preço coloridas como sendo produtos, se o limiar de confiança estiver muito baixo.

## 6. Deployment Notes

- **Input requirements:** Resolução ótima recomendada de 640x640.
- **Output format:** Formato padrão YOLO (Caixa delimitadora: x_center, y_center, width, height, confidence).
- **Compute:**
  - Treinado em NVIDIA GeForce RTX 4060 (8GB).
  - Tempos médios de Inferência: **3.2ms** por imagem (GPU). Altamente viável para Edge computing ou processadores comuns (CPU) devido ao seu tamanho Nano (yolov8n).

## 7. Ethical / Safety / Privacy Considerations

- Riscos: O modelo não processa dados humanos (como rostos). Não existem riscos éticos diretos previstos para privacidade individual.

## 8. Versioning and Contact

- **Version:** v1.0
- **Date:** Maio 2026
- **Authors:** Bruno Carvalho e Guilherme Canha
