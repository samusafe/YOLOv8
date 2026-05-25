# Model Card — YOLOv8 SKU-110K Shelf Detector (Nano - v3.0)

## 1. Overview

- **Goal:** Detetar e isolar produtos em prateleiras de retalho, utilizando o benchmark público SKU-110K para maximizar a diversidade e capacidade de generalização do modelo.
- **Task:** Object Detection
- **Classes:** 1 (Produto/Item)
- **Intended users:** Operadores de armazém, engenheiros de controlo de qualidade e gestão de stock.

## 2. Intended Use and Scope

- **Intended environment:** Armazéns logísticos, supermercados, arquivos industriais.
- **Assumptions:** Câmara com visão frontal ou ligeiramente angular para as prateleiras. Boa a razoável condição de iluminação artificial.
- **Out-of-scope uses:** Este modelo não foi treinado para identificar a marca ou tipo específico de produto. Não deve ser usado ao ar livre ou em contextos não relacionados com prateleiras (e.g., objetos em chão ou mesas).

## 3. Training Data

- **Data source:** Dataset SKU-110K — benchmark público de referência para deteção de produtos em prateleiras de retalho.
- **Source URL:** https://universe.roboflow.com/large-benchmark-datasets/sku-110k-dataset
- **Dataset size:** ~11 000 imagens com partições de treino, validação e teste.
- **Augmentations:** Mosaic, Horizontal Flip, ajustes HSV, translação, escala e erasing (ver `train_config.yaml` para valores exatos).

## 4. Evaluation

- **Test set description:** Partição de teste incluída no dataset SKU-110K.
- **Metrics (Após 50 Epochs):**
  - mAP@0.5: **0.899**
  - mAP@0.5:0.95: **0.553**
  - Precision: **0.901**
  - Recall: **0.840**
- **Qualitative analysis:** Este modelo obteve resultados **significativamente superiores** aos dos modelos Nano e Small treinados no dataset mais pequeno "shelves" (mAP50: **0.899** vs 0.867/0.865). Apesar de utilizar a mesma arquitetura Nano (mais leve), a superioridade das métricas demonstra que a **qualidade e diversidade do dataset** são mais determinantes do que o tamanho da arquitetura da rede. As ~11 000 imagens diversificadas do SKU-110K, provenientes de diferentes ambientes de retalho, conferiram ao modelo uma capacidade de generalização muito superior.

## 5. Limitations and Failure Modes

- Treinado exclusivamente em imagens de prateleiras de retalho; o desempenho pode degradar-se significativamente em contextos visuais distintos (e.g., objetos em chão, mesas ou ao ar livre).
- Produtos empilhados de forma irregular podem ser fundidos numa só caixa delimitadora.
- Reflexos muito intensos na embalagem podem gerar falsos negativos.

## 6. Deployment Notes

- **Input requirements:** Resolução ótima recomendada de 640x640.
- **Output format:** Formato padrão YOLO (x_center, y_center, width, height, confidence).
- **Compute:**
  - Treinado em NVIDIA GeForce RTX 4060 (8GB).
  - Tempo total de treino: ~5.2 horas (18 712 segundos).
  - Tempos médios de Inferência: **~3.2 ms** por imagem (GPU). Altamente viável para Edge computing devido à arquitetura Nano (yolov8n).

## 7. Ethical / Safety / Privacy Considerations

- Riscos: O modelo não processa dados humanos (como rostos). Não existem riscos éticos diretos previstos para privacidade individual.

## 8. Versioning and Contact

- **Version:** v3.0 (Modelo Nano — SKU-110K)
- **Date:** Maio 2026
- **Authors:** Bruno Carvalho e Guilherme Canha
