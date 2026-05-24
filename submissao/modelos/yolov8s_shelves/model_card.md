# Model Card — YOLOv8 Shelves Detector (Small - v2.0)

## 1. Overview
- **Goal:** Detetar e isolar produtos em prateleiras de armazém/supermercado para potenciar a gestão de inventário.
- **Task:** Object Detection
- **Classes:** 1 (Produto/Item)
- **Intended users:** Operadores de armazém, engenheiros de controlo de qualidade e gestão de stock.

## 2. Intended Use and Scope
- **Intended environment:** Armazéns logísticos, supermercados, arquivos industriais. 
- **Assumptions:** Câmara com visão frontal ou ligeiramente angular para as prateleiras. Boa a razoável condição de iluminação artificial.
- **Out-of-scope uses:** Este modelo não foi treinado para identificar a marca ou tipo específico de produto. Não deve ser usado ao ar livre.

## 3. Training Data
- **Data source:** Dataset "shelves" importado do Roboflow.
- **Dataset size:** ~1673 imagens geradas. (Treino: 1462 imagens; Validação: 211 imagens).
- **Augmentations:** Resize (640x640), Horizontal Flip, Brightness (-15% a +15%), Blur (1.25px), 3x Multiplicador.

## 4. Evaluation
- **Test set description:** Conjunto de validação isolado de 211 imagens (cerca de 12.6% dos dados).
- **Metrics (Após 150 Epochs):** 
  - mAP@0.5: **0.865**
  - mAP@0.5:0.95: **0.499**
  - Precision: **0.884**
  - Recall: **0.827**
- **Qualitative analysis:** Em comparação com o modelo Nano (que obteve mAP50 de 0.867), o modelo Small (mAP50 de 0.865) obteve resultados estaticamente idênticos, com uma ligeira melhoria na precisão (0.884 vs 0.879) mas sem ganhos significativos no mAP global. Isto sugere que o estrangulamento na deteção reside na qualidade/tamanho do dataset e não na arquitetura da rede, sendo o Nano mais eficiente por ser mais rápido.

## 5. Limitations and Failure Modes
- Produtos em zonas de grande penumbra continuam a ter dificuldade de deteção.
- Rótulos visuais muito idênticos podem não ser distinguidos.

## 6. Deployment Notes
- **Input requirements:** Resolução ótima recomendada de 640x640.
- **Output format:** Formato padrão YOLO (x_center, y_center, width, height, confidence).
- **Compute:** 
  - Treinado em NVIDIA GeForce RTX 4060 (8GB).
  - Tempos médios de Inferência: **~5.5 ms** por imagem. (Ligeiramente mais pesado que a versão Nano, mas muito viável).

## 7. Ethical / Safety / Privacy Considerations
- Riscos: O modelo não processa dados humanos (como rostos). Não existem riscos éticos diretos previstos para privacidade individual.

## 8. Versioning and Contact
- **Version:** v2.0 (Modelo Small)
- **Date:** Maio 2026
- **Authors:** Grupo de IA
