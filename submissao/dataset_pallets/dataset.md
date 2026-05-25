# Dataset Terciário — Pallet Detection (Armazém Industrial)

Este dataset foi utilizado como base para especializar o modelo de deteção num contexto estritamente logístico e industrial, demonstrando a incrível capacidade de _Transfer Learning_ — transitar o conhecimento de deteção de produtos de retalho (caixas pequenas) para prateleiras de armazém de alta densidade (paletes completas).

## Detalhes do Conjunto de Dados

- **Origem / Fonte:** Roboflow Universe
- **Projeto:** `pallet-detection-hcw8s` (Workspace: `pallet-detection-qy9uv`)
- **Link de Acesso Público:** [Roboflow Universe - Pallet Detection](https://universe.roboflow.com/pallet-detection-qy9uv/pallet-detection-hcw8s)
- **Data de Obtenção:** Maio de 2026

## Estrutura e Distribuição

O dataset é composto por fotografias recolhidas em ambientes industriais reais, centros de distribuição e armazéns logísticos. Foca-se em prateleiras do tipo _rack_ industrial.

**Total de Imagens:** 972

| Partição  | Número de Imagens | Percentagem |
| --------- | ----------------- | ----------- |
| Treino    | 840               | ~86%        |
| Validação | 112               | ~12%        |
| Teste     | 20                | ~2%         |

## Classes e Anotações

Para garantir total compatibilidade arquitetural com os restantes modelos do projeto, o dataset foi formatado para possuir **1 classe única**:

- `0: item` — Representa, neste contexto, uma **unidade de carga paletizada** (paletes embrulhadas, caixotes de madeira, material de construção empilhado) nas estantes industriais.

As anotações encontram-se no formato normalizado do YOLOv8 (`class x_center y_center width height`).
