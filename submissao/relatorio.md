# Relatório Técnico — Monitorização Inteligente de Prateleiras

**Unidade Curricular:** Inteligência Artificial  
**Curso:** Licenciatura em Engenharia Informática — ESTG/IPP  
**Ano Letivo:** 2025/2026  
**Autores:** Bruno Carvalho (8220442) · Guilherme Canha (8220449)  
**Data:** Maio 2026

---

## 1. Descrição do Caso de Uso

### 1.1. Motivação Industrial e Cenário-Alvo

A gestão eficiente de inventário em ambientes de retalho e armazéns logísticos constitui um desafio operacional significativo. Atualmente, a verificação da disponibilidade de produtos nas prateleiras é feita maioritariamente de forma manual, o que é ineficiente, demorado e propenso a erros humanos. Prateleiras vazias representam vendas perdidas e insatisfação do cliente.

O presente projeto propõe uma solução de **Monitorização Inteligente de Prateleiras** baseada em visão por computador, capaz de detetar automaticamente a presença e localização de produtos em prateleiras de armazém ou supermercado. Utilizando modelos de deteção de objetos YOLOv8, o sistema pode processar imagens de prateleiras em tempo real e identificar cada item individual visível.

O cenário-alvo inclui:
- **Supermercados e lojas de retalho:** Monitorização contínua da ocupação das prateleiras para alertar sobre ruturas de stock.
- **Armazéns logísticos:** Verificação da correta colocação e presença de produtos nas localizações designadas.
- **Centros de distribuição:** Auditoria visual automatizada da organização do inventário.

### 1.2. O que o Modelo Deteta

O sistema deteta uma classe única: **item** — qualquer produto individual visível numa prateleira. Cada deteção é representada por uma caixa delimitadora (bounding box) com coordenadas normalizadas e um nível de confiança associado.

**O que o sistema explicitamente NÃO cobre:**
- Identificação da marca, tipo ou categoria específica de cada produto.
- Deteção de prateleiras vazias (apenas se infere indiretamente pela ausência de deteções numa zona).
- Avaliação do estado da embalagem (dano, deformação).

A decisão de utilizar uma classe única fundamenta-se no objetivo principal do caso de uso: monitorizar a **presença/ausência** de produtos para gestão de stock, e não a identificação individual de cada produto. Como trabalho futuro, planeia-se a integração de leitura de códigos QR para identificação de produtos e deteção de zonas sem stock.

### 1.3. Suposições e Restrições

- **Câmara:** Posicionamento frontal ou com ângulo ligeiro em relação à prateleira (até ~30°).
- **Distância:** 0.5 a 3 metros da prateleira.
- **Iluminação:** Condições de iluminação artificial interior (fluorescente ou LED). O sistema não foi validado para exteriores.
- **Fundo:** Prateleiras de retalho convencionais (metálicas ou de madeira) com produtos dispostos de forma organizada.

---

## 2. Descrição do Conjunto de Dados

### 2.1. Recolha de Dados

Foram utilizados dois datasets complementares, ambos focados na deteção de produtos em prateleiras:

**Dataset Primário — Shelves (Dataset Curado):**
- **Fonte:** Plataforma Roboflow, projeto `shelves-exlrc-bvskw`.
- **Imagens:** Recolhidas e curadas a partir de imagens públicas de ambientes de retalho, cobrindo diferentes tipos de prateleiras, ângulos e condições de iluminação.
- **Anotação:** Realizada no Roboflow com bounding boxes em formato YOLOv8 (coordenadas normalizadas).

**Dataset Secundário — SKU-110K (Benchmark Público):**
- **Fonte:** Benchmark público SKU-110K, disponível em Roboflow Universe.
- **Imagens:** ~11 762 imagens diversificadas de prateleiras de retalho, provenientes de múltiplos ambientes comerciais.
- **Justificação:** Utilizado para treinar o terceiro modelo, com o objetivo de avaliar o impacto da dimensão e diversidade do dataset na capacidade de generalização.

### 2.2. Tamanho do Conjunto de Dados por Split e Classe

**Dataset Shelves (utilizado nos modelos Nano 50ep e Small 150ep):**

| Partição    | Nº de Imagens |
|-------------|---------------|
| Treino      | 1462          |
| Validação   | 151           |
| Teste       | 60            |
| **Total**   | **1673**      |

**Dataset SKU-110K (utilizado no modelo Nano SKU):**

| Partição    | Nº de Imagens |
|-------------|---------------|
| Treino      | 8 233         |
| Validação   | 588           |
| Teste       | 2 941         |
| **Total**   | **11 762**    |

Ambos os datasets contêm **1 classe** (`item`), com milhares de instâncias por imagem (tipicamente 20-60 produtos por prateleira).

### 2.3. Política de Etiquetagem

- Cada produto individual visível na prateleira é anotado com uma bounding box.
- Produtos parcialmente ocluídos (>50% visíveis) são anotados.
- Produtos completamente ocluídos são ignorados.
- Quando dois produtos estão sobrepostos, cada um recebe a sua própria caixa delimitadora, mesmo que haja sobreposição.
- Etiquetas de preço e sinalética da loja não são anotados.

### 2.4. Estratégias de Augmentation

**Augmentações aplicadas no Roboflow (dataset Shelves):**

| Augmentação         | Parâmetros              | Justificação                                            |
|---------------------|-------------------------|---------------------------------------------------------|
| Resize              | Stretch para 640×640    | Normalização para input do YOLOv8                       |
| Horizontal Flip     | Ativado                 | Invariância à orientação da prateleira                  |
| Brightness          | -15% a +15%             | Robustez a variações de iluminação                      |
| Blur                | Até 1.25 px             | Simulação de câmaras de menor qualidade                 |
| Multiplicador       | 3×                      | Triplicação dos dados para maior diversidade            |

**Augmentações aplicadas durante o treino (YOLOv8, via Ultralytics):**

| Augmentação  | Valor  | Justificação                                    |
|--------------|--------|-------------------------------------------------|
| Mosaic       | 1.0    | Combina 4 imagens para contexto variado         |
| Flip LR      | 0.5    | Invariância horizontal                          |
| HSV (H/S/V)  | 0.015 / 0.7 / 0.4 | Variação de cor e iluminação          |
| Translate    | 0.1    | Tolerância a posicionamento da câmara           |
| Scale        | 0.5    | Robustez a diferentes distâncias                |
| Erasing      | 0.4    | Simulação de oclusão parcial                    |

### 2.5. Verificações de Qualidade

- **Distribuição por classe:** Verificada como equilibrada, dado que existe apenas 1 classe.
- **Duplicados:** O Roboflow realiza verificação automática de duplicados durante o upload.
- **Consistência de etiquetas:** Verificada visualmente por amostragem aleatória de imagens anotadas.

---

## 3. Modelo(s) e Configuração de Treino

### 3.1. Arquitetura e Pipeline de Treino

Foram treinados **3 modelos** utilizando a framework Ultralytics YOLOv8, com treino local:

| Modelo                | Arquitetura | Dataset     | Épocas | Motivação                                         |
|-----------------------|-------------|-------------|--------|----------------------------------------------------|
| Nano Shelves (v1.0)   | YOLOv8n     | Shelves     | 50     | Modelo base — rápido e leve                        |
| Small Shelves (v2.0)  | YOLOv8s     | Shelves     | 150    | Avaliar se mais parâmetros melhoram desempenho      |
| Nano SKU (v3.0)       | YOLOv8n     | SKU-110K    | 50     | Avaliar impacto de dataset maior e mais diverso    |

O treino foi realizado localmente em vez do Roboflow, utilizando o script `train.py` com a biblioteca Ultralytics. Todas as configurações de treino são reproduzíveis a partir dos ficheiros `train_config.yaml` incluídos em cada pasta de modelo.

### 3.2. Hiperparâmetros

| Parâmetro        | Valor       |
|------------------|-------------|
| Optimizer        | Auto (SGD)  |
| Learning Rate (lr0) | 0.01     |
| LR Final (lrf)  | 0.01        |
| Momentum         | 0.937       |
| Weight Decay     | 0.0005      |
| Warmup Epochs    | 3.0         |
| Batch Size       | 16          |
| Image Size       | 640×640     |
| AMP              | Ativado     |

### 3.3. Hardware Utilizado

- **GPU:** NVIDIA GeForce RTX 4060 (8GB VRAM)
- **CPU:** AMD Ryzen 7 5700X (16 threads @ 4.65 GHz)
- **RAM:** 32 GB DDR4
- **SO:** Windows 11

Tempos de treino:
- Nano Shelves (50 epochs): ~1.1 horas
- Small Shelves (150 epochs): ~4.7 horas
- Nano SKU (50 epochs): ~5.2 horas

---

## 4. Resultados da Avaliação

### 4.1. Métricas Padrão no Conjunto de Validação

| Modelo                | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-----------------------|---------|---------------|-----------|--------|
| Nano Shelves (50ep)   | 0.867   | 0.506         | 0.879     | 0.811  |
| Small Shelves (150ep) | 0.865   | 0.499         | 0.884     | 0.827  |
| Nano SKU (50ep)       | **0.899** | **0.553**   | **0.901** | **0.840** |

### 4.2. Análise de Resultados e Erro

**Observação fundamental:** O modelo Nano treinado no SKU-110K (11k imagens) superou significativamente os dois modelos treinados no dataset Shelves (1.6k imagens), mesmo usando a mesma arquitetura mais leve (Nano). Isto demonstra que a **qualidade e diversidade do dataset** é mais determinante para o desempenho do que o tamanho da arquitetura da rede.

**Comparação Nano vs Small no mesmo dataset:** O modelo Small (150 épocas) obteve métricas praticamente idênticas ao Nano (50 épocas) — mAP50 de 0.865 vs 0.867. Isto indica que o estrangulamento (bottleneck) reside no dataset e não na capacidade da rede. O modelo Nano é preferível por ser mais rápido (~3.2ms vs ~5.5ms).

**Modos comuns de falha:**
- **Falsos positivos:** Etiquetas de preço coloridas podem ser detetadas como produtos quando o limiar de confiança está muito baixo (<0.3).
- **Falsos negativos:** Produtos em zonas de sombra ou com embalagens muito escuras podem não ser detetados.
- **Fusão de bounding boxes:** Produtos muito próximos uns dos outros podem ocasionalmente ser fundidos numa única deteção.

**Mitigações sugeridas:**
- Utilizar limiar de confiança ≥ 0.50 para produção.
- Para ambientes com iluminação deficiente, reduzir para 0.35 com supervisão humana.
- Para produção industrial, considerar treino com dados específicos do ambiente-alvo.

### 4.3. Exemplos Qualitativos

As pastas de cada modelo incluem exemplos visuais de previsões corretas (`val_batch*_pred.jpg`) e as labels reais (`val_batch*_labels.jpg`), bem como matrizes de confusão normalizadas e curvas de treino (`results.png`).

---

## 5. Notas de Implementação

### 5.1. Aplicação de Demonstração

Foi desenvolvida uma aplicação web interativa utilizando **Streamlit**, com 4 funcionalidades principais organizadas em tabs:

1. **Inferência** — Upload de imagem, seleção de modelo via dropdown, visualização com bounding boxes, tabela de deteções estruturada com classe/confiança/coordenadas, e exportação JSON.
2. **Comparar Modelos** — Comparação visual lado a lado entre dois modelos selecionados sobre a mesma imagem.
3. **Webcam ao Vivo** — Feed de vídeo em tempo real com inferência YOLO, permitindo auditoria visual instantânea de prateleiras.
4. **Dashboard** — Painel com métricas comparativas dos 3 modelos, curvas de aprendizagem (lidas dos CSVs reais de treino), e matrizes de confusão.

A interface incorpora design premium com tema escuro, gradientes, micro-animações, e tipografia Google Fonts (Inter).

### 5.2. Velocidade de Inferência e Tamanho dos Modelos

| Modelo                | Tamanho do Ficheiro | Inferência (GPU) | Arquitetura |
|-----------------------|---------------------|-------------------|-------------|
| Nano Shelves          | 6.25 MB             | ~3.2 ms           | YOLOv8n     |
| Small Shelves         | 22.5 MB             | ~5.5 ms           | YOLOv8s     |
| Nano SKU              | 6.25 MB             | ~3.2 ms           | YOLOv8n     |

Os modelos Nano são altamente viáveis para edge computing e dispositivos com recursos limitados.

### 5.3. Parâmetros e Configurações

- **Limiar de confiança recomendado:** 0.50 (padrão). Configurável na barra lateral da aplicação entre 0.10 e 1.00.
- **Resolução de input:** 640×640 (normalização automática pelo YOLOv8).
- **Formato de saída:** Bounding boxes em formato YOLO + JSON exportável com classe, confiança e coordenadas.

### 5.4. Ponto de Entrada para Inferência (CLI)

Além da aplicação gráfica, é fornecido um script CLI (`infer.py`) para inferência direta:

```bash
python infer.py --model modelos/yolov8n_sku/weights.pt --image caminho/imagem.jpg --conf 0.5
```

Este produz:
- Imagem anotada com bounding boxes guardada em disco.
- Saída JSON estruturada com todas as deteções (impressa no terminal e guardada num ficheiro `_detections.json`).

---

## 6. Conclusões e Trabalho Futuro

O projeto demonstrou com sucesso a viabilidade de sistemas de deteção de objetos para monitorização de prateleiras em ambientes industriais. A principal conclusão técnica é que **a dimensão e diversidade do dataset são fatores mais determinantes que a complexidade da arquitetura** — o modelo Nano treinado no SKU-110K (11k imagens) superou o modelo Small treinado num dataset 7× mais pequeno.

**Trabalho futuro planeado:**
- Integração de leitura de códigos QR/barras para identificação automática de produtos.
- Implementação de lógica de deteção de ruturas de stock (colunas sem produtos).
- Treino com múltiplas classes para diferenciação de categorias de produto.
- Validação em ambientes industriais reais com câmaras fixas.

---

## Referências

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO*. https://github.com/ultralytics/ultralytics
2. Goldman, E., et al. (2019). *Precise Detection in Densely Packed Scenes*. CVPR 2019. (Dataset SKU-110K)
3. Roboflow. (2024). *Computer Vision Tools and Platform*. https://roboflow.com
