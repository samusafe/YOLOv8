# Dataset — SKU-110K (Benchmark Público)

## 1. Visão Geral
- **Nome:** SKU-110K (Store Keeping Unit)
- **Fonte:** Benchmark público, descarregado via Roboflow Universe
- **URL original:** [SKU-110K no Roboflow](https://universe.roboflow.com/large-benchmark-datasets/sku-110k-dataset)
- **Formato:** YOLOv8 (coordenadas de caixas delimitadoras normalizadas)
- **Licença:** CC BY 4.0

## 2. Descrição das Classes
O dataset foi otimizado para deteção genérica e possui apenas uma classe:
- `0: item` (Representa qualquer produto logístico ou de retalho acondicionado numa prateleira).

*Nota: Originalmente o dataset podia ter a classe nomeada como "object", mas foi unificada para "item" de forma a manter compatibilidade com o dataset primário deste projeto.*

## 3. Semântica das Etiquetas (Labels)
O dataset utiliza a anotação padrão do YOLO. Cada imagem tem um ficheiro `.txt` correspondente onde cada linha representa um objeto detetado no formato:
`class_id center_x center_y width height`
(Os valores x, y, width e height estão normalizados entre 0.0 e 1.0).

## 4. Estatísticas do Dataset
O dataset é massivo e caracteriza-se por uma densidade extrema de objetos por imagem (prateleiras de supermercado repletas).
- **Total de imagens:** 11 762 imagens
- **Treino (Train):** 8 233 imagens
- **Validação (Valid):** 588 imagens
- **Teste (Test):** 2 941 imagens
- **Total de instâncias (Bounding Boxes):** Aproximadamente 1.7 milhões de produtos anotados.

## 5. Limitações e Características Conhecidas
- **Alta Densidade:** As imagens contêm produtos muito próximos e sobrepostos. O modelo treinado nestes dados torna-se excelente a separar objetos, mas pode gerar falsos positivos em texturas densas.
- **Contexto Fixo:** A esmagadora maioria das imagens retrata cenários clássicos de supermercado. Modelos treinados exclusivamente nestes dados podem não generalizar bem para objetos do quotidiano isolados noutros contextos (ex: um pacote em cima de uma mesa).
- **Caixas Agrupadas:** Em zonas de fraca visibilidade, o dataset original por vezes agrupa vários produtos numa só caixa delimitadora, o que pode influenciar a rede neural a fundir objetos semelhantes.

## 6. Nota sobre a Submissão
> **As imagens deste dataset NÃO estão incluídas neste ficheiro .zip** devido à sua dimensão (~11 762 imagens, vários GB). O dataset completo pode ser descarregado gratuitamente a partir do URL indicado na secção 1. Apenas a documentação e configuração são incluídas para referência e reprodutibilidade.
