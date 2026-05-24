# Dataset — Deteção de Produtos em Prateleiras

## 1. Origem e Fonte

- **Plataforma:** [Roboflow](https://roboflow.com) — projeto `shelves-exlrc-bvskw`
- **Curadoria:** Dataset curado para fins académicos, baseado em imagens de ambientes de retalho/supermercado.
- **Licença:** Utilização académica.

## 2. Formato

- **Formato de anotação:** YOLOv8 (bounding boxes normalizadas)
- **Estrutura dos ficheiros de etiqueta (.txt):**
  Cada linha contém:
  ```
  class_id x_center y_center width height
  ```
  Todos os valores de coordenadas e dimensões são normalizados entre **0** e **1**, relativamente à largura e altura da imagem.

## 3. Classes

| ID | Nome   | Descrição                                      |
|----|--------|-------------------------------------------------|
| 0  | `item` | Qualquer produto individual visível na prateleira |

> **Nota:** O dataset contém apenas **1 classe**. Não existe diferenciação por tipo, marca ou categoria de produto — apenas a presença e localização de cada item.

## 4. Estatísticas

| Partição    | Nº de Imagens |
|-------------|---------------|
| Treino      | 1462          |
| Validação   | 151           |
| Teste       | 60            |

- **Total de imagens geradas (com augmentations):** ~1673
- **Partição de teste:** A partição de teste foi extraída a partir do conjunto de validação original, garantindo que as imagens de teste não foram usadas durante o treino nem a validação dos modelos.

## 5. Augmentações Aplicadas (Roboflow)

As seguintes transformações foram aplicadas automaticamente durante a exportação do dataset pelo Roboflow:

| Augmentação         | Parâmetros              |
|---------------------|-------------------------|
| Resize              | Stretch para 640×640    |
| Horizontal Flip     | Ativado                 |
| Brightness          | -15% a +15%             |
| Blur                | Até 1.25 px             |
| Multiplicador       | 3× (triplicação dos dados) |

## 6. Limitações Conhecidas

- **Classe única:** O dataset não permite diferenciar tipos ou categorias de produtos — apenas deteta a presença genérica de um item.
- **Enviesamento ambiental:** As imagens são predominantemente provenientes de ambientes de retalho e supermercados, com iluminação artificial. O desempenho em contextos visuais distintos (e.g., armazéns escuros, prateleiras ao ar livre) não foi avaliado.
- **Dimensão limitada:** Com ~1673 imagens, o dataset é relativamente pequeno para tarefas de deteção de objetos, o que pode limitar a capacidade de generalização — uma conclusão validada pela comparação com o modelo treinado no SKU-110K (11k imagens).
