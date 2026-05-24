# Warehouse Vision AI — Aplicação de Demonstração

## Descrição

Aplicação web desenvolvida com **Streamlit** para demonstração do sistema de deteção de produtos em prateleiras de armazém, utilizando modelos YOLOv8 treinados com aceleração GPU (NVIDIA CUDA).

## Funcionalidades

| Tab | Descrição |
|---|---|
| 🔍 **Inferência** | Upload de imagem → seleção de modelo → visualização com bounding boxes + tabela de deteções + exportação JSON |
| ⚖️ **Comparar Modelos** | Slider arrastável para comparação visual entre qualquer par dos 3 modelos |
| 📷 **Webcam ao Vivo** | Deteção em tempo real via câmara do computador |
| 📊 **Dashboard** | Métricas de avaliação, curvas de treino e matrizes de confusão comparativas |

## Modelos Disponíveis

- **YOLOv8 Nano** — Shelves Dataset (50 épocas)
- **YOLOv8 Small** — Shelves Dataset (150 épocas)
- **YOLOv8 Nano** — SKU-110K Benchmark (50 épocas)

## Instalação e Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar a aplicação
streamlit run app.py
```

## Requisitos de Sistema

- Python 3.9+
- NVIDIA GPU com CUDA (recomendado para inferência rápida)
- Webcam (opcional, para a funcionalidade de deteção ao vivo)
