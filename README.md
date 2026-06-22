# 🛒 Monitorização Inteligente de Prateleiras com YOLOv8

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple.svg)

Este repositório contém uma solução de Visão Computacional *end-to-end* para deteção de objetos em ambientes industriais e de retalho, baseada na arquitetura **YOLOv8**. O projeto foca-se na **Monitorização Inteligente de Prateleiras**, permitindo a deteção automática e em tempo real de produtos para otimização da gestão de inventário, auditoria visual e prevenção de ruturas de stock.

## 📌 Caso de Uso e Motivação

A gestão de inventário em ambientes de retalho e armazéns logísticos constitui um desafio operacional significativo que consome muitos recursos de forma manual. Este sistema foi desenhado para atuar sobre os seguintes cenários:
- **Supermercados e Retalho:** Monitorização contínua para alertar atempadamente sobre ruturas de stock nas prateleiras.
- **Armazéns Logísticos:** Verificação e validação da correta colocação de produtos ou paletes.
- **Centros de Distribuição:** Auditoria visual automatizada e em larga escala.

O modelo deteta a classe generalizada `item` — focando-se na presença ou ausência física de produtos de forma a mapear a ocupação, garantindo assim alta capacidade de generalização e rapidez.

## 🛠️ Tecnologias e Ferramentas Utilizadas

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics):** Arquitetura State-of-the-Art para treino de modelos e deteção de objetos ultrarrápida (útil em Edge Computing).
- **[Roboflow](https://roboflow.com/):** Etiquetagem (anotação), verificação de qualidade, *data augmentation* e gestão de *datasets*.
- **[Streamlit](https://streamlit.io/):** Framework para a rápida criação e implementação de aplicações web orientadas a Machine Learning.
- **Python:** A base tecnológica de todo o pipeline de processamento e modelação.

## 🚀 Componentes da Aplicação de Demonstração

Para facilitar a integração e demonstração do sistema, foi desenvolvida uma aplicação interactiva baseada em Streamlit com **5 funcionalidades principais**:

1. **🔍 Inferência Base:** Permite o *upload* de uma imagem, seleção do modelo a testar via *dropdown*, visualização instantânea com *bounding boxes*, além de exportar uma tabela estruturada e um relatório em JSON.
2. **⚖️ Comparador de Modelos:** Visualização sincronizada lado-a-lado das previsões de dois modelos distintos sobre a mesma imagem, ideal para analisar trade-offs de precisão/velocidade.
3. **🎥 Monitorização via Webcam ao Vivo:** Feed de vídeo contínuo com inferência do YOLO, permitindo auditorias dinâmicas.
4. **📊 Dashboard Analítico:** Painel com as métricas técnicas comparativas dos modelos, matrizes de confusão e curvas de aprendizagem extraídas diretamente dos logs de treino.
5. **📜 Registo e Histórico:** Visualização de resultados passados da plataforma.

## 📊 Datasets e Resultados dos Modelos

Foi adotada uma estratégia iterativa de treino sobre múltiplos datasets para avaliar qual o impacto do volume e diversidade de imagens na performance do sistema.

| Dataset | Total Imagens | Descrição e Propósito |
| :--- | :--- | :--- |
| **Shelves** (Custom) | 1 673 | Dataset criado manualmente para criar modelos Nano e Small focados em prateleiras de retalho. |
| **SKU-110K** (Público)| 11 762 | Benchmark público de grande escala para provar a escalabilidade do modelo e melhorar generalização. |
| **Pallets** (Custom) | 972 | Domínio alternativo (logística pesada/paletes) para validar técnicas de *Transfer Learning*. |

Todos os modelos beneficiaram de técnicas avançadas de *Data Augmentation* nativas ao YOLO (Mosaic, HSV transform, Translate, Erasing) bem como processamentos via Roboflow. O treino resultou em métricas de alta fiabilidade no conjunto de Validação/Teste:

| Modelo Desenvolvido | Arquitetura | Base de Treino | Épocas | mAP@0.5 | Precisão | Recall | Speed (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nano Shelves v1.0** | YOLOv8n | Shelves | 50 | 86.7% | 87.9% | 81.1% | ~3.2 ms |
| **Small Shelves v2.0**| YOLOv8s | Shelves | 150 | 86.5% | 88.4% | 82.7% | ~5.5 ms |
| **Nano SKU v3.0** | YOLOv8n | SKU-110K | 50 | **89.9%** | **90.1%** | 84.0% | ~3.2 ms |
| **Nano Pallets v4.0** | YOLOv8n | Pallets | 50 | **98.7%** | 96.0% | 97.6% | ~3.2 ms |

> 💡 **Principal Insight:** O modelo Nano treinado num dataset substancial (SKU-110K com ~11k imagens) superou na generalização um modelo maior (YOLOv8s) exposto a menos dados (1.6k). A dimensão e diversidade do dataset revelaram-se fatores mais determinantes para o desempenho que a profundidade da arquitetura da rede.

## 💻 Guia de Instalação e Utilização

### 1. Pré-Requisitos
Recomenda-se a utilização de uma versão de **Python 3.9+** e de um ambiente virtual isolado (como `venv` ou `conda`).

### 2. Preparação do Ambiente
```bash
# Clone este repositório
git clone https://github.com/samusafe/YOLOv8.git
cd YOLOv8

# Ative um ambiente virtual e instale os pacotes (exemplo com venv)
python -m venv venv
source venv/bin/activate  # em windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Iniciar a Aplicação Interface Gráfica
A aplicação Streamlit encontra-se no diretório `app` (ou equivalente no repositório final).
```bash
streamlit run app/app.py
```

### 4. Inferência Direta via CLI
Para automatização e rápida inferência usando a Linha de Comandos, pode correr:
```bash
python infer.py --model modelos/yolov8n_sku/weights.pt --image caminho/para/imagem.jpg --conf 0.50
```
O script gerará a imagem contendo as *bounding boxes* aplicadas, bem como um ficheiro `_detections.json` estruturado de suporte à integração com outros sistemas industriais.
