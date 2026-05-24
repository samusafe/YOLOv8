import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import json


# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Warehouse Vision AI | Sistema de Deteção",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&display=swap');

    *, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0e17; }

    /* Header hero */
    .hero-title {
        font-size: 2.4rem; font-weight: 900; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00FFCC 0%, #00B4D8 50%, #7B2FFF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1rem; color: #64748b; margin-top: 4px; margin-bottom: 24px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1420 0%, #0a0e17 100%);
        border-right: 1px solid rgba(0, 255, 204, 0.08);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00FFCC !important; font-weight: 800;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #141824 0%, #0d1117 100%);
        border-radius: 16px; padding: 24px;
        border: 1px solid rgba(0, 255, 204, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center; transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 255, 204, 0.35);
        box-shadow: 0 8px 40px rgba(0, 255, 204, 0.08);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.8rem; font-weight: 900; margin: 0; line-height: 1;
        background: linear-gradient(135deg, #00FFCC, #00B4D8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.8rem; color: #64748b; margin-top: 8px;
        text-transform: uppercase; letter-spacing: 2px; font-weight: 600;
    }
    .metric-card-alt .metric-value {
        background: linear-gradient(135deg, #7B2FFF, #E040FB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* Detection table */
    .det-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    .det-table th {
        background: #141824; color: #00FFCC; padding: 10px 14px;
        text-align: left; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 1.5px; border-bottom: 2px solid rgba(0,255,204,0.2);
    }
    .det-table td {
        padding: 8px 14px; color: #cbd5e1; font-size: 0.85rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .det-table tr:hover td { background: rgba(0,255,204,0.04); }
    .conf-high { color: #00FFCC; font-weight: 700; }
    .conf-mid { color: #fbbf24; font-weight: 700; }
    .conf-low { color: #f87171; font-weight: 700; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #141824; border-radius: 8px 8px 0 0; padding: 10px 20px;
        color: #94a3b8; font-weight: 600; border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1f2e !important; color: #00FFCC !important;
        border-color: rgba(0,255,204,0.3) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00FFCC, #00B4D8) !important;
        color: #0a0e17 !important; font-weight: 800 !important;
        border: none !important; border-radius: 12px !important;
        padding: 12px 24px !important; transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(0,255,204,0.3) !important;
    }

    /* Section headers */
    h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Carregamento dos Modelos
# ==========================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'modelos')

MODEL_REGISTRY = {
    "YOLOv8 Nano — Shelves (50ep)": {
        "path": os.path.join(MODELS_DIR, "yolov8_shelves", "weights.pt"),
        "tag": "nano_shelves", "arch": "yolov8n", "dataset": "Shelves (1.6k)",
        "epochs": 50, "color": "#00FFCC"
    },
    "YOLOv8 Small — Shelves (150ep)": {
        "path": os.path.join(MODELS_DIR, "yolov8s_shelves", "weights.pt"),
        "tag": "small_shelves", "arch": "yolov8s", "dataset": "Shelves (1.6k)",
        "epochs": 150, "color": "#7B2FFF"
    },
    "YOLOv8 Nano — SKU-110K (50ep)": {
        "path": os.path.join(MODELS_DIR, "yolov8n_sku", "weights.pt"),
        "tag": "nano_sku", "arch": "yolov8n", "dataset": "SKU-110K (11.7k)",
        "epochs": 50, "color": "#E040FB"
    },
}

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

def get_available_models():
    available = {}
    for name, info in MODEL_REGISTRY.items():
        if os.path.exists(info["path"]):
            available[name] = info
    return available

def get_detections_df(results):
    """Converte os resultados YOLO numa tabela estruturada."""
    rows = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [round(float(c), 1) for c in box.xyxy[0]]
        rows.append({
            "Classe": cls_name,
            "Confiança": conf,
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2
        })
    return pd.DataFrame(rows)

def render_detections_table(df):
    """Renderiza uma tabela HTML estilizada com as deteções."""
    if df.empty:
        st.info("Nenhuma deteção encontrada com o limiar de confiança atual.")
        return
    html = "<table class='det-table'><thead><tr>"
    html += "<th>#</th><th>Classe</th><th>Confiança</th><th>Caixa (x1,y1 → x2,y2)</th>"
    html += "</tr></thead><tbody>"
    for i, row in df.iterrows():
        conf = row["Confiança"]
        css = "conf-high" if conf >= 0.7 else ("conf-mid" if conf >= 0.5 else "conf-low")
        html += f"<tr><td>{i+1}</td><td>{row['Classe']}</td>"
        html += f"<td class='{css}'>{conf:.1%}</td>"
        html += f"<td>({row['X1']}, {row['Y1']}) → ({row['X2']}, {row['Y2']})</td></tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

def get_results_json(df):
    """Gera a saída JSON estruturada das deteções."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "class": row["Classe"],
            "confidence": round(row["Confiança"], 4),
            "bbox": {"x1": row["X1"], "y1": row["Y1"], "x2": row["X2"], "y2": row["Y2"]}
        })
    return json.dumps({"detections": records, "total": len(records)}, indent=2, ensure_ascii=False)

def load_metrics_from_csv(model_tag):
    """Carrega as métricas reais do ficheiro results.csv de cada modelo."""
    for info in MODEL_REGISTRY.values():
        if info["tag"] == model_tag:
            csv_path = os.path.join(os.path.dirname(info["path"]), "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                last = df.iloc[-1]
                return {
                    "mAP50": round(float(last.get("metrics/mAP50(B)", 0)), 3),
                    "mAP50_95": round(float(last.get("metrics/mAP50-95(B)", 0)), 3),
                    "Precision": round(float(last.get("metrics/precision(B)", 0)), 3),
                    "Recall": round(float(last.get("metrics/recall(B)", 0)), 3),
                }
    return None

# ==========================================
# SIDEBAR
# ==========================================
available_models = get_available_models()

st.sidebar.markdown("## 🧠 Painel de Controlo")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Parâmetros de Inferência")
confidence = st.sidebar.slider(
    "Confiança Mínima", 0.10, 1.0, 0.50, 0.05,
    help="Valores mais altos eliminam deteções com fraca certeza (falsos positivos)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Opções da Webcam")
webcam_model = st.sidebar.selectbox("Modelo para vídeo:", list(available_models.keys()))
rotacao_cam = st.sidebar.selectbox("Orientação:", ["Normal", "Rodar 90° Direita", "Rodar 90° Esquerda"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.75rem; padding:12px 0;'>"
    "Projeto Universitário — Inteligência Artificial<br>"
    "Engenharia Informática · ESTG/IPP · 2025/2026"
    "</div>", unsafe_allow_html=True
)

# ==========================================
# HEADER
# ==========================================
st.markdown("<p class='hero-title'>📦 Warehouse Vision AI</p>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Sistema inteligente de deteção de inventário em prateleiras · Powered by YOLOv8</p>", unsafe_allow_html=True)

# ==========================================
# TABS PRINCIPAIS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Inferência", "⚖️ Comparar Modelos", "📷 Webcam ao Vivo", "📊 Dashboard"
])

# ─────────── TAB 1: INFERÊNCIA ───────────
with tab1:
    col_upload, col_results = st.columns([1, 1.4])

    with col_upload:
        st.markdown("### Inserir Imagem")
        selected_model_name = st.selectbox("Modelo a utilizar:", list(available_models.keys()), key="inf_model")
        uploaded = st.file_uploader("Arraste ou escolha uma imagem", type=["jpg", "jpeg", "png"], key="inf_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        model = load_model(available_models[selected_model_name]["path"])

        with col_results:
            st.markdown("### Resultados")
            with st.spinner("A processar inferência na GPU..."):
                results = model.predict(source=image, conf=confidence, verbose=False)
                annotated = results[0].plot()[..., ::-1]
                st.image(annotated, width="stretch")

                df = get_detections_df(results)
                num = len(df)

            # Métricas
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"<div class='metric-card'><p class='metric-value'>{num}</p>"
                            f"<p class='metric-label'>Produtos Detetados</p></div>", unsafe_allow_html=True)
            with m2:
                avg_conf = f"{df['Confiança'].mean():.1%}" if num > 0 else "N/A"
                st.markdown(f"<div class='metric-card metric-card-alt'><p class='metric-value'>{avg_conf}</p>"
                            f"<p class='metric-label'>Confiança Média</p></div>", unsafe_allow_html=True)

        # Tabela de deteções estruturada (exigido pelo enunciado D4!)
        st.markdown("### 📋 Saída Estruturada — Deteções Individuais")
        render_detections_table(df)

        # Botão de exportação JSON
        if num > 0:
            json_out = get_results_json(df)
            st.download_button(
                "⬇️ Exportar Resultados (JSON)",
                data=json_out,
                file_name="detections.json",
                mime="application/json"
            )

# ─────────── TAB 2: COMPARAR MODELOS ───────────
with tab2:
    st.markdown("### Comparação Visual entre Modelos")
    st.markdown("Escolhe dois modelos e carrega uma imagem para comparar as deteções lado a lado.")

    model_names = list(available_models.keys())
    if len(model_names) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            model_a_name = st.selectbox("Modelo A (Esquerda):", model_names, index=0, key="cmp_a")
        with c2:
            default_b = min(2, len(model_names) - 1)
            model_b_name = st.selectbox("Modelo B (Direita):", model_names, index=default_b, key="cmp_b")

        uploaded_cmp = st.file_uploader("Imagem para comparação", type=["jpg", "jpeg", "png"], key="cmp_upload")

        if uploaded_cmp:
            img = Image.open(uploaded_cmp).convert("RGB")
            model_a = load_model(available_models[model_a_name]["path"])
            model_b = load_model(available_models[model_b_name]["path"])

            with st.spinner("A comparar modelos..."):
                res_a = model_a.predict(source=img, conf=confidence, verbose=False)
                res_b = model_b.predict(source=img, conf=confidence, verbose=False)

                img_a = res_a[0].plot()[..., ::-1]
                img_b = res_b[0].plot()[..., ::-1]

                items_a = len(res_a[0].boxes)
                items_b = len(res_b[0].boxes)

                # Layout Lado a Lado (2 Colunas Esticadas)
                col_img_a, col_img_b = st.columns(2)
                
                with col_img_a:
                    st.markdown(f"##### Deteções: {model_a_name.split('—')[0].strip()}")
                    st.image(img_a, width="stretch")
                    st.markdown(f"<div class='metric-card'><p class='metric-value'>{items_a}</p>"
                                f"<p class='metric-label'>Produtos Detetados (Modelo A)</p></div>",
                                unsafe_allow_html=True)
                                
                with col_img_b:
                    st.markdown(f"##### Deteções: {model_b_name.split('—')[0].strip()}")
                    st.image(img_b, width="stretch")
                    st.markdown(f"<div class='metric-card metric-card-alt'><p class='metric-value'>{items_b}</p>"
                                f"<p class='metric-label'>Produtos Detetados (Modelo B)</p></div>",
                                unsafe_allow_html=True)
    else:
        st.warning("São necessários pelo menos 2 modelos disponíveis para comparação.")

# ─────────── TAB 3: WEBCAM AO VIVO ───────────
with tab3:
    st.markdown("### 📷 Auditoria em Tempo Real")
    st.markdown("O feed de vídeo corre diretamente no browser. Seleciona o modelo e a orientação na barra lateral.")
    st.info(f"Modelo ativo: **{webcam_model}** · Confiança: **{confidence*100:.0f}%**")

    run_webcam = st.checkbox("🟢 Ligar Câmara", key="webcam_toggle")
    FRAME_WINDOW = st.empty()

    if run_webcam:
        cam = cv2.VideoCapture(0)
        model_cam = load_model(available_models[webcam_model]["path"])

        while run_webcam:
            ret, frame = cam.read()
            if not ret:
                st.error("Não foi possível aceder à webcam.")
                break

            if rotacao_cam == "Rodar 90° Direita":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotacao_cam == "Rodar 90° Esquerda":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if model_cam:
                res = model_cam.predict(source=frame, conf=confidence, verbose=False)
                plotted = res[0].plot()
                FRAME_WINDOW.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), width="stretch")

        cam.release()

# ─────────── TAB 4: DASHBOARD ───────────
with tab4:
    st.markdown("### 📊 Dashboard Comparativo de Modelos")
    st.markdown("Métricas lidas diretamente dos ficheiros de treino (`results.csv`) de cada modelo.")

    # Construir tabela de métricas REAL (lida dos CSVs)
    metric_rows = []
    for name, info in available_models.items():
        m = load_metrics_from_csv(info["tag"])
        if m:
            metric_rows.append({
                "Modelo": name,
                "Arquitetura": info["arch"],
                "Dataset": info["dataset"],
                "Épocas": info["epochs"],
                "mAP@0.5": f"{m['mAP50']:.1%}",
                "mAP@0.5:0.95": f"{m['mAP50_95']:.1%}",
                "Precision": f"{m['Precision']:.1%}",
                "Recall": f"{m['Recall']:.1%}",
            })

    if metric_rows:
        st.table(pd.DataFrame(metric_rows).set_index("Modelo"))

    # Gráficos de treino — lado a lado por modelo
    st.markdown("---")
    st.markdown("### 📈 Curvas de Aprendizagem")

    graph_cols = st.columns(len(available_models))
    for idx, (name, info) in enumerate(available_models.items()):
        model_dir = os.path.dirname(info["path"])
        results_img = os.path.join(model_dir, "results.png")
        with graph_cols[idx]:
            short_name = name.split("—")[1].strip() if "—" in name else name
            st.markdown(f"#### {short_name}")
            if os.path.exists(results_img):
                st.image(results_img, width="stretch")
            else:
                st.caption("Gráfico não disponível")

    st.markdown("---")
    st.markdown("### 📉 Matrizes de Confusão")

    cm_cols = st.columns(len(available_models))
    for idx, (name, info) in enumerate(available_models.items()):
        model_dir = os.path.dirname(info["path"])
        cm_img = os.path.join(model_dir, "confusion_matrix.png")
        with cm_cols[idx]:
            if os.path.exists(cm_img):
                st.image(cm_img, width="stretch")
            else:
                st.caption("Matriz não disponível")
