"""
Script Universal de Download Roboflow
=====================================
Uso por defeito (lê o .env):
  python download_roboflow.py

Uso para novos datasets:
  python download_roboflow.py --workspace omeuworkspace --project omeuprojecto --version 1 --out submissao/dataset_novo
"""
import os
import shutil
import argparse
from dotenv import load_dotenv
from roboflow import Roboflow

def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download genérico do Roboflow")
    parser.add_argument("--workspace", type=str, default=os.getenv("ROBOFLOW_WORKSPACE"),
                        help="Workspace do Roboflow")
    parser.add_argument("--project", type=str, default=os.getenv("ROBOFLOW_PROJECT"),
                        help="Nome do Projeto no Roboflow")
    parser.add_argument("--version", type=int, default=int(os.getenv("ROBOFLOW_VERSION", 1)),
                        help="Versão do Dataset")
    parser.add_argument("--out", type=str, default="submissao/dataset",
                        help="Pasta de destino (ex: submissao/dataset_pallets)")
    return parser.parse_args()

def download_dataset():
    print("=== Roboflow Dataset Downloader ===")
    args = parse_args()
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or not args.workspace or not args.project:
        print("Erro: API Key, Workspace ou Projeto não estão definidos.")
        print("Verifica o teu ficheiro .env ou passa os argumentos corretamente.")
        return
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(args.workspace).project(args.project)
        
        print(f"A descarregar o dataset {args.project} (v{args.version}) para '{args.out}'...")
        dataset = project.version(args.version).download("yolov8")
        
        pasta_origem = dataset.location
        pasta_destino = args.out
        
        if os.path.exists(pasta_destino):
            shutil.rmtree(pasta_destino)
            
        shutil.move(pasta_origem, pasta_destino)
        
        print(f"\nDownload concluído com sucesso!")
        print(f"O dataset foi movido e guardado em: {pasta_destino}")
    except Exception as e:
        print(f"\nOcorreu um erro ao fazer download: {e}")

if __name__ == "__main__":
    download_dataset()
