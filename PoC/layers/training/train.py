#!/usr/bin/env python3
"""
Script principal de treinamento.
Carrega configuração, treina modelos, avalia e salva artefatos.

Usage:
    python training/train.py --config training/config.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# TODO: Descomentar quando implementar os módulos
# from models import train_model, evaluate_model, save_model
# from models.utils import prepare_data, calculate_feature_importance


def setup_logging(config: dict) -> None:
    """Configura logging baseado na config."""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )


def load_config(config_path: Path) -> dict:
    """Carrega arquivo de configuração YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: Path) -> None:
    """
    Fluxo principal de treinamento.
    
    1. Carregar configuração
    2. Preparar dados
    3. Treinar modelos
    4. Avaliar e salvar artefatos
    """
    # Carregar config
    config = load_config(config_path)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Iniciando pipeline de treinamento...")
    
    # TODO: Implementar fluxo completo
    # 1. Carregar dados
    # df = pd.read_csv(config['data']['path'])
    # logger.info(f"✅ Dataset carregado: {df.shape}")
    
    # 2. Preparar dados
    # X_train, X_test, y_train, y_test = prepare_data(
    #     df, 
    #     config['data']['target'],
    #     config['data']['test_size']
    # )
    
    # 3. Treinar modelos
    # for model_name in ['xgboost', 'random_forest']:
    #     logger.info(f"🔧 Treinando {model_name}...")
    #     model = train_model(
    #         X_train, y_train,
    #         model_type=model_name,
    #         **config['models'][model_name]
    #     )
    #     
    #     # 4. Avaliar
    #     metrics = evaluate_model(model, X_test, y_test)
    #     logger.info(f"📊 Métricas {model_name}: {metrics}")
    #     
    #     # 5. Salvar
    #     model_path = Path(config['output']['models_dir']) / f"{model_name}_latest.joblib"
    #     save_model(model, model_path, metadata=metrics)
    #     logger.info(f"💾 Modelo salvo: {model_path}")
    
    logger.info("✅ Pipeline concluído!")
    logger.info("➡️ Próximo passo: executar agent/ para gerar explicações")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar modelos ML')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('training/config.yaml'),
        help='Caminho do arquivo de configuração'
    )
    
    args = parser.parse_args()
    main(args.config)
