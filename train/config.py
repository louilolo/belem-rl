"""Carrega train/config.yaml -- hiperparâmetros e espaço de busca do Optuna num lugar só,
em vez de espalhados pelos scripts de train/ppo/ e train/dqn/."""
import os

import yaml

_CAMINHO_CONFIG = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Parâmetros do ambiente/recompensa, não do modelo (PPO/DQN) -- os scripts que montam os
# kwargs do modelo tiram esses antes de passar pra criar_modelos().
NAO_SAO_HIPERPARAMS_MODELO = {'min_green', 'peso_recompensa'}


def carregar_config():
    with open(_CAMINHO_CONFIG) as f:
        return yaml.safe_load(f)


def sugerir_do_espaco(trial, nome, espec):
    """Sorteia 1 hiperparâmetro do Optuna a partir de uma entrada do espaço de busca
    (dict com 'tipo': float/int/categorical -- ver train/config.yaml)."""
    tipo = espec['tipo']
    if tipo == 'float':
        return trial.suggest_float(nome, espec['min'], espec['max'], log=espec.get('log', False))
    if tipo == 'int':
        return trial.suggest_int(nome, espec['min'], espec['max'])
    if tipo == 'categorical':
        return trial.suggest_categorical(nome, espec['valores'])
    raise ValueError(f"tipo de hiperparâmetro desconhecido em config.yaml: {tipo!r}")


def montar_hiperparams_trial(trial, config_algo, pular=()):
    """Monta o dict de hiperparâmetros pra 1 trial: começa dos valores fixos em
    config_algo['hyperparams'], e sobrescreve só os listados em
    config_algo['optuna']['search_params'] com um valor sorteado do espaco_busca.
    Hiperparâmetros em 'pular' (ex: peso_recompensa quando a recompensa não usa peso)
    ficam de fora da busca mesmo que estejam em search_params.
    """
    hiperparams = dict(config_algo['hyperparams'])
    espaco = config_algo['optuna']['espaco_busca']
    for nome in config_algo['optuna']['search_params']:
        if nome in pular:
            continue
        hiperparams[nome] = sugerir_do_espaco(trial, nome, espaco[nome])
    return hiperparams
