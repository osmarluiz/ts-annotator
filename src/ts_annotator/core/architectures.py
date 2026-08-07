"""Quais redes o treinador pode instanciar.

Duas fontes. A PRIMEIRA e' a InceptionTime escrita aqui (`it`), que continua sendo
o padrao e cujos pesos ja' salvos precisam seguir carregando — por isso ela nao
foi trocada pela homonima do tsai, que tem outra estrutura de camadas e nao leria
um checkpoint antigo.

A SEGUNDA e' o `tsai` (https://github.com/timeseriesAI/tsai), quando instalado:
qualquer arquitetura exportada por `tsai.models.all` fica disponivel pelo nome,
sem que nada aqui a conheca. O tsai e' opcional; sem ele, sobra `it`, e a lista de
escolhas encolhe em vez de o programa quebrar.

Os modelos do tsai sao nn.Module comuns. O laco de treino, os pesos por classe, o
agendador e a validacao cruzada continuam sendo os daqui: o tsai entra so' como
catalogo de arquiteturas, nao como framework de treino.
"""
from __future__ import annotations

import inspect

BUILTIN = "it"


def _tsai_models():
    """{nome: classe} do tsai, ou {} se ele nao estiver instalado."""
    try:
        import tsai.models.all as M
    except Exception:
        return {}
    import torch.nn as nn
    out = {}
    for name in dir(M):
        if name.startswith("_"):
            continue
        obj = getattr(M, name)
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            out[name] = obj
    return out


def available():
    """Arquiteturas instanciaveis agora, `it` sempre primeiro."""
    return [BUILTIN] + sorted(_tsai_models())


def has_tsai() -> bool:
    return bool(_tsai_models())


def build(arch, c_in, n_classes, seq_len=None):
    """Instancia a arquitetura pelo nome.

    As assinaturas do tsai variam: quase todas aceitam (c_in, c_out), algumas
    exigem `seq_len`. Em vez de manter uma tabela que envelhece, a assinatura e'
    inspecionada e `seq_len` so' e' passado a quem o declara.
    """
    if arch in (None, "", BUILTIN):
        from ts_annotator.core.trainer import IT
        return IT(c_in, n_classes)
    models = _tsai_models()
    if arch not in models:
        if not models:
            raise ValueError(
                f"arquitetura {arch!r} exige o tsai, que nao esta instalado "
                f"(pip install tsai). Disponivel agora: {BUILTIN}"
            )
        raise ValueError(
            f"arquitetura {arch!r} desconhecida. Disponiveis: {', '.join(available())}"
        )
    cls = models[arch]
    kw = {}
    params = inspect.signature(cls.__init__).parameters
    if "seq_len" in params:
        if seq_len is None:
            raise ValueError(f"{arch} exige seq_len e ele nao foi informado")
        kw["seq_len"] = int(seq_len)
    return cls(int(c_in), int(n_classes), **kw)
