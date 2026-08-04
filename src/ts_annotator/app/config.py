"""Generic app configuration constants (project-independent)."""

# Dynamic prediction basemap. CONTRACT: this string is simultaneously the display
# label AND the key in the window's basemaps dict AND the combo-text compared by
# the controller — single definition here so renaming can't silently break logic.
PRED_LABEL = "Predição (modelo)"

# Classes que NÃO entram no treino nem na similaridade — bucket "don't know":
# pontos marcados à mão só p/ inspecionar depois. Ficam no dataset, mas não viram
# rótulo do modelo nem poluem as sugestões de similaridade.
NON_TRAINING_CLASSES = {"nao_sei"}


def order_classes(names):
    """Ordena nomes de classe deixando as NON_TRAINING (nao_sei) SEMPRE por último
    (classe especial de inspeção). Estável: mantém a ordem original no resto."""
    return sorted(names, key=lambda n: (1 if n in NON_TRAINING_CLASSES else 0,))


# Estratégias de sugestão ativa (aba Rotular). Cada uma declara a DIREÇÃO de seleção
# (metric, order) e o REQUISITO p/ rodar:
#   "model" -> precisa de um modelo treinado; classifica os candidatos NA HORA na área
#             de busca (NÃO exige o mapa todo pré-classificado). As 4 métricas usam o
#             vetor de probabilidades do modelo, que o raster de rótulo não guarda.
#   "curve" -> precisa de uma curva clicada no mapa (protótipo).
#   "class" -> precisa de uma classe-alvo escolhida abaixo.
# Campos: (label, metric, order, req, hint)
GOALS = [
    ("Modelo incerto", "confidence", "asc", "model",
     "pontos onde a MAIOR probabilidade do modelo é baixa — bom uso geral p/ achar o que ele não sabe"),
    ("Modelo ambíguo", "margin", "asc", "model",
     "pontos onde as 2 melhores classes quase empatam — bom p/ refinar a FRONTEIRA entre duas classes"),
    ("Modelo muito incerto", "entropy", "desc", "model",
     "probabilidade espalhada entre MUITAS classes (entropia alta) — bom p/ achar confusão/classes faltando"),
    ("Modelo × similaridade", "disagreement", "desc", "model",
     "o modelo diz uma classe, a similaridade com os teus rótulos diz outra — bom p/ achar erro sistemático"),
    ("Parecidas com a curva", "similar_current", "desc", "curve",
     "os pixels da área MAIS parecidos com a curva que você clicou — clique numa curva antes"),
    ("Exemplos de uma classe", "class", "desc", "class",
     "os candidatos MAIS parecidos com a classe escolhida abaixo — bom p/ reforçar uma classe fraca"),
]

# Cabeçalhos (não selecionáveis) que agrupam as estratégias por REQUISITO no dropdown.
# (título do grupo, requisitos que caem nele)
GOAL_GROUPS = [
    ("— precisam de modelo —", ("model",)),
    ("— precisam de escolha —", ("curve", "class")),
]


def goal_data(label, metric, order, req):
    """Payload guardado no item do combo (o que o controller lê em currentData)."""
    return {"metric": metric, "order": order, "req": req, "label": label}
