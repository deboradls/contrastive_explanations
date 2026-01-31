from typing import Dict, Tuple, List, Set, Any, Dict as TDict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier

from pysat.formula import IDPool, WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc

import time
import random
from statistics import mean

import pandas as pd
from ucimlrepo import fetch_ucirepo

# ----------------------------- utilitários comuns -----------------------------

def collect_thresholds(models: List[Any]) -> Dict[int, List[float]]:
    per_feat: Dict[int, Set[float]] = {} 
    for m in models: 
        estimators = [m] if hasattr(m, "tree_") else list(m.estimators_) 
        for est in estimators: 
            tr = est.tree_ 
            for f, t in zip(tr.feature, tr.threshold):
                if f != _tree.TREE_UNDEFINED: 
                    per_feat.setdefault(int(f), set()).add(float(t)) 
    return {j: sorted(list(vals)) for j, vals in per_feat.items()} 

def add_atleast_k(w: WCNF, lits: List[int], k: int, pool: IDPool): 
    if k <= 0: return # 
    enc = CardEnc.atleast(lits=lits, bound=k, vpool=pool, encoding=1)
    for cls in enc.clauses: w.append(cls)  

def add_atmost_one(w: WCNF, lits: List[int]):
    # pairwise
    for i in range(len(lits)):
        for j in range(i+1, len(lits)):
            w.append([-lits[i], -lits[j]])           

def add_sigma_monotonicity(w: WCNF, thresholds: Dict[int,List[float]], yvars: Dict[Tuple[int,float], int]):
    # y(j, t_high) -> y(j, t_low)  quando t_high > t_low
    for j, ts in thresholds.items():
        for i in range(len(ts)-1):
            t_low, t_high = ts[i], ts[i+1]
            w.append([-yvars[(j, t_high)], yvars[(j, t_low)]])           

def add_soft_tx(w: WCNF, x: np.ndarray, thresholds: Dict[int,List[float]], yvars: Dict[Tuple[int,float], int]):
    # y_{j,t} := (x_j > t)  -> soft units
    for j, ts in thresholds.items():
        for t in ts:
            y = yvars[(j, t)]
            if x[j] > t:
                w.append([y], weight=1)
            else:
                w.append([-y], weight=1)              

def diff_cost_from_model(model, yvars: Dict[Tuple[int,float], int], thresholds: Dict[int,List[float]], x: np.ndarray):
    if model is None: return None, []
    pos = set(l for l in model if l > 0)
    changes = []
    for j, ts in thresholds.items():
        for t in ts:
            v = yvars[(j, t)]
            want = 1 if x[j] > t else 0
            got  = 1 if (v in pos) else 0
            if got != want:
                changes.append((j, t, got))  # got=1 -> ">" ; got=0 -> "<="
    return len(changes), changes

def fmt_changes(changes, feature_names):
    out = []
    for j,t,got in sorted(changes, key=lambda z:(z[0], z[1])):
        sign = ">" if got==1 else "<="
        out.append(f"{feature_names[j]} {sign} {t:.3f}")
    return out

# ----------------------------- caminhos de árvore -----------------------------

def enumerate_paths_to_leaves(dt: DecisionTreeClassifier):
    """Retorna lista de caminhos. Cada caminho: [(feat, thr, dir)], dir in {'L','R'}, e a classe da folha."""
    tr = dt.tree_
    paths = []
    stack = [(0, [])]  # (node_id, path_so_far)

    while stack:
        nid, path = stack.pop()
        f = tr.feature[nid]
        if f == _tree.TREE_UNDEFINED:
            leaf_cls = int(np.argmax(tr.value[nid][0]))
            paths.append((path, leaf_cls))
        else:
            thr = float(tr.threshold[nid])
            left, right = tr.children_left[nid], tr.children_right[nid]
            stack.append((right, path + [(int(f), thr, 'R')]))  # x>thr
            stack.append((left,  path + [(int(f), thr, 'L')]))  # x<=thr
    return paths  # list of (path, class)

def enumerate_target_paths_tree(dt: DecisionTreeClassifier, target_class: int):
    paths = enumerate_paths_to_leaves(dt)
    return [p for p,c in paths if c == target_class]

# ----------------------------- ÁRVORE: resolver por caminho -----------------------------

def solve_tree_min_changes(dt: DecisionTreeClassifier, x: np.ndarray, target_class: int,
                           feature_names: List[str]) -> Tuple[int, List[str], List[Tuple[int,float,str]]]:
    """Retorna (custo, lista strings mudanças, caminho escolhido)"""
    all_thresholds = collect_thresholds([dt])
    best = None  # (cost, changes, path)

    target_paths = enumerate_target_paths_tree(dt, target_class)
    if not target_paths:
        return None, [], []  # sem folha alvo

    for path in target_paths:
        pool = IDPool()
        w = WCNF()

        # vars y(j,t)
        y = {(j,t): pool.id(('y', j, t)) for j, ts in all_thresholds.items() for t in ts}
        # Σ e Csoft
        add_sigma_monotonicity(w, all_thresholds, y)
        add_soft_tx(w, x, all_thresholds, y)

        # f = conjunção dos testes do caminho
        for (feat, thr, d) in path:
            lit = y[(feat, thr)]
            w.append([ lit] if d=='R' else [-lit])  # R: x>thr ; L: x<=thr

        # solve
        with RC2(w) as rc2:
            m = rc2.compute()
        cost, changes = diff_cost_from_model(m, y, all_thresholds, x)
        if cost is None:  # deveria não acontecer para caminho válido
            continue
        if (best is None) or (cost < best[0]):
            best = (cost, fmt_changes(changes, feature_names), path)

    return best if best is not None else (None, [], [])

# ----------------------------- FLORESTA: maioria via disjunção de caminhos -----------------------------

# Percorre cada arvore do modelo de decisao randonforest treinado, e retorna 
# os caminhos que levam a classe alvo 
def get_target_paths_per_tree(rf: RandomForestClassifier, target_class: int):
    """Retorna lista de caminhos alvo por árvore."""
    per_tree = []
    for est in rf.estimators_:
        per_tree.append(enumerate_target_paths_tree(est, target_class))
    return per_tree  # list of list-of-paths

# prepara o ambiente pra resolver o problema lógico com base nas regras aprendidas pelo modelo 
def setup_solver(rf: RandomForestClassifier, x: np.ndarray): # debug: x = array([4.8, 3.4, 1.6, 0.2])
    """Cria pool, WCNF, thresholds e variáveis y(j,t)."""
    pool = IDPool()
    w = WCNF()
    thresholds = collect_thresholds([rf]) # debug: {3: [0.6000000014901161, 0.6500000059604645,...]}
    y_vars = {(j, t): pool.id(('y', j, t)) for j, ts in thresholds.items() for t in ts} # debug: {(3, 0.6000000014901161): 1, (3, 0.6500000059604645): 2...}
    return pool, w, thresholds, y_vars

def add_tree_constraints(w: WCNF, pool: IDPool, y_vars: Dict[Tuple[int,float], int],
                         per_tree_paths: List[List[List[Tuple[int,float,str]]]]):
    """Adiciona variáveis z_t, k_{t,p} e todas as constraints de árvores."""
    z_vars = []
    k_vars = {}
    for t_idx, paths in enumerate(per_tree_paths):
        z_t = pool.id(('z', t_idx))
        z_vars.append(z_t)

        if not paths:
            w.append([-z_t])
            continue

        k_list = []
        for p_idx, path in enumerate(paths): 
            k = pool.id(('k', t_idx, p_idx))
            k_vars[(t_idx, p_idx)] = k
            k_list.append(k)
            
            # se um caminho é escolhido, todos os testes desse caminho devem ser satisfeitos pelos atributos da instância
            for feat, thr, d in path: # debug: path = [(3, 0.800000011920929, 'L')] == (x(3) <= 0.8?) 
                lit = y_vars[(feat, thr)]
                w.append([-k, lit] if d == 'R' else [-k, -lit])
            # k -> z_t
            w.append([-k, z_t])
        # z_t -> (∨ k)
        w.append([-z_t] + k_list)
        add_atmost_one(w, k_list)

    return z_vars, k_vars

# adiciona restrições de variáveis need serem verdadeiras
def add_majority_constraint(w: WCNF, z_vars: List[int], rf: RandomForestClassifier):
    need = (len(rf.estimators_) // 2) + 1
    pool = IDPool()  # necessário para add_atleast_k
    add_atleast_k(w, z_vars, need, pool)

def extract_solver_result(m, y_vars, thresholds, x, feature_names, k_vars, per_tree_paths):
    """Decodifica resultado do solver: custo, mudanças legíveis e caminhos escolhidos."""
    if m is None:
        return None, [], {}
    cost, changes = diff_cost_from_model(m, y_vars, thresholds, x)
    changes_fmt = fmt_changes(changes, feature_names)
    pos = set(l for l in m if l > 0)
    chosen_paths = {}
    for (t_idx, p_idx), kv in k_vars.items():
        if kv in pos:
            chosen_paths.setdefault(t_idx, []).append(per_tree_paths[t_idx][p_idx])
    return cost, changes_fmt, chosen_paths

# Encontrar o menor número de mudanças necessárias na instância x para que a floresta rf vote na classe target_class.
def solve_forest_min_changes(rf: RandomForestClassifier, x: np.ndarray, target_class: int,
                                     feature_names: List[str]) -> Tuple[int, List[str], Dict[int, List[Tuple[int,float,str]]]]:

    per_tree_paths = get_target_paths_per_tree(rf, target_class)
    pool, w, thresholds, y_vars = setup_solver(rf, x)
    add_sigma_monotonicity(w, thresholds, y_vars)
    add_soft_tx(w, x, thresholds, y_vars)
    z_vars, k_vars = add_tree_constraints(w, pool, y_vars, per_tree_paths)
    add_majority_constraint(w, z_vars, rf)

    with RC2(w) as rc2:
        m = rc2.compute()

    return extract_solver_result(m, y_vars, thresholds, x, feature_names, k_vars, per_tree_paths)

# ----------------------------- Avaliação / métricas -----------------------------

def evaluate_explanations(
    dt: DecisionTreeClassifier,
    rf: RandomForestClassifier,
    Xtest: np.ndarray,
    n_classes: int,
    feature_names: List[str],
    n_samples: int = 30,
    random_seed: int = 0,
):
    """
    Roda explicações contrastivas em n_samples instâncias aleatórias do conjunto de teste
    e retorna estatísticas agregadas:
      - tempo médio (árvore, floresta)
      - custo médio (# mudanças) (árvore, floresta)
      - tamanho médio da explicação (árvore: len(path); floresta: soma len(paths))
      - contagens de exemplos válidos usados (cada método pode não achar solução para alguns alvos)
    """
    if n_samples <= 0:
        raise ValueError("n_samples deve ser > 0")

    rng = random.Random(random_seed)
    indices = rng.sample(range(len(Xtest)), min(n_samples, len(Xtest)))

    # métricas coletadas
    tree_times = []
    forest_times = []

    tree_costs = []
    forest_costs = []

    tree_sizes = []    # tamanho dos caminhos da árvore
    forest_sizes = []  # tamanho total dos caminhos usados na floresta

    tree_valid = 0
    forest_valid = 0

    for idx in indices:
        x = Xtest[idx]

        # para cada classe-alvo
        for target_class in range(n_classes):

            # === árvore ===
            t0 = time.time()
            cost_dt, chg_dt, path_dt = solve_tree_min_changes(dt, x, target_class, feature_names)
            t1 = time.time()

            if cost_dt is not None:
                tree_times.append(t1 - t0)
                tree_costs.append(cost_dt)
                tree_sizes.append(len(path_dt))
                tree_valid += 1

            # === floresta ===
            t0 = time.time()
            cost_rf, chg_rf, chosen_paths = solve_forest_min_changes(rf, x, target_class, feature_names)
            t1 = time.time()

            if cost_rf is not None:
                forest_times.append(t1 - t0)
                forest_costs.append(cost_rf)
                # tamanho da explicação = soma dos tamanhos dos caminhos das árvores escolhidas
                total_path_size = sum(len(path) for plist in chosen_paths.values() for path in plist)
                forest_sizes.append(total_path_size)
                forest_valid += 1

    def safe_mean(lst):
        return mean(lst) if lst else None

    stats = {
        "tree_avg_time": safe_mean(tree_times),
        "forest_avg_time": safe_mean(forest_times),
        "tree_avg_cost": safe_mean(tree_costs),
        "forest_avg_cost": safe_mean(forest_costs),
        "tree_avg_size": safe_mean(tree_sizes),
        "forest_avg_size": safe_mean(forest_sizes),
        "tree_valid_count": tree_valid,
        "forest_valid_count": forest_valid,
        "n_samples_used": len(indices),
    }
    return stats

# ----------------------------- LOAD DATASETS REAIS -----------------------------

def load_bupa() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dataset BUPA (Liver Disorders)"""
    data = fetch_ucirepo(id=8)
    X = data.data.features.apply(pd.to_numeric, errors='coerce').to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    y = data.data.targets.to_numpy().reshape(-1)
    y = (y == y.max()).astype(int)
    fnames = np.array(list(data.data.features.columns))
    cnames = np.array(["no-disorder", "disorder"])
    return X, y, fnames, cnames

def load_breast_tumor() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Breast Cancer Coimbra"""
    data = fetch_ucirepo(id=451)
    X = data.data.features.to_numpy()
    y = data.data.targets.to_numpy().reshape(-1)
    fnames = np.array(list(data.data.features.columns))
    cnames = np.array(["benign", "malignant"])
    return X, y, fnames, cnames

# ===================== Seleção do dataset =====================

DATASET = "breast"  # "iris" | "bupa" | "breast" 

if DATASET == "iris":
    iris = load_iris()
    X, y = iris.data, iris.target
    fnames, cnames = np.array(iris.feature_names), np.array(iris.target_names)
elif DATASET == "bupa":
    X, y, fnames, cnames = load_bupa()
elif DATASET == "breast":
    X, y, fnames, cnames = load_breast_tumor()
else:
    raise ValueError("Dataset inválido!")

print(f"Dataset carregado: {DATASET}")
print("X shape:", X.shape, "| y shape:", y.shape)
print("Features:", len(fnames), "| Classes:", cnames)

# ===================== Treino =====================

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
dt = DecisionTreeClassifier(random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
dt.fit(Xtr, ytr); rf.fit(Xtr, ytr)

# escolha da instância
idx = 3  # mude se quiser
x = Xte[idx]

cur_dt = int(dt.predict([x])[0])
cur_rf = int(rf.predict([x])[0])
print("=== Instance ===")
print(x)
print("Predições atuais: DT =", cnames[cur_dt], "| RF =", cnames[cur_rf])

target_name = None  

def run_for_target(tgt_idx: int):
    print("\n============================")
    print("TARGET:", cnames[tgt_idx])
    print("============================")

    cost_dt, chg_dt, path_dt = solve_tree_min_changes(dt, x, tgt_idx, fnames)
    if cost_dt is None:
        print("[ÁRVORE] Sem folha-alvo ou UNSAT (caminho inválido).")
    else:
        print(f"[ÁRVORE] #mín. mudanças: {cost_dt}")
        for s in chg_dt: print(" -", s)
        if path_dt:
            print("Caminho usado (feat, thr, dir):")
            for (f,t,d) in path_dt:
                print(f"  ({fnames[f]}, {t:.3f}, {'right' if d=='R' else 'left'})")

    cost_rf, chg_rf, chosen = solve_forest_min_changes(rf, x, tgt_idx, fnames)
    if cost_rf is None:
        print("[FLORESTA] UNSAT (maioria impossível sob S).")
    else:
        print(f"[FLORESTA] #mín. mudanças: {cost_rf}")
        for s in chg_rf: print(" -", s)
        # opcional: quantas árvores votaram alvo
        print("Árvores que votaram alvo:", len(chosen))

if target_name is None:
    for k in range(len(cnames)):
        run_for_target(k)
else:
    tgt_idx = int(np.where(cnames == target_name)[0][0])
    run_for_target(tgt_idx)

    # ----------------------------- Gráfico Comparativo (Árvore vs Floresta) -----------------------------
def plot_comparison_bar(stats):
    """
    Gera um gráfico comparando:
      - tempo médio
      - custo médio
      - tamanho médio do caminho
    entre árvore de decisão e floresta.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["avg_time", "avg_cost", "avg_size"]
    tree_values = [
        stats["tree_avg_time"],
        stats["tree_avg_cost"],
        stats["tree_avg_size"],
    ]
    forest_values = [
        stats["forest_avg_time"],
        stats["forest_avg_cost"],
        stats["forest_avg_size"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, tree_values, width, label='Árvore', edgecolor='black')
    ax.bar(x + width/2, forest_values, width, label='Floresta', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(["Tempo Médio", "Custo Médio", "Tamanho Médio"])
    ax.set_ylabel("Média")
    ax.set_title("Comparação: Árvore vs Floresta")
    ax.legend()

    plt.tight_layout()
    plt.savefig("comparacao_tree_forest.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Gráfico comparativo salvo como: comparacao_tree_forest.png")
# ----------------------------- Avaliação agregada (métricas) -----------------------------

# Ajuste n_samples para o número de instâncias que quer testar (max: len(Xte))
n_samples = 30
print("\n==== AVALIAÇÃO AGREGADA ====")
stats = evaluate_explanations(
    dt=dt,
    rf=rf,
    Xtest=Xte,
    n_classes=len(np.unique(y)),        
    feature_names=fnames,
    n_samples=n_samples,
    random_seed=1,
)
print("\n==== Estatísticas Agregadas ====")

for k, v in stats.items():
    print(f"{k}: {v}")
    # --- gerar gráfico comparativo ---
plot_comparison_bar(stats)

# ----------------------------- Gráficos das Árvores -----------------------------
import matplotlib.pyplot as plt
from sklearn import tree

print("\n==== GERANDO GRÁFICOS DAS ÁRVORES ====")

# --- 1) Árvore única (dt) ---
plt.figure(figsize=(14, 10))
tree.plot_tree(
    dt,
    feature_names=fnames,
    class_names=cnames,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree (dt)")
plt.tight_layout()
plt.savefig("tree_dt.png", dpi=200, bbox_inches="tight")
plt.show()
print("Árvore dt salva em: tree_dt.png")

# --- 2) Primeira árvore da floresta (rf.estimators_[0]) ---
est0 = rf.estimators_[0]
plt.figure(figsize=(14, 10))
tree.plot_tree(
    est0,
    feature_names=fnames,
    class_names=cnames,
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("Random Forest - Estimator 0")
plt.tight_layout()
plt.savefig("tree_rf_0.png", dpi=200, bbox_inches="tight")
plt.show()
print("Árvore rf[0] salva em: tree_rf_0.png")