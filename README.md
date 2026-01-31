# 🧠 Contrastive Explanations via Partial MaxSAT
Este projeto implementa **explicações contrastivas** para modelos de aprendizado de máquina baseados em árvores, conforme descrito no artigo:

> **On Contrastive Explanations for Tree-Based Classifiers**  
> Pierre Audemard, Gilles Audemard, Luís Moniz Pereira, João Marques-Silva, *ECAI 2023*

A técnica usa **Partial MaxSAT** para encontrar o **conjunto mínimo de mudanças** nas características de uma instância que fariam o modelo mudar sua previsão para uma classe-alvo diferente.


## 📚 Contexto
Dado um exemplo `x` e um modelo (Árvore de Decisão ou Floresta Aleatória), o método responde à pergunta:

> “O que precisa mudar em `x` para que o modelo mude sua decisão para a classe `C`?”

O código reproduz o método apresentado na **Seção 5 do artigo** — *Computing Minimum-Size Contrastive Explanations* — usando o solucionador **RC2** do pacote [`python-sat`](https://pysathq.github.io/).

## ⚙️ Requisitos
Antes de executar o projeto, instale as dependências:

```bash
pip install -r requirements.txt
```
Ou instale manualmente no seu terminal:
```
pip install python-sat scikit-learn numpy
```

## ▶️ Como Executar (VS Code ou Terminal)
1. Clone o repositório
2. Execute o script principal:
    ```
    python contrastive_explanations.py
    ```
3. O programa irá:
- Treinar uma árvore de decisão e uma floresta aleatória no dataset Iris;
- Escolher uma instância de teste;
- Calcular as mudanças mínimas necessárias para alterar a classe prevista.