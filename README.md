# Previsão de Temperaturas em Motor PMSM (RNN/TCN)

Projeto de **aprendizado profundo** para estimar temperaturas em um **PMSM** (Permanent Magnet Synchronous Motor) a partir de sinais elétricos e velocidade. Replica e estende o estudo da IEEE sobre RNN/TCN para termometria de motores, adicionando **engenharia de atributos**, **validação**, **callbacks de treino** e **explicabilidade** (PFI/SHAP).

- Artigo base: https://ieeexplore.ieee.org/abstract/document/9296842  
- Dataset: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

---

## Dados

- CSV com **6 entradas** (correntes/tensões em dq, velocidade etc.) e **4 saídas**: 3 temperaturas do **estator** (bobina/dente/núcleo) e 1 do **rotor**.  
- As séries são janeladas para modelagem temporal e **normalizadas** por média e desvio.

---

## Arquiteturas e busca de hiperparâmetros

Modelos para **estator** (3 saídas) e **rotor** (1 saída):

- **RNN (LSTM residual)** e **TCN (convolução temporal dilatada residual)**, com regularização (dropout, grad clip, ruído de gradiente) e janelas temporais configuráveis.  
- Quatro buscas reproduzindo o paper (tabela abaixo):  
  - **Ξ**: RNN para estator  
  - **Φ**: RNN para rotor  
  - **Ψ**: TCN para estator  
  - **Ω**: TCN para rotor

| Symbol | Hyperparameter                  | Interval            | Ξ (RNN–Stator) | Φ (RNN–Rotor) | Ψ (TCN–Stator) | Ω (TCN–Rotor) |
|:-----:|----------------------------------|---------------------|:--------------:|:-------------:|:--------------:|:-------------:|
| _nₗ_  | No. hidden layers                | [1, 7]              | 2              | 1             | 4              | 2             |
| _n_units_ | No. hidden units (ou canais) | [4, 256]            | 256            | 4             | 121            | 126           |
| α      | ℓ₂ regularization rate          | [1e−9, 1e−1]        | 1e−9           | 1e−1          | 1e−8           | 1e−9          |
| β      | Dropout rate                    | [0.2, 0.5]          | 0.37           | 0.50          | 0.29           | 0.35          |
| η      | Initial learn rate              | [1e−7, 1e−2]        | 1.4e−3         | 1e−2          | 1.4e−4         | 1e−4          |
| s₁     | Span 1 (EWM)                    | [500, 1500]         | 500            | 1500          | 620            | 500           |
| s₂     | Span 2 (EWM)                    | [2000, 3000]        | 2204           | 2000          | 2915           | 2161          |
| s₃     | Span 3 (EWM)                    | [4000, 6000]        | 6000           | 4000          | 4487           | 4000          |
| s₄     | Span 4 (EWM)                    | [7000, 9000]        | 9000           | 7000          | 8825           | 8895          |
|        | **RNN only**                    |                     |                |               |                |               |
| arch   | Architecture                     | {LSTM, GRU, Residual LSTM} | Residual LSTM | Residual LSTM | –              | –             |
| cₙ     | Grad. clip-norm                 | [0.25, 15]          | 9.4            | 0.25          | –              | –             |
| c_max  | Grad. clipping                  | [1e−2, 1]           | 0.076          | 0.01          | –              | –             |
| σ_GN   | Gradient noise                  | [1e−9, 1e−2]        | 1e−9           | 1e−2          | –              | –             |
| tbptt  | TBPTT length                    | [1, 128]            | 42             | 128           | –              | –             |
|        | **TCN only**                    |                     |                |               |                |               |
| arch   | Architecture                     | {Plain, Residual}   | –              | –             | Residual       | Residual      |
| l_kernel | Kernel size                   | [2, 7]              | –              | –             | 6              | 2             |
| w      | Window size                     | [8, 128]            | –              | –             | 32             | 33            |

---

### Variáveis
- **Derivadas físicas**  
  `i_s = √(i_d²+i_q²)`, `u_s = √(u_d²+u_q²)`, **potência aparente** `s_e ≈ u_s·i_s`, **potência ativa** `p_e ≈ u_d·i_d + u_q·i_q`, e interações com rotação (`i_s·w`, `s_e·w`) para capturar efeitos termo-mecânicos.
- **EWMs (médias e desvios exponenciais)**  
  Várias janelas (**spans** curto/médio/longo prazo) para modelar memória térmica: suavizam ruído, destacam tendência e volatilidade que influenciam picos de temperatura.

---

### Treinamento e avaliação
- **Callbacks**  
  *ReduceLROnPlateau* (ajuste dinâmico do LR), *EarlyStopping* (evita overfitting e restaura melhor época), *ModelCheckpoint* (salva pesos ótimos), *CSVLogger* (log fiel por época).
- **Métricas**  
  **MSE** (sensível a grandes erros), **MAE** (interpretação direta em °C), **RMSE** (mesma unidade da saída, penaliza picos), **R² por saída** (bobina/dente/núcleo/rotor) e **R² global** (agregado do estator).

---

### Explicabilidade
- **Permutation Feature Importance (PFI)**  
  Permuta cada feature e mede a piora do erro; fornece **importância global** normalizada pelo erro base → ranking objetivo para seleção/redução de variáveis.
- **SHAP (GradientExplainer)**  
  Atribui contribuição de cada feature **por amostra** (direção e magnitude); gera visão **local** e resumo **global** (|SHAP| médio) com tabelas e gráficos.
- **Complementaridade**  
  **PFI** foca impacto no desempenho; **SHAP** explica o *porquê* de cada predição. Usados juntos para diagnóstico e *feature pruning*.

---

### Saídas
- **Pesos de modelos** (`.h5`) para inferência/reuso.
- **Métricas e curvas** (CSV + gráficos de treino/validação; R², MAE, RMSE por saída e global).
- **Relatórios de importância** (rankings PFI/SHAP, tabelas e plots) para decisão e redução de dimensionalidade.
- **Organização em `out/`**: modelos, logs, métricas, plots e explicações, facilitando comparação entre arquiteturas e saídas.
