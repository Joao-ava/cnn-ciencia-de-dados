## CNN Ciência de dados

Classificação de frutas e vegetais saudáveis x podres com CNN.
Dataset: [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No VS Code, selecione o interpretador/kernel `.venv`.

## Antes de começar: download do dataset (~5 GB)

O dataset **não fica no repositório**. Ele é baixado do Kaggle pelo `kagglehub` na primeira execução
e reaproveitado nas próximas (fica em `~/.cache/kagglehub`).

| | Tamanho |
|---|---|
| Download / dataset extraído | **~5 GB** (29.277 imagens, 28 classes) |
| Cache das imagens em 224×224 (`data/cache/`, criado no 1º treino) | ~4,5 GB |
| **Espaço livre recomendado** | **~10 GB** |

Baixe uma vez num terminal e deixe terminar (pode levar bastante tempo, dependendo da internet):

```bash
source .venv/bin/activate
python -c "from src.data_loader import download_dataset; download_dataset()"
```

Quando aparecer `Path to dataset files: ...`, terminou.

> **Internet lenta ou pouco espaço?** Use o **Kaggle Notebooks** (o dataset já está lá, sem download, e tem GPU grátis)
> ou o **Colab** (download bem mais rápido). Se já tiver o dataset numa pasta, use
> `load_datasets(data_dir="caminho/da/pasta")`.

## Como usar os dados

```python
from src.data_loader import load_datasets
from src import config

train_ds, val_ds, test_ds, class_names = load_datasets()
num_classes = len(class_names)
```

- O dataset é baixado automaticamente pelo `kagglehub` na primeira vez.
- **Rótulos** são inteiros → use `loss="sparse_categorical_crossentropy"`.
- **Imagens** saem em `float32` com valores em `[0, 255]`, tamanho `config.IMG_SIZE` (224×224).
  Normalize **dentro do modelo**: `layers.Rescaling(1/255)` ou o `preprocess_input` do backbone
  (ex.: `tf.keras.applications.efficientnet.preprocess_input`).

### Opções do `load_datasets`

| Parâmetro | Padrão | Para que serve |
|---|---|---|
| `cache` | `"disk"` | Guarda as imagens já redimensionadas em `data/cache/`. A 1ª época é lenta (grava o cache); as próximas e os próximos trials ficam bem mais rápidos. `"memory"` usa RAM, `None` desliga. |
| `img_size` | `(224, 224)` | Tamanho das imagens. Cada tamanho gera um cache próprio. |
| `batch_size` | `32` | Tamanho do batch. |

### ⚠️ Classes desbalanceadas

A maior classe tem **14,6x** mais imagens de treino que as menores:

| Grupo | Imagens de treino por classe |
|---|---|
| Grape, Guava, Jujube, Pomegranate (Healthy e Rotten) | 140 |
| Carrot, Potato, Bellpepper, Cucumber, Tomato | ~405–433 |
| Strawberry, Mango, Banana, Orange, Apple | ~1.118–2.047 |

Sem tratamento, o modelo tende a ignorar as classes pequenas. Avalie com **F1 macro** e
métricas por classe (`classification_report`), não só acurácia.

### Splits fixos

A divisão treino/validação/teste (70/15/15, estratificada, seed 42) fica salva em
`data/splits.csv`, que está no git. Assim **todo mundo treina e avalia com exatamente as mesmas imagens**.
Não regenere esse arquivo sem combinar com o grupo (`load_datasets(regenerate_splits=True)`).

As constantes compartilhadas (seed, tamanho da imagem, caminhos) ficam em `src/config.py`.

## Pré-processamento (`src/preprocessing.py`)

> O notebook [`notebooks/01_preprocessamento.ipynb`](notebooks/01_preprocessamento.ipynb) mostra todas as
> etapas com gráficos e exemplos (distribuição das classes, duplicatas, normalização, augmentation e pesos).

Há duas formas de aplicar augmentation e normalização. **Escolha uma, nunca as duas juntas**:

**A) No `tf.data.Dataset` (recomendada).** O pré-processamento roda na CPU, em paralelo com o treino.
A augmentation vai só no treino; validação e teste recebem só a normalização.

```python
from src.preprocessing import load_preprocessed_datasets, compute_class_weights

train_ds, val_ds, test_ds, class_names = load_preprocessed_datasets(
    normalization="0_1",                    # ver tabela abaixo
    augmentation={"rotation": 0.1},         # opcional: ajusta a augmentation
)
model.fit(train_ds, validation_data=val_ds, class_weight=compute_class_weights(class_names))
```

**B) Como camadas no início do modelo.** A augmentation é desligada sozinha no `evaluate`/`predict`,
e o modelo salvo já recebe imagens 0–255.

```python
import keras
from src import config
from src.data_loader import load_datasets
from src.preprocessing import build_preprocessing, compute_class_weights

train_ds, val_ds, test_ds, class_names = load_datasets()
class_weight = compute_class_weights(class_names)   # compensa o desbalanceamento

inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
x = build_preprocessing(normalization="0_1")(inputs)
# ... camadas da CNN ...
model.fit(train_ds, validation_data=val_ds, class_weight=class_weight)
```

| `normalization` | Quando usar |
|---|---|
| `"0_1"` (padrão) | CNN do zero (pixels / 255) |
| `"-1_1"` | MobileNetV2/V3, Xception, Inception |
| `"imagenet"` | ResNet/VGG com média e desvio do ImageNet |
| `None` | EfficientNet/ConvNeXt do `keras.applications` (já normalizam por dentro) |

- Augmentation padrão: flip, rotação ±36°, zoom, translação, contraste e brilho ±10%. Ajuste ou desligue
  com argumentos, ex.: `build_preprocessing(rotation=0.05, zoom=0)`. Não há mudança de cor
  porque a cor é o que separa saudável de podre.
- `compute_class_weights` lê só o `splits.csv`. Os pesos vão de 0,36 (classes grandes) a 5,23 (pequenas).
- `python -m src.preprocessing` mostra os pesos e salva `augmentation_preview.png` para conferir a augmentation.

## Duplicatas entre splits (`src/duplicates.py`)

O dataset tem muitas imagens repetidas ou quase iguais (o mesmo objeto fotografado em sequência).
Se uma cópia fica no treino e outra no teste, o modelo "já viu" a resposta e as métricas ficam infladas.

```bash
python -m src.duplicates          # só relatório, não muda nada
python -m src.duplicates --fix    # corrige data/splits.csv
```

- **Exatas**: mesmo MD5 (arquivo idêntico).
- **Quase iguais**: hash perceptual (dHash) com até 4 bits diferentes **e** diferença de cor ≤ 6 numa
  miniatura 16×16. Só o hash não basta: ele junta frutas diferentes fotografadas no mesmo fundo.
  Os cortes foram escolhidos olhando pares deste dataset (`--threshold` e `--max-color-diff` para ajustar).
- **Correção**: cada grupo de duplicatas fica inteiro num só split (prioridade treino > val > teste).
  O resto da divisão não muda.
- **Saídas**: `data/duplicates.csv` (grupos), `data/duplicates_entre_classes.csv` (mesma imagem com
  rótulos diferentes, só aviso), `duplicates_preview.png` (exemplos para conferir) e
  `duplicates_entre_classes.png` (os pares com rótulos diferentes lado a lado).
- **Prévia de uma classe**: `python -m src.duplicates --preview-class Mango__Healthy --preview-groups 10`
  gera `duplicates_preview_Mango__Healthy.png`.
- **Limitação**: frutas pequenas num fundo grande e liso (algumas mangas) às vezes são agrupadas mesmo
  sendo frutas diferentes. O efeito é só mover algumas imagens a mais para o treino; nunca deixa vazamento.
- Leva ~15 s (16 núcleos); os hashes ficam em `data/cache/image_hashes.npz`.

Resultado neste dataset: 3.358 grupos (7.642 imagens), 1.648 deles espalhados em mais de um split;
o `--fix` move 1.868 imagens (val/teste → treino). Há também 10 pares de cenouras com a mesma imagem
em `Carrot__Healthy` e `Carrot__Rotten` (erro de rótulo do dataset; não alterados).

### Problemas comuns

- Se um treino for interrompido no meio da 1ª época, o cache pode ficar incompleto
  ou dar erro de *lockfile*: apague a pasta `data/cache/` e rode de novo.
- Sem GPU o treino é lento; o Colab/Kaggle Notebooks têm GPU grátis.
- O cache ocupa ~4,4 GB. Em máquina com muitos núcleos o ganho é pequeno (uma passada pelo treino
  já leva ~6 s sem cache num PC de 16 núcleos); se faltar espaço, use `cache=None`.

## Construção do modelo e busca de hiperparâmetros (`src/model.py`, `src/tuning.py`)

A arquitetura é uma CNN do zero: blocos de `Conv2D` (+ `BatchNormalization` opcional) terminando
em `MaxPooling2D`, seguidos por `GlobalAveragePooling2D` e camadas densas até a saída (`softmax`,
28 classes). Espera imagens já normalizadas (`load_preprocessed_datasets`, normalização `"0_1"`).

```python
from src.model import build_model

model = build_model(num_classes=28)   # valores default
model.summary()
```

### Hiperparâmetros buscados

`src/model.py` define o espaço de busca (`HYPERPARAMETER_SPACE`) e `make_build_model` gera a função
que o **Keras Tuner** espera (`build_model(hp)`):

| Hiperparâmetro | Espaço | O que controla |
|---|---|---|
| `conv_blocks` | 2 a 5 | nº de blocos convolucionais |
| `filters_inicial` | 16 / 32 / 64 | filtros do 1º bloco |
| `filters_growth` | 1 ou 2 | se os filtros dobram a cada bloco |
| `kernel_size` | 3 ou 5 | tamanho do kernel das convoluções |
| `conv_por_bloco` | 1 ou 2 | nº de `Conv2D` em sequência antes do pooling |
| `batchnorm` | ligado/desligado | `BatchNormalization` após cada `Conv2D` |
| `dropout_conv` | 0,0 a 0,3 | dropout após o pooling de cada bloco |
| `dense_units` | 128 / 256 / 512 | neurônios da camada densa final |
| `dropout_dense` | 0,2 a 0,5 | dropout antes da saída |
| `learning_rate` | 1e-4 a 1e-2 (log) | taxa de aprendizado |
| `optimizer` | adam / rmsprop | otimizador |

### Rodando a busca

```python
from src.tuning import run_search

tuner, best_hp, best_model = run_search(max_epochs=15, max_trials=20)
```

```bash
python -m src.tuning
```

- Usa o algoritmo **Hyperband** (`keras_tuner.Hyperband`): elimina trials ruins cedo, então custa
  bem menos do que treinar `max_trials` modelos completos.
- Objetivo: `val_accuracy`. Usa `class_weight` (de `compute_class_weights`) para compensar o
  desbalanceamento das classes também durante a busca. A avaliação final do melhor modelo deve
  olhar **F1 macro** e métricas por classe, não só a acurácia (ver seção de desbalanceamento acima).
- `EarlyStopping` (monitorando `val_loss`) encerra um trial antes de `max_epochs` se ele parar de melhorar.
- Resultados intermediários ficam em `tuning/` (fora do git); `run_search(overwrite=True)` descarta
  uma busca anterior com o mesmo nome e começa do zero.

## Testar o carregamento

```bash
python -m src.data_loader
```

Mostra quantas imagens há em cada split, as classes e o formato de um batch.
