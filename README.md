# Curso: Knowledge Distillation Para Imagens Médicas em PyTorch

Curso prático de introdução à **Knowledge Distillation (KD)** em PyTorch (https://mtcazzolato.github.io/eabda-kd/). Um modelo grande já treinado (*teacher*) ensina um modelo pequeno (*student*), com o objetivo de manter boa qualidade com muito menos parâmetros, menor tamanho e menor tempo de inferência.

## O que o curso mostra

- Como carregar e balancear um dataset de imagens médicas.
- Como treinar um **teacher** (ResNet18 adaptada) e um **student baseline** (CNN pequena, sem KD).
- Como funciona a loss de KD: temperatura **T** e peso **α**.
- Como treinar students com KD (T=4 e T=10) e compará-los com o baseline.
- Como avaliar no teste: métricas, matriz de confusão, fidelidade ao teacher e custo (parâmetros, tamanho, latência).

## Conteúdo

O curso está em um único notebook, dividido em cinco blocos:

| Bloco | Tema | Pergunta que responde |
|---|---|---|
| 00 | Setup e dados | Com o que estamos trabalhando? |
| 01 | Teacher | Qual é o modelo "professor"? |
| 02 | Student baseline | Até onde o modelo pequeno vai sozinho? |
| 03 | Knowledge Distillation | O professor ajuda? Com quais T e α? |
| 04 | Avaliação final | Vale a pena (qualidade vs. custo)? |

## Dataset

**PneumoniaMNIST** (MedMNIST), versão **128×128**, radiografias em escala de cinza com duas classes: Normal e Pneumonia. O download é automático e fica em `./data`.

Os splits de treino, validação e teste são balanceados por *undersampling* (50% de cada classe, com seed fixa).

## Modelos

| Modelo | Arquitetura | Papel |
|---|---|---|
| Teacher | ResNet18 (pesos ImageNet), com `conv1` de 1 canal e `fc` de 2 saídas | Modelo grande, congelado durante a KD |
| Student | `SmallCNN`: 3 convoluções + `AdaptiveAvgPool2d` + camada linear | Modelo pequeno (23.426 parâmetros) |

## Loss de KD

```
L = α · CE(rótulo real, student) + (1 − α) · T² · KL(softmax(teacher/T) ‖ softmax(student/T))
```

- **T**: temperatura, que suaviza as probabilidades do teacher.
- **α**: peso do **rótulo real**. Com α = 1 a loss vira só a cross-entropy (equivale ao baseline).
- O teacher só gera logits em cada batch e nunca é atualizado (KD *offline*).

## Configuração usada no notebook

| Item | Valor |
|---|---|
| Batch size | 32 |
| Épocas (teacher e students) | 20 |
| Otimizador | Adam |
| Learning rate | 1e-4 (teacher), 1e-3 (students) |
| KD | T=4 e T=10, com α=0.75 |
| Varredura opcional | T ∈ {1, 2, 4, 10} × α ∈ {0.25, 0.5, 0.75}, 5 épocas cada |
| Seed | 42 |

## Estrutura do repositório

```
.
├── KD_in_medical_images.ipynb   # notebook do curso
├── utils.py                   # funções compartilhadas
├── data/                      # criado automaticamente (dataset)
└── weights/                   # criado ao salvar os modelos
    ├── teacher.pth
    ├── student_baseline.pth
    ├── student_kd_4.pth
    └── student_kd_10.pth
```

## Requisitos

- Python 3
- GPU recomendada (na CPU, o treino do teacher é muito lento)

Instalação (a mesma célula do notebook):

```bash
pip install torch torchvision medmnist matplotlib scikit-learn seaborn
```

## Como executar

1. Coloque `KD_course_revisado.ipynb` e `utils.py` na mesma pasta.
2. Abra o notebook e execute as células na ordem.
3. No bloco 01, a variável `TRAIN_TEACHER` controla o teacher:
   - `True`: treina o teacher do zero e salva em `weights/teacher.pth`.
   - `False`: apenas carrega `weights/teacher.pth` (o arquivo precisa existir).
4. O bloco 04 carrega os quatro modelos de `weights/`, então os blocos 01 a 03 precisam ter sido executados antes (ou os pesos precisam estar na pasta).

No Google Colab, descomente as células que montam o Drive e definem a pasta de trabalho (blocos 00).

## O arquivo `utils.py`

| Grupo | Funções |
|---|---|
| Dados e reprodutibilidade | `get_device`, `set_seed`, `balance_dataset`, `get_labels`, `get_dataloaders` |
| Modelos | `Teacher`, `SmallCNN`, `load_teacher`, `load_student`, `count_parameters` |
| Treino e validação | `train_teacher_and_student_baseline`, `train_knowledge_distillation`, `validate`, `distillation_loss` |
| Inferência e métricas | `get_logits`, `get_predictions`, `metrics_table`, `model_size_mb`, `measure_latency`, `compare_models` |
| Gráficos | `plot_temperature`, `plot_prob_hist`, `plot_history`, `plot_confusion_matrices` |
| Persistência | `save_weights`, `load_weights` |

## Observações

- Os resultados valem para o **teste balanceado**, e não para a prevalência real de pneumonia.
- O curso usa **uma seed**. Diferenças pequenas entre baseline e KD podem ser ruído; para conclusões mais firmes, repita com várias seeds.
- A `train_loss` impressa durante o treino com KD inclui o termo do teacher, então não é comparável com a do baseline. Compare pela validação (`val_loss`, `val_acc`) ou pela tabela final.
- Hiperparâmetros (T e α) devem ser escolhidos na validação; o teste é usado uma única vez, no bloco 04.

## Referências

- Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network*, 2015.
- Yang, J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, 2023.
- Knowledge Distillation Tutorial by [PyTorch](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).
