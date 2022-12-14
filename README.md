# Дневник по научной работе
Спецсеминар А. Г. Дьяконова, 4 курс, кафедра ММП, ВМК МГУ

## Предобработка табличных данных для глубокого машинного обучения

**03.11.22**

Данная тема была выбрана и согласована с научным руководителем в качестве темы дипломной работы. Первоначальная задача - ознакомиться с темой в целом и существующими подходами и проблемами. Нужно сформировать список статей для общего и подробного изучения. Основные темы - существующие архитектуры, способы препроцессинга, сравнение с алгоритмами классического машинного обучения применительно к табличным данным. Также нужно будет спланировать соответствующие эксперименты.

**06.11.22**

По этой же теме было решено проходить преддипломную практику, в том числе в рамках стажировки.

Собран набор статей для изучения:

- Общие обзорные статьи:
  - [Deep Neural Networks and Tabular Data: A Survey](https://arxiv.org/pdf/2110.01889.pdf)
  - [Tabular Data: Deep Learning Is Not All You Need](https://arxiv.org/pdf/2106.03253.pdf)
  - [Deep Learning for Tabular Data: An Exploratory Study](https://core.ac.uk/download/pdf/196259727.pdf)
  - [A Short Chronology Of Deep Learning For Tabular Data (есть большой сборник статей по теме в хронологическом порядке)](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html)
  - [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959v2.pdf)
  - [Why do tree-based models still outperform deep learning on tabular data](https://arxiv.org/pdf/2207.08815.pdf)
  
- Существующие архитектуры:
  - [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/pdf/1908.07442.pdf)
  - [Neural Oblivious Decision Ensembles For Deep Learning On Tabular Data](https://arxiv.org/pdf/1909.06312.pdf)
  - [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)
  - [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)
  - [TabNN: A Universal Neural Network Solution For Tabular Data](https://openreview.net/pdf?id=r1eJssCqY7)
  - [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf)
  - [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/pdf/2106.01342.pdf)
  - [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf)
  - [SuperTML: Two-Dimensional Word Embedding for the Precognition on Structured Tabular Data](https://arxiv.org/pdf/1903.06246.pdf)
  - [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)
  - [Converting tabular data into images for deep learning with convolutional neural networks](https://pubmed.ncbi.nlm.nih.gov/34059739/)
  - [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)

- Стратегии препроцессинга
  - [An Embedding Learning Framework for Numerical Features in CTR Prediction](https://arxiv.org/pdf/2012.08986.pdf)
  - [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/pdf/2106.02795.pdf)
  - [Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains](https://arxiv.org/pdf/2006.10739.pdf)
  - [Methods for Numeracy-Preserving Word Embeddings](https://aclanthology.org/2020.emnlp-main.384.pdf)
  - [Time-Dependent Representation For Neural Event Sequence Prediction](https://arxiv.org/pdf/1708.00065.pdf)
  - [Survey on categorical data for neural networks](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00305-w)
  - [Supervised and unsupervised discretization of continuous features](https://ai.stanford.edu/~ronnyk/disc.pdf)
  
**09.11.22**

Нашлось несколько интересных статей по теме:
  - [TabPFN: A Transformer That Solves Small Tabular Classification Problems In A Second](https://arxiv.org/pdf/2207.01848.pdf)
  - [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723.pdf)
  - [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/pdf/2209.07663.pdf)
  
**19.11.22**

Нашлась полезная статья по теме: [On Embeddings for Numerical Features in Tabular Deep Learning](https://openreview.net/pdf?id=pfI7u0eJAIr). Стоит также ознакомиться с библиотекой [rtdl](https://github.com/Yura52/rtdl) и реализациями препроцессингов в ней.

**29.11.22**

Разобрал и законспектировал основные моменты отобранных статей, из исследований литературы можно сделать следующие выводы:
- Нейронные сети пока что плохо справляются с табличными данными, часто уступают градиентному бустингу и ансамблям решающих деревьев. Как причины этого исследователи выделяют:
  - Разнородность признаков в таблицах, их гетерогенность, в отличие от изображений, текста и т. п., где есть пространственная и семантическая близость, наличие непрерывных и категориальных признаков одновременно, разный масштаб и тип признаков
  - Наличие признаков с различной важностью, влиянием на целевую переменную, частое наличие неинформативных и шумовых признаков в табличных данных
  - Наличие пропусков и выбросов в табличных данных, частый дисбаланс классов в случае задач классификации
  - Относительно малое количество признаков в некоторых случаях
  - Отсутствие порядка в расположении столбцов в общем случае, отсутствие упорядоченности между значениями категорий отдельных признаков
  - Существенное влияние масштаба числовых признаков, необходимость нормализации для нейросетей
- Применение существенно новых архитектур и специализированных архитектур не так сильно влияет на качество предсказаний нейросетей на таблицах, как преобразование входных данных. Наиболее частое и эффективное решение - преобразование табличных данных и использование соответствующей известной архитектуры глубокого обучения
- Наиболее распространенными и эффективными архитектурами в задачах на табличных данных являются многослойный персептрон и модель трансформер, а также архитектуры по типу ResNet, DenseNet

Необходимо определить набор видов препроцессинга и базовых архитектур для проведения вычислительных экспериментов, а также тип задач и набор датасетов.

**02.12.22**

В качестве основных архитектур нейросетей для начальных экспериментов было решено взять многослойный персептрон и трансформер как наиболее распространенные и используемые в данных задачах. Из видов препроцессинга были выбраны следующие методы:

- Кодирование с помощью введения периодичности (по мотивам данной [статьи](https://arxiv.org/pdf/2006.10739.pdf)), рассмотреть [RFF](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf), [ORF](https://arxiv.org/pdf/1610.09072.pdf), [Positional encoding](https://arxiv.org/pdf/1706.03762.pdf), а также подобное преобразование можно сделать обучаемым, для числовых и категориальных переменных. В дальнейшем можно рассмотреть вместо функций синуса или косинуса, например, RBF ядро или иные функции.
- Использование линейных слоев с различными активациями
- [AutoDis](https://arxiv.org/pdf/2012.08986v2.pdf)
- SoftEmbeddings из библиотеки [transformers4rec](https://github.com/NVIDIA-Merlin/Transformers4Rec)
- Построение обучаемых эмбеддингов для числовых и категориальных признаков по отдельности, или же одновременно, проецируя их по сути в одно пространство

**05.12.22**

 Для начала будут рассматриваться датасеты из открытых источников, преимущественно из задач бинарной классификации и из финансовой сферы (кредитный скоринг, прогнозирование оттока клиентов, детекция мошеннических операйций и т. д.)(в контексте стажировки). В данных в основном наблюдается заметный дисбаланс классов, а также присутствуют категорильные переменные с высокой кардинальностью, встречаются пропуски и шумовые признаки
 
 **09.12.22**
 
 Проведены несколько первичных экспериментов, необходимо подкорректировать пайплайн обучения и валидации, уточнить набор метрик для оценивания качества модели. В дальнейшем нужно сделать расширенный подбор гиперпараметров.
