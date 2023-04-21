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


**16.12.22**

Проведены эксперименты для многослойного персептрона и трансформера, а ткаже для градиентного бустинга в качестве бейзлайна, на данных с kaggle из задач бинарной классификации. Данные в целом с большим дисбалансом классов, порядок признаков произвольный, среди них встречаются шумовые, относительно много категориальных переменных. Можно сделать основные промежуточные выводы:
- В полученных экспериментально результатах сложно выделить явные и значительные различия или закономерности в рассмотренных методах препроцессинга, результаты в целом схожи, в том числе и на разных данных. Однако в целом это согласуется с результатами, полученными в различных исследованиях по этой теме. Для получения более содержательных результатов требуется провести более объемные эксперименты и рассмотреть большее число методов, рассмотреть другие задачи, данные, архитектуры и т. д. Однако использование предобработки признаков в большинстве случаев оказывается полезным и повышает качество классификации, хотя в отдельных случаях любые преобразования приводят к ухудшению качества в сравнении с исходными признаками.
- Также сложно выделить наилучшую архитектуру нейронной сети по результатам проведенных экспериментов, итоговое значение ROC-AUC для моделей достаточно близко во многих случаях. Однако все же наилучший результат для конкретного датасета достигался для многослойного персептрона с различными методами предобработки признаков. Схожие наблюдения присутствуют в различных исследованиях, можно сказать, что вероятно в рассмотренных задачах преимущества архитектуры играют небольшую роль. Причиной этому могли стать также и особенности данных (дисбаланс, неинформативные признаки, присутствие пропусков, разреженность и т. д.).
- Использование матрицы обучаемых эмбеддингов для отдельных признаков чаще всего оказывается наиболее эффективным и позволяет достичь наилучшего качества для конкретного датасета. Применение линейных слоев с активациями также положительно сказывается на результате, в особенности для модели трансформера. Применение же периодических преобразований или методов с "проецированием" на набор эмбеддингов демонстрируют результаты хуже. Можно предположить, что в рассмотренных данных и задачах взаимосвязи между признаками не так сильны и важны, и поэтому использование отдельных эмбеддингов оказалось более успешным.
- Алгоритм градиентного бустинга по-прежнему превосходит модели глубокого обучения на табличных данных, в некоторых случаях значительно. Однако использование различных методов предобработки признаков позволяет уменьшить разницу в итоговом качестве, и для одного из наборов данных удалось достигнуть результата градиентного бустинга. Таким образом, разработка и применение различных способов препроцессинга табличных данных может позволить нейронным сетям достичь более высокого уровня качества решения задач с табличными данными.
- Использование одного типа построения эмбеддингов для числовых и категориальных признаков не привело к значительному улучшению по сравнению с исходными данными (со стандартизацией и one-hot кодированием). Возможно комбинирование принципиально разных способов преобразования разнотипных признаков могло бы привести к более значительным улучшениям, однако это во многом зависит от самих данных и присутствия и силы взаимосвязей между признаками, и в целом кодирование разнотипных признаков идейно различными методами интуитивно не позволит в полной мере рассматривать трансформированные признаки в совокупности для извлечения различных закономерностей в данных.
- Оптимизация гиперпараметров в проведенных экспериментах мало повлияла на итоговое качество, в большинстве случаев наилучшие конфигурации состояли из одних и тех же значений. Однако использовалось относительно небольшое количество итераций, возможное более тщательный процесс подбора привел бы к более существенным результатам.

В дальнейшем нужно будет:
- Провести более объемные эксперименты, рассмотреть другие типы данных и задач, провести тщательную оптимизацию гиперпараметров, провести большее число запусков, рассмотреть большее число метрик.
- Рассмотреть больше базовых архитектур, провести эксперименты для ResNet и DenseNet, возможно и для специализированных под таблицы архитектур.
- Поэкспериментировать со способами препроцессинга, рассмотреть разные значения их гиперпараметров, сэмплировать веса из других распределений, рассмотреть разное число слоев и разные функции активации при построении эмбеддингов линейными слоями с активациями, попробовать методы с различным биннингом числовых признаков, рассмотреть комбинации разных способов для числовых и категориальных признаков по отдельности.
- Возможно рассмотреть принципиально другие подходы, например, [преобразование таблиц в черно-белое изображение и применение сверточных сетей](https://pubmed.ncbi.nlm.nih.gov/34059739/), и т. п.
- Подкорректировать в целом процесс обучения и валидации, исправить существующие недостатки.
- Возможно попробовать вносить различные изменения в саму архитектуру.

**28.12.22**

Защищена преддипломная практика по выбранной теме.

**20.01.22**

Собраны и рассмотрены статьи по теме:
- [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)
- [Deep Learning in Unconventional Domains](https://thesis.library.caltech.edu/13669/1/Cvitkovic_Milan_2020_Thesis.pdf)
- [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)
- [Deep Neural Networks for YouTube Recommendations](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45530.pdf?ysclid=lencv6ec7v845705100)

Добавлена в рассмотрение аритектура ResNet, проведены эксперименты для классификации. Нужно будет собрать датасеты для регрессии.

**06.02.23**

Нашлась статья [ARM-Net: Adaptive Relation Modeling Network for Structured Data](https://arxiv.org/pdf/2107.01830v1.pdf), описанная идея с моделированием мультипликативных зависимостей выглядит полезно, стоит лучше изучить и добавить метод в эксперименты. 

**16.02.23**

Проведено больше экспериментов для классификации с большим количеством значений гиперпараметров, в некоторых случаях качество возросло, на некоторых датасетах получилось обойти градиентный бустинг. Нужно будет рассмотреть следующие моменты:

- Трансформер плохо работает с методами AutoDis и SoftEmbeddings, варьирование температуры, числа эмбеддингов в таблице проекций, размер эмбеддинга и т. д. не сильно повлияли на ситуацию, причины довольно низкого качества пока что не очень понятны.
- MLP и ResNet не особенно чувствительны к гиперпараметрам обучения (learning rate, weight decay, batch size и т. д.) почти на всех данных, трансформер же наоборот очень чувствителен к данным параметрам, настройка конфигов самой архитектуры не сильно меняла качество. Также для всех моделей наилучший размер эмбеддинга примерно 10-16, число слоев - 2-3.
- Не особо заметно влияние способа инициализации весов в методах, использование позиционного кодирования, весов классов и разных scheduler-ов также не улучшило показатели

**09.03.23**

Проведены эксперименты для регрессии на данных с OpenML. В некоторых случаях удалось заметно превзойти XGBoost, оданако трансформер в больинстве случаев обучается плохо, качество довольно низкое. Рассмотрены разные преобразования таргета в регресии (масштабирование, логарифмирование и т. д.), заметных улучшений не произошло.

**20.03.23**

Добавлен способ построения эмбеддингов с помощью отдельного трансформера, как в TabTransformer, но уже для всех типов признаков. Работает хуже ARM и остальных методов в большей части случаев, учится долго, хотя для MLP и ResNet иногда получается хорошее качество.

**22.03.23**

Расширены эксперименты, рассмотрены 3 основные схемы преобразований:

- Раздельное построение эмбеддингов для числовых и категориальных признаков, наиболее логичный и интуитивный способ.
- Рассмотрение всех признаков как числовых (категории - OHE), затем построение эмбеддингов. Очевидно это искажает суть категориальных переменных, но возможно это как-то позволяет использовать взаимосвязи между признаками.
- Рассмотрение всех признаков как категориальных (числовые - квантильный биннинг), затем построение эмбеддингов, мотивация использования аналогична предыдущей схеме. 

Стоит отметить, что все рассмотренные в работе методы формирования эмбеддингов применимы к исходным числовым признакам, к необработанным категориальным применимы только способ с lookup-таблицей эмбеддингов, ARM и трансформер, т. к. в них есть стадия сопоставления категории вещественнозначного вектора. Остальным методам на вход уже требуется вещественное значение. 

**10.04.23**

Проведено больше экспериментов, из наблюдений:

- Использование методов с ARM и трансформером для построения эмбеддингов особого прироста качества не дает, но гораздо дольше работает и учится. По сути в них используется еще одна полноценная модель, видимо это сильно усложняет всю архитектуру в целом.
- Раздельное преобразование чесел и категорий работает лучше всего, что ожидаемо.
- В случае регрессии AutoDis и SoftEmbeddings для трансформеров работают гораздо лучше и стабильнее, чем для классификации. 
- На одном из датасетов (HS) трансформер принципиально плохо обучается, качество гораздо хуже остальных моделей, возможно стоит больше поварьировать параметры.
- Использование периодических функций по прежнему работает лучше всего, видимо они действительно хорошо моделируют неоднородные признки и сложные зависимости в таблицах, также там эмбеддинги по сути в 2 раза больше задаваемого размера.
- Иногда периодические функции с необучаемыми коэффициентами работают лучше, чем с обучаемыми, нужно будет рассмотреть отдельно.
- MLP с трансформером для построения эмбеддингов иногда дает неплохие результаты, вероятно композиция различных по устройству моделей может быть полезна для таблиц.

**17.04.23**

Рассмотрено больше гиперпараметров для трансформера для регересии (пока что вручную), в некоторых случаях качество стало получше. Исправил ошибку в своей реализации ARM.
