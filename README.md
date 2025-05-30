# Awsome_Affective_Model_Compression

A curated list of 76 papers focusing on model compression techniques in affective computing, including tasks such as emotion recognition, expression recognition, stress detection, depression recognition, and personality detection. Categorized by task, modality, and compression method.

## Emotion recognition

<details>
  <summary>👁️ Visual Modality </summary>


- **Light-FER: A Lightweight Facial Emotion Recognition System on Edge Devices**  
  *Method:* Pruning and Quantization  
  *Dataset:* FER2013  
  [Link](https://pubmed.ncbi.nlm.nih.gov/36502225/)

- **Factorized Higher-Order CNNs with an Application to Spatio-Temporal Emotion Estimation**  
  *Method:* CP Tensor Decomposition  
  *Dataset:* SEWA  
  [Link](https://arxiv.org/abs/1906.06196)

- **A Lightweight Method for Face Expression Recognition Based on Improved MobileNetV3**  
  *Method:* Lightweight (Improved MobileNetV3)  
  *Datasets:* FER2013, RAF-DB  
  [Link](https://www.researchgate.net/publication/370097017)

- **Three Convolutional Neural Network Models for Facial Expression Recognition in the Wild**  
  *Method:* Lightweight (Depthwise Separable CNN)  
  *Datasets:* FER2013, RAF-DB  
  [Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231219306137)

- **Comparison of Different Depth of Convolutional Neural Network Models for Facial Expression Recognition**  
  *Method:* Lightweight (Shallow CNN)  
  *Dataset:* FER2013  
  [Link](https://drpress.org/ojs/index.php/HSET/article/view/6746)

- **Facial Emotion Recognition Through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets**  
  *Method:* Lightweight (Custom Low-Parameter CNN)  
  *Datasets:* FER2013, RAF-DB, AffectNet, CK+  
  [Link](https://www.researchgate.net/publication/379193339)

- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**  
  *Method:* Lightweight (Inverted Bottleneck Convolution)  
  *Note:* Foundational for later work using MobileNetV2 structure  
  [Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

- **Pseudo-Inverted Bottleneck Convolution for DARTS Search Space**  
  *Method:* Lightweight (Inverted Bottleneck Convolution)  
  *Datasets:* RAF-DB, FER2013H  
  [Link](https://arxiv.org/abs/2301.01286)

</details>

<details>
  <summary>🔊 Audio Modality  </summary>

- **Discriminative pruning of deep neural networks for speech emotion recognition**  
  *Method:* Pruning  
  *Dataset:* IEMOCAP  
  [Link](https://doi.org/10.1109/ICASSP40776.2020.9052994)

- **Knowledge distillation using HuBERT embeddings for small-footprint emotion recognition**  
  *Method:* Knowledge Distillation  
  *Dataset:* IEMOCAP  
  [Link](https://arxiv.org/abs/2203.07082)

- **LIGHT-SERNET: A Lightweight Deep Learning Architecture for Speech Emotion Recognition**  
  *Method:* Lightweight  
  *Dataset:* RAVDESS  
  [Link](https://www.sciencedirect.com/science/article/pii/S266682702100008X)

- **Parallel attention-based CNN model for speech emotion recognition**  
  *Method:* Lightweight  
  *Dataset:* IEMOCAP  
  [Link](https://www.sciencedirect.com/science/article/pii/S187705092030150X)

- **Low-footprint convolutional model for real-time speech emotion recognition**  
  *Method:* Lightweight  
  *Dataset:* EmoDB  
  [Link](https://ieeexplore.ieee.org/document/9206043)

- **Structured pruning for lightweight SER on mobile devices**  
  *Method:* Pruning  
  *Dataset:* RAVDESS  
  [Link](https://arxiv.org/abs/2106.05684)

- **Multiscale CNN with quantization-aware training for efficient SER**  
  *Method:* Quantization  
  *Dataset:* IEMOCAP  
  [Link](https://www.mdpi.com/2079-9292/9/4/585)

- **Compact transformers for end-to-end speech emotion recognition**  
  *Method:* Lightweight  
  *Dataset:* IEMOCAP  
  [Link](https://arxiv.org/abs/2109.01169)

- **Speech emotion recognition with tiny speech CNNs**  
  *Method:* Lightweight  
  *Dataset:* EmoDB  
  [Link](https://arxiv.org/abs/2112.07100)

- **Efficient CNN architecture using spectrogram compression for SER**  
  *Method:* Compression (custom)  
  *Dataset:* RAVDESS  
  [Link](https://www.sciencedirect.com/science/article/pii/S0167639321000883)
  
</details>

<details>
  <summary> 🧠 Physiological Modality  </summary>

- **Disentangling EEG Representation Using Neuroscience Priors for Emotion Recognition**  
  *Method:* Pruning  
  *Dataset:* DEAP  
  [Link](https://ieeexplore.ieee.org/document/10143835)

- **A Teacher–Student Framework for Emotion Recognition Using EEG Signals**  
  *Method:* Knowledge Distillation  
  *Dataset:* DEAP  
  [Link](https://ieeexplore.ieee.org/document/10144557)

- **A Quantized CNN for Emotion Recognition from EEG Signals**  
  *Method:* Quantization  
  *Dataset:* DEAP  
  [Link](https://www.mdpi.com/2076-3417/10/20/7136)

- **Temporal Convolutional 3D Network for Emotion Recognition with EEG**  
  *Method:* Lightweight  
  *Dataset:* SEED  
  [Link](https://ieeexplore.ieee.org/document/10144531)

- **SHAP-guided Pruning of GCNs for EEG-based Emotion Recognition**  
  *Method:* Pruning  
  *Dataset:* AMIGOS  
  [Link](https://www.sciencedirect.com/science/article/pii/S1746809423002881)

- **Quantization-aware Training for LSTM in Emotion Recognition using PPG**  
  *Method:* Quantization  
  *Dataset:* WESAD  
  [Link](https://www.mdpi.com/1424-8220/20/5/1380)

- **Tiny 1D CNN for Real-time ECG-based Emotion Classification**  
  *Method:* Lightweight  
  *Dataset:* DRIVE  
  [Link](https://www.sciencedirect.com/science/article/pii/S1746809421001297)

- **Distilled EEGNet for Real-time Affective State Detection**  
  *Method:* Knowledge Distillation  
  *Dataset:* DEAP  
  [Link](https://www.sciencedirect.com/science/article/pii/S1746809421000631)

- **Hybrid CNN-LSTM with Pruned Structure for Emotion Detection**  
  *Method:* Pruning  
  *Dataset:* DEAP  
  [Link](https://www.sciencedirect.com/science/article/pii/S1877050920302085)

- **Low-complexity Temporal CNN for Wearable EEG Emotion Recognition**  
  *Method:* Lightweight  
  *Dataset:* DREAMER  
  [Link](https://www.sciencedirect.com/science/article/pii/S1746809420301934)

- **EEG Emotion Recognition via Quantized Mobile CNN**  
  *Method:* Quantization  
  *Dataset:* MAHNOB-HCI  
  [Link](https://ieeexplore.ieee.org/document/10042782)

</details>

<details>
  <summary> 🔄 Multimodal Modality  </summary>


- **Efficient Audio-Visual Emotion Recognition Using Structured Pruning**  
  *Method:* Pruning  
  *Dataset:* MuSe-Humor  
  [Link](https://arxiv.org/abs/2211.01067)

- **Cross-Modal Knowledge Distillation for Multimodal Emotion Recognition**  
  *Method:* Knowledge Distillation  
  *Dataset:* RECOLA  
  [Link](https://arxiv.org/abs/2110.07210)

- **Tensor Fusion Network Compression via Tensor Decomposition**  
  *Method:* Decomposition  
  *Dataset:* CMU-MOSEI  
  [Link](https://arxiv.org/abs/1905.10395)

- **Mobile Multimodal Emotion Recognition Using Lightweight Fusion Network**  
  *Method:* Lightweight  
  *Dataset:* IEMOCAP  
  [Link](https://ieeexplore.ieee.org/document/9426126)

- **Distilling Emotion Representations from Audio-Visual Models to Unimodal Models**  
  *Method:* Knowledge Distillation  
  *Dataset:* CMU-MOSEI  
  [Link](https://arxiv.org/abs/2206.12475)

- **A Low-Resource Multimodal Transformer for Emotion Recognition**  
  *Method:* Lightweight  
  *Dataset:* CMU-MOSEI  
  [Link](https://arxiv.org/abs/2110.05104)

- **Multimodal Emotion Recognition with Compact Multiscale Attention Fusion**  
  *Method:* Lightweight  
  *Dataset:* IEMOCAP  
  [Link](https://ieeexplore.ieee.org/document/10018925)

- **Multimodal Emotion Recognition with Feature-level Compression and KD**  
  *Method:* Knowledge Distillation + Feature Compression  
  *Dataset:* CMU-MOSEI  
  [Link](https://arxiv.org/abs/2303.00791)

</details>

## Expression Recognition
<details>
  <summary>👁️ Visual Modality </summary>


- **Magnitude-based Pruning for Facial Expression Recognition on Mobile Devices**  
  *Method:* Pruning  
  *Dataset:* RAF-DB  
  [Link](https://ieeexplore.ieee.org/document/10027330)

- **Facial Expression Recognition Based on Pruning Optimization Technology**  
  *Method:* Pruning  
  *Dataset:* CK+, JAFFE  
  [Link](https://www.researchgate.net/publication/369876962_Facial_Expression_Recognition_Based_on_Pruning_Optimization_Technology)

- **Adaptive CNN Pruning Based on Structural Similarity of Filters (APSSF)**  
  *Method:* Pruning  
  *Dataset:* FER2013  
  [Link](https://link.springer.com/article/10.1007/s44196-024-00518-4)

- **Teacher-Bounded Loss for FER Knowledge Distillation**  
  *Method:* Knowledge Distillation  
  *Dataset:* AffectNet  
  [Link](https://www.researchgate.net/figure/Proposed-knowledge-distillation-method-based-on-teacher-bounded-loss_fig4_371061560)

- **Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition**  
  *Method:* Knowledge Distillation  
  *Dataset:* FairFace  
  [Link](https://arxiv.org/html/2408.16563v1)

- **Distilled VGG16 for Real-time FER**  
  *Method:* Knowledge Distillation  
  *Dataset:* FER2013  
  [Link](https://arxiv.org/abs/1910.01752)

- **FER using MobileNet with Distilled Attention**  
  *Method:* Knowledge Distillation  
  *Dataset:* FER2013  
  [Link](https://arxiv.org/abs/2202.01527)

- **Distilled ResNet with Emotion Prior Masks for FER**  
  *Method:* Knowledge Distillation  
  *Dataset:* FERPlus  
  [Link](https://arxiv.org/abs/2107.07500)

- **IRBN: Iterative Residual Binarized Network for Efficient FER**  
  *Method:* Quantization  
  *Dataset:* RaFD  
  [Link](https://ieeexplore.ieee.org/document/9179629)

- **BitNet: Binary CNN for Embedded FER**  
  *Method:* Quantization  
  *Dataset:* AffectNet  
  [Link](https://www.sciencedirect.com/science/article/pii/S0262885620301850)

- **Weight Quantization of CNNs for Expression Recognition**  
  *Method:* Quantization  
  *Dataset:* JAFFE  
  [Link](https://www.mdpi.com/2227-7390/9/8/851)

- **Ternary Quantization-aware FER CNN**  
  *Method:* Quantization  
  *Dataset:* RAF-DB  
  [Link](https://ieeexplore.ieee.org/document/9565769)

- **BinaryNet for Facial Expression Recognition on Edge Devices**  
  *Method:* Quantization  
  *Dataset:* FER2013  
  [Link](https://www.sciencedirect.com/science/article/pii/S2666827022000056)

- **Quantized Residual Networks for FER in Unconstrained Environments**  
  *Method:* Quantization  
  *Dataset:* FERPlus  
  [Link](https://ieeexplore.ieee.org/document/9567197)

- **Low-rank Approximation for Expression Recognition Networks**  
  *Method:* Decomposition  
  *Dataset:* FER2013  
  [Link](https://www.sciencedirect.com/science/article/pii/S1877050921003132)

- **EfficientNet-lite for Mobile Facial Expression Recognition**  
  *Method:* Lightweight  
  *Dataset:* AffectNet  
  [Link](https://arxiv.org/abs/2012.01426)

- **Attention-based Shallow CNN for FER with Fusion Layer**  
  *Method:* Lightweight  
  *Dataset:* RAF-DB  
  [Link](https://www.sciencedirect.com/science/article/pii/S0925231220305761)

- **MobileFaceNet-based FER with Tiny Parameter Count**  
  *Method:* Lightweight  
  *Dataset:* FER2013  
  [Link](https://www.mdpi.com/2076-3417/10/24/9052)

- **Compact Residual Attention Network for FER**  
  *Method:* Lightweight  
  *Dataset:* RAF-DB  
  [Link](https://www.sciencedirect.com/science/article/pii/S0957417421001813)

- **FER with Lightweight Transformer and Attention**  
  *Method:* Lightweight  
  *Dataset:* AffectNet  
  [Link](https://arxiv.org/abs/2204.10242)

- **AutoFER: Neural Architecture Search for Facial Expression Recognition**  
  *Method:* Lightweight  
  *Dataset:* CK+  
  [Link](https://www.sciencedirect.com/science/article/pii/S016786552030223X)

- **Tiny Dual-branch CNN for FER in Video**  
  *Method:* Lightweight  
  *Dataset:* AFEW  
  [Link](https://ieeexplore.ieee.org/document/9369711)

- **Efficient Deep Feature Compression for FER on the Edge**  
  *Method:* Compression (custom)  
  *Dataset:* AffectNet  
  [Link](https://arxiv.org/abs/2301.06610)

- **Multi-scale Lightweight CNN for FER in Real-World Scenarios**  
  *Method:* Lightweight  
  *Dataset:* AffectNet  
  [Link](https://arxiv.org/abs/2108.06579)

- **Compact CNN using Group Convolutions for FER**  
  *Method:* Lightweight  
  *Dataset:* RAF-DB  
  [Link](https://www.mdpi.com/2079-9292/10/6/709)

- **Lightweight Attention-Guided FER with Mobile Efficiency**  
  *Method:* Lightweight  
  *Dataset:* AffectNet  
  [Link](https://www.sciencedirect.com/science/article/pii/S0957417422005707)

- **Pruned ResNet-18 for FER with Knowledge Guidance**  
  *Method:* Pruning + Knowledge Distillation  
  *Dataset:* RAF-DB  
  [Link](https://arxiv.org/abs/2109.07796)

</details>





