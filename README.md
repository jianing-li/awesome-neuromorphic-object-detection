# Neuromorphic Object Detection Resources
This repository presents  a systematic survey and benchmarks of exsiting object detection methods using neuromorphic cameras.

## Table of Contents
- [Recommened Surveys](#recommened_surveys)
- [Datasets](#detection_datasets)
- [Event Representations](#event_representations)
- [Temporal Modeling](#temporal_modeling)
- [Asynchronous Processing](#asynchronous_processing)
- [Energy-efficient Computing](#energy-efficient_computing)
- [Network Compression](#network_compression)
- [Hardware Deployment](#hardware_deployment)
- [Application Scenarios](#application_scenarios)
- [New Directions](#new_directions)
- [Modality - All Publication List](#modality)

___
<br>

<a name="recommened_surveys"></a>
# Recommened Surveys
- <a name="Gallego20tpami"></a>Gallego G, Delbrück T, Orchard G, et al., **_[Event-based Vision: A Survey](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)_**, IEEE Trans. Pattern Anal. Machine Intell. (TPAMI), 2022.
- <a name="li2024brain"></a>Li G, Deng L, Tang H, et al., **_[Brain-inspired computing: A systematic survey and future trends](https://ieeexplore.ieee.org/abstract/document/10636118/)_**, Proceedings of the IEEE, 2024.
- <a name="li21recent"></a>李家宁, 田永鸿, **_[神经形态视觉传感器的研究进展及应用综述](http://cjc.ict.ac.cn/online/onlinepaper/ljn-20216781514.pdf)_**, 计算机学报, 2021.



<a name="detection_datasets"></a>
# Object Detection Datasets
### Events
- [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101), *Converting static image datasets to spiking neuromorphic datasets using saccades*, Front. Neurosci., 2015.
- [Pedestrian Detection Dataset](https://github.com/CrystalMiaoshu/PAFBenchmark), *Neuromorphic vision datasets for pedestrian detection, action recognition, and fall detection*, Front. Neurorobotics, 2019.
- [Gen1 Detection](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/), *A large scale event-based detection dataset for automotive*, arXiv, 2020.
- [1Mpx Detection](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/), *Learning to detect objects with a 1 Megapixel event camera*, Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
- [eTraM](https://eventbasedvision.github.io/eTraM), *eTraM: Event-based traffic monitoring dataset*, arXiv, 2024.
- [NU-AIR](https://bit.ly/nuair-data), *NU-AIR: A neuromorphic urban aerial dataset for detection and localization of pedestrians and vehicles*, Int. J. Comput. Vis. (IJCV), 2025.
- [EvDET200K](https://github.com/Event-AHU/OpenEvDET), *Object detection using event camera: A moe heat conduction based detector and a new benchmark dataset*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.


### Events and Frames
- [PKU-DDD17-CAR](https://www.pkuml.org/resources/pku-ddd17-car.html), *Event-based vision enhanced: A joint detection framework in autonomous driving*, Proc. IEEE Int. Conf. Multimedia Expo. (ICME), 2019.
- [DAD](https://github.com/Liumy213/Dateset-of-APS-and-DVS), *An attention fusion network for event-based vehicle object detection*, Proc. IEEE Int. Conf. Imag. Process. (ICIP), 2021.
- [PKU-Vidar-DVS](https://www.pkuml.org/resources/pku-vidar-dvs.html), *Retinomorphic object detection in asynchronous visual streams*, Proc. AAAI Conf. on Artificial Intell. (AAAI), 2022.
- [DSEC-Fusing](https://github.com/abhishek1411/event-rgb-fusion), *Fusing event-based and rgb camera for robust object detection in adverse conditions*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2022.
- [DSEC-MOD](https://github.com/ZZY-Zhou/RENet), *RGB-event fusion for moving object detection in autonomous driving*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2023.
- [DSEC-Det](https://github.com/YN-Yang/SFNet), *Enhancing traffic object detection in variable illumination with RGB-event fusion*, IEEE Trans. Intell. Transp. Syst. (TITS), 2024.
- [Aqua-Eye](https://github.com/lunaWU628/Aqua-Eye-Dataset), *Transcodnet: Underwater transparently camouflaged object detection via RGB and event frames collaboration*, IEEE Robot. Autom. Lett. (RAL), 2023.
- [PEDRo](https://github.com/SSIGPRO/PEDRo-Event-Based-Dataset.git), *PEDRo: An event-based dataset for person detection in robotics*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops. (CVPRW), 2023.
- [TUMTraf](https://innovation-mobility.com/en/project-providentia/a9-dataset/), *TUMTraf event: Calibration and fusion resulting in a dataset for roadside event-based and rgb cameras*, IEEE Trans. Intell. Veh. (TIV), 2024.
- [PKU-DAVIS-SOD](https://www.pkuml.org/research/pku-davis-sod-dataset.html), *SODFormer: Streaming object detection with transformer using events and frames*, IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2023.
- [DSEC-Detection](https://github.com/uzh-rpg/dsec-det), *Low-latency automotive vision with event cameras*, Nature, 2024.


<a name="event_representations"></a>
# Event Representations (Selected Works)

### Event Images
- **Binary images**, [An attention fusion network for event-based vehicle object detection](https://ieeexplore.ieee.org/abstract/document/9506561/), Proc. IEEE Int. Conf. Image Process. (ICIP), 2021.
- **Event images**, [SODFormer: Streaming object detection with transformer using events and frames](https://ieeexplore.ieee.org/abstract/document/10195232/), IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2023.
- **Grayscale image**, [Spike-event object detection for neuromorphic vision](https://ieeexplore.ieee.org/abstract/document/10195232/), IEEE Access, 2023.


### Handcrafted Features
- **SAE**, [Multi-cue event information fusion for pedestrian detection with neuromorphic vision sensors](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00010/full), Front. Neurosci., 2019.
- **Voxel grid**, [Learning to detect objects with a 1 megapixel event camera](https://proceedings.neurips.cc/paper/2020/hash/c213877427b46fa96cff6c39e837ccee-Abstract.html), Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
- **Voxel cube**, [Object detection with spiking neural networks on automotive event data](https://proceedings.neurips.cc/paper/2020/hash/c213877427b46fa96cff6c39e837ccee-Abstract.html), Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2022.
- **Temporal extension embedding**, [SpikingViT: A multi-scale spiking vision transformer model for event-based object detection](https://ieeexplore.ieee.org/abstract/document/10586833), IEEE Trans. Cogn. Dev. Syst. (TCDS), 2024.
- **Hyper histogram**, [Better and faster: Adaptive event conversion for event-based object detection](https://ojs.aaai.org/index.php/AAAI/article/view/25298), Proc. AAAI Conf. on Artificial Intell. (AAAI), 2023.
- **ERGO**, [From chaos comes order: Ordering event representations for object recognition and detection](https://openaccess.thecvf.com/content/ICCV2023/html/Zubic_From_Chaos_Comes_Order_Ordering_Event_Representations_for_Object_Recognition_ICCV_2023_paper.html), Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2023


### ANN-based Learned Representations
- **Event embedding**, [Asynchronous spatio-temporal memory network for continuous event-based object detection](https://ieeexplore.ieee.org/abstract/document/9749022/), IEEE Trans. Image Process. (TIP), 2022.
- **Group token embedding**, [GET: Group event transformer for event-based vision](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_GET_Group_Event_Transformer_for_Event-Based_Vision_ICCV_2023_paper.html), Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2023.
- **Reconstructed image**, [Event-to-video conversion for overhead object detection](https://arxiv.org/pdf/2402.06805), arXiv, 2024.
- **SET**, [Spatio-temporal aggregation transformer for object detection with neuromorphic vision sensors](https://ieeexplore.ieee.org/abstract/document/10516298), IEEE Sensors J., 2024.


### SNN-based Learned Representations
- **Spike image**, [Event-based vision enhanced: A joint detection framework in autonomous driving](https://ieeexplore.ieee.org/abstract/document/8784777/), Proc. IEEE Int. Conf. Multimedia Expo. (ICME), 2019.
- **Leaky surface**, [Asynchronous convolutional networks for object detection in neuromorphic cameras](https://openaccess.thecvf.com/content_CVPRW_2019/html/EventVision/Cannici_Asynchronous_Convolutional_Networks_for_Object_Detection_in_Neuromorphic_Cameras_CVPRW_2019_paper.html), Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops. (CVPRW), 2019.
- **ARSNN**, [EAS-SNN: End-to-end adaptive sampling and representation for event-based detection with recurrent spiking neural networks](https://link.springer.com/chapter/10.1007/978-3-031-73027-6_18), Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.


<a name="temporal_modeling"></a>
# Temporal Modeling

### Temporal Aggregation Operations
- <a name="li2022retinomorphic"></a> Li J, Wang X, Zhu L, et al., *[Retinomorphic object detection in asynchronous visual streams](https://ojs.aaai.org/index.php/AAAI/article/view/20021)*, Proc. AAAI Conf. on Artificial Intell. (AAAI), 2022.
- <a name="zhou2023rgb"></a>  Zhou Z, Wu Z, Boutteau R, et al., *[RGB-event fusion for moving object detection in autonomous driving](https://ieeexplore.ieee.org/abstract/document/10161563)*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2023.
- <a name="hamaguchi2023hierarchical"></a> Hamaguchi R, Furukawa Y, Onishi M, et al., *[Hierarchical neural memory network for low latency event processing](https://openaccess.thecvf.com/content/CVPR2023/html/Hamaguchi_Hierarchical_Neural_Memory_Network_for_Low_Latency_Event_Processing_CVPR_2023_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.
- <a name="han2024real"></a> Han Y, Suo J, Zhang B, et al., *[Real-time sketching of harshly lit driving environment perception by neuromorphic sensing](https://ieeexplore.ieee.org/abstract/document/10536615)*, IEEE Trans. Intell. Veh. (TIV), 2024.


### Recurrent-Convolutional Architectures
- <a name="Perot2020learning"></a> Perot E, De Tournemire P, Nitti D, et al, *[Learning to detect objects with a 1 megapixel event camera](https://proceedings.neurips.cc/paper/2020/hash/c213877427b46fa96cff6c39e837ccee-Abstract.html)*, Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
- <a name="li2022asynchronous"></a> Li J, Li J, Zhu L, et al., *[Asynchronous spatio-temporal memory network for continuous event-based object detection](https://ieeexplore.ieee.org/abstract/document/9749022/)*, IEEE Trans. Image Process. (TIP), 2021.
- <a name="wang2023dual"></a> Wang D, Jia X, Zhang Y, et al., *[Dual memory aggregation network for event-based object detection with learnable representation](https://ojs.aaai.org/index.php/AAAI/article/view/25346)*, Proc. AAAI Conf. on Artificial Intell. (AAAI), 2023.
- <a name="Andersen2022event"></a> Andersen K F, Pham H X, Ugurlu H I, et al., *[Event-based navigation for autonomous drone racing with sparse gated recurrent network](https://ieeexplore.ieee.org/abstract/document/9838538)*, Proc. Eur. Control Conf. (ECC), 2022.
- <a name="silva2024recurrent"></a> Silva D A, Smagulova K, Elsheikh A, et al., *[A recurrent yolov8-based framework for event-based object detection](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1477979/full)*, Front. Neurosci., 2024.
- <a name="zhu2024spatio"></a> Guo Z, Gao J, Ma G, et al., *[Spatio-temporal aggregation transformer for object detection with neuromorphic vision sensors](https://ieeexplore.ieee.org/abstract/document/10516298)*, IEEE Sensors J., 2024.
- <a name="jing2025esvt"></a> Jing S, Guo G, Xu X, et al., *[ESVT: Event-based streaming vision transformer for challenging object detection](https://ieeexplore.ieee.org/abstract/document/10835753/)*, IEEE Trans. Geosci. Remote Sens. (TGRS), 2025.


### Temporal Transformers
- <a name="li2023sodformer"></a> Li D, Tian Y, Li J, *[SODFormer: Streaming object detection with transformer using events and frames](https://ieeexplore.ieee.org/abstract/document/10195232/)*, IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2023.
- <a name="gehrig2023recurrent"></a> Gehrig M, Scaramuzza D, *[Recurrent vision transformers for object detection with event cameras](https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.


### Temporal Dynamics in SNNs (Selected Works)
- <a name="cordone2022object"></a> Cordone L, Miramond B, Thierion P, *[Object detection with spiking neural networks on automotive event data](https://ieeexplore.ieee.org/abstract/document/9892618/)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2022.
- <a name="su2023deep"></a> Su Q, Chou Y, Hu Y, et al., *[Deep directly-trained spiking neural networks for object detection](https://openaccess.thecvf.com/content/ICCV2023/html/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.html?ref=https://githubhelp.com)*, Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2023
- <a name="wang2024eas"></a> Wang Z, Wang Z, Li H, et al., *[EAS-SNN: End-to-end adaptive sampling and representation for event-based detection with recurrent spiking neural networks](https://link.springer.com/chapter/10.1007/978-3-031-73027-6_18)*, Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.
- <a name="luo2024integer"></a> Luo X, Yao M, Chou Y, et al., *[Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_15)*, Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.
- <a name="fan2024sfod"></a> Fan Y, Zhang W, Liu C, et al., *[SFOD: Spiking fusion object detector](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_SFOD_Spiking_Fusion_Object_Detector_CVPR_2024_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.


<a name="asynchronous_processing"></a>
# Asynchronous Processing

### Sparse Convolutions
- Messikommer N, Gehrig D, Loquercio A, et al., *[Event-based asynchronous sparse convolutional network](https://arxiv.org/pdf/2003.09148)*, Proc. Eur. Conf. Comput. Vis. (ECCV), 2020. [Code](https://github.com/uzh-rpg/rpg_asynet)
- Jack D, Maire F, Denman S, et al, *[Sparse convolutions on continuous domains for point cloud and event stream networks](https://openaccess.thecvf.com/content/ACCV2020/html/Jack_Sparse_Convolutions_on_Continuous_Domains_for_Point_Cloud_and_Event_ACCV_2020_paper.html)*, Proc. Asian Conf. Comput. Vis. (ACCV), 2020.
- Durvasula S, Guan Y, Vijaykumar N, *[Ev-Conv: Fast cnn inference on event camera inputs for high-speed robot perception](https://arxiv.org/pdf/2303.04670)*, IEEE Robot. Autom. Lett. (RAL), 2023. [Code](https://github.com/utcsz/evconv)


### Graph Neural Networks
- Schaefer S, Gehrig D, Scaramuzza D, *[AEGNN: Asynchronous event-based graph neural networks](https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022. [Code](https://uzh-rpg.github.io/aegnn/)
- Gehrig D, Scaramuzza D, *[Pushing the limits of asynchronous graph-based object detection with event cameras](https://arxiv.org/pdf/2211.12324)*, arXiv, 2022.
- Jeziorek K, Pinna A, Kryjak T. *[Memory-efficient graph convolutional networks for object classification and detection with event cameras](https://ieeexplore.ieee.org/abstract/document/10274464/)*, Proc. IEEE Signal Process.: Algorithms Archit. Arrangements Appl. (SPA), 2023.
- Sun D, Ji H. *[Event-based object detection using graph neural networks](https://ieeexplore.ieee.org/abstract/document/10166491)*, Proc. IEEE Data Driven Control Learn. Syst. Conf. (DDCL), 2023.
- Deng Y, Chen H, Xie B, et al. *[A dynamic graph cnn with cross-representation distillation for event-based recognition](https://arxiv.org/pdf/2302.04177)*. arXiv, 2023.
- Jeziorek K, Wzorek P, Blachut K, et al. *[Optimising graph representation for hardware implementation of graph convolutional networks for event-based vision](https://link.springer.com/chapter/10.1007/978-3-031-62874-0_9)*, Proc. Int. Workshop Des. Archit. Signal Image Process. (DASIP), 2024.
- Gehrig D, Scaramuzza D, *[Low-latency automotive vision with event cameras](https://www.nature.com/articles/s41586-024-07409-w)*, Nature, 2024. [Code](https://github.com/uzh-rpg/dagr)
- Wu S, Sheng H, Feng H, et al.,*[EGSST: Event-based graph spatiotemporal sensitive transformer for object detection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/da733d44e4be3902d952d6c1ffcb7db6-Abstract-Conference.html)*, Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.
- Li D, Li J, Liu X, et al.,*[Asynchronous collaborative graph representation for frames and events]()*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025. [Code](https://github.com/dianzl/ACGR)


### Event-driven SNNs
- Liu Q, Richter O, Nielsen C, et al, *[Live demonstration: Face recognition on an ultra-low power event-driven convolutional neural network asic](https://openaccess.thecvf.com/content_CVPRW_2019/html/EventVision/Liu_Live_Demonstration_Face_Recognition_on_an_Ultra-Low_Power_Event-Driven_Convolutional_CVPRW_2019_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops. (CVPRW), 2019.
- Ziegler A, Vetter K, Gossard T, et al., *[Spiking neural networks for fast-moving object detection on neuromorphic hardware devices using an event-based camera](https://arxiv.org/abs/2403.10677)*, arXiv, 2024. [Resource](https://cogsys-tuebingen.github.io/snn-edge-benchmark)


<a name="energy-efficient_computing"></a>
# Energy-efficient Computing

### ANN-to-SNN Conversion
- Kim S, Park S, Na B, et al., *[Spiking-YOLO: Spiking neural network for energy-efficient object detection](https://ojs.aaai.org/index.php/AAAI/article/view/6787)*, Proc. AAAI Conf. on Artificial Intell. (AAAI), 2020.
- Chakraborty B, She X, Mukhopadhyay S, *[A fully spiking hybrid neural network for energy-efficient object detection](https://ieeexplore.ieee.org/abstract/document/9591302/)*, IEEE Trans. Image Process. (TIP), 2021.
- Wang Y K, Wang S E, Wu P H, *[Spike-event object detection for neuromorphic vision](https://ieeexplore.ieee.org/abstract/document/10016699)*, IEEE Access, 2023.

### Directly-trained SNNs
- Cordone L, Miramond B, Thierion P, *[Object detection with spiking neural networks on automotive event data](https://proceedings.neurips.cc/paper/2020/hash/c213877427b46fa96cff6c39e837ccee-Abstract.html)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2022.
- Su Q, Chou Y, Hu Y, et al., *[Deep directly-trained spiking neural networks for object detection](https://openaccess.thecvf.com/content/ICCV2023/html/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.html?ref=https://githubhelp.com)*, Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2023. [Code](https://github.com/BICLab/EMS-YOLO)
- Yuan M, Zhang C, Wang Z, et al., *[Trainable spiking-yolo for low-latency and high-performance object detection](https://www.sciencedirect.com/science/article/pii/S0893608023007530)*, Neural Netw. (NN), 2023.
* Barchid S, Mennesson J, Eshraghian J, et al., *[Spiking neural networks for frame-based and event-based single object localization](https://www.sciencedirect.com/science/article/pii/S0925231223009281)*, Neurocomputing, 2023.
* Fan Y, Zhang W, Liu C, et al., [SFOD: Spiking fusion object detector](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_SFOD_Spiking_Fusion_Object_Detector_CVPR_2024_paper.html)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024. [Code](https://github.com/yimeng-fan/SFOD)
- Luo X, Yao M, Chou Y, et al., *[Integer-valued training and spike-driven inference spiking neural network for highperformance and energy-efficient object detection](https://arxiv.org/pdf/2407.20708)*, in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024. [Code](https://github.com/BICLab/SpikeYOLO)
- Wang Z, Wang Z, Li H, et al., *[EAS-SNN: End-to-end adaptive sampling and representation for event-based detection with recurrent spiking neural networks](https://link.springer.com/chapter/10.1007/978-3-031-73027-6_18)*, Proc. Eur. Conf. Comput. Vis. (ECCV), 2024. [Code](https://github.com/Windere/EAS-SNN)
- Yu L, Chen H, Wang Z, et al., *[SpikingViT: A multi-scale spiking vision transformer model for event-based object detection]()*, IEEE Trans. Cogn. Dev. Syst. (TCDS), 2024.
- Ziegler A, Vetter K, Gossard T, et al., *[Spiking neural networks for fast-moving object detection on neuromorphic hardware devices using an event-based camera](https://arxiv.org/abs/2403.10677)*, arXiv, 2024. [Resource](https://cogsys-tuebingen.github.io/snn-edge-benchmark)
- Caccavella C, Paredes-Vallés F, Cannici M, et al., *[Low-power event-based face detection with asynchronous neuromorphic hardware](https://ieeexplore.ieee.org/abstract/document/10650843/)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2024.
- Bodden L, Ha D B, Schwaiger F, et al., *[Spiking centernet: A distillation-boosted spiking neural network for object detection](https://arxiv.org/pdf/2402.01287)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2024.
- Wang Z, Wang Z, Lian S, et al., *[Adaptive gradient-based timesurface for event-based detection](https://ieeexplore.ieee.org/abstract/document/10888665/)*, IEEE Int. Conf. Acoust. Speech Signal Process. (ICCASP), 2025.
- Mao R, Shen A, Tang L, et al., *[Crest: An efficient conjointlytrained spike-driven framework for event-based object detection exploiting spatiotemporal dynamics](https://ojs.aaai.org/index.php/AAAI/article/view/32649)*, Proc. AAAI Conf. on Artificial Intell. (AAAI), 2025. [Code](https://github.com/shen-aoyu/CREST/)
- Fan Y, Liu C, Li M, et al., *[SpikSSD: Better extraction and fusion for object detection with spiking neuron networks](https://arxiv.org/abs/2501.15151)*, 2025. [Code](https://github.com/yimeng-fan/SpikSSD)


### Hybrid ANN-SNN Architectures
- Li J, Dong S, Yu Z, et al., *[Event-based vision enhanced: A joint detection framework in autonomous driving](https://ieeexplore.ieee.org/abstract/document/8784777/)*, Proc. IEEE Int. Conf. Multimedia Expo. (ICME), 2019.
- Kugele A, Pfeil T, Pfeiffer M, et al., *[Hybrid snn-ann: Energy-efficient classification and object detection for event-based vision](https://arxiv.org/pdf/2112.03423)*, Proc. DAGM German Conf. Pattern Recognit., 2021.
- Zhao R, Yang Z, Zheng H, et al., *[A framework for the general design and computation of hybrid neural networks](https://www.nature.com/articles/s41467-022-30964-7)*, Nat. Commun. (NC), 2022.
- Ahmed S H, Finkbeiner J, Neftci E, *[A Hybrid SNN-ANN network for event-based object detection with spatial and temporal attention](https://arxiv.org/abs/2403.10173)*. arXiv, 2024.
- Li D, Li J, Liu X, et al., *[HDI-Former: Hybrid dynamic interaction ann-snn transformer for object detection using frames and events](https://arxiv.org/pdf/2411.18658)*, arXiv, 2024.



<a name="network_compression"></a>
# Network Compression

### Quantization
- Przewlocka-Rus D, Kryjak T, Gorgon M, *[Poweryolo: Mixed precision model for hardware efficient object detection with event data](https://arxiv.org/pdf/2407.08272)*, arXiv, 2024.

### Distillation
- Deng Y, Chen H, Xie B, et al. *[A dynamic graph cnn with cross-representation distillation for event-based recognition](https://arxiv.org/pdf/2302.04177)*. arXiv, 2023.
- Bodden L, Ha D B, Schwaiger F, et al., *[Spiking centernet: A distillation-boosted spiking neural network for object detection](https://arxiv.org/pdf/2402.01287)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2024.
- Li L, Linger A, Millhaeusler M, et al. *[Object-centric cross-modal feature distillation for event-based object detection](https://arxiv.org/pdf/2311.05494)*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2024.



<a name="hardware_deployment"></a>
# Hardware Deployment

### FPGAs
- Qiu N, Li Z, Li Y, et al., *[Highly efficient SNNs for high-speed object detection](https://arxiv.org/pdf/2309.15883)*. arXiv, 2023.
- Kryjak T, *[Event-based vision on FPGAs: A survey](https://arxiv.org/pdf/2407.08356)*, arXiv, 2024.
- Courtois J, Novac P E, Lemaire E, et al., *[Embedded event based object detection with spiking neural network](https://ieeexplore.ieee.org/abstract/document/10649943/)*, Proc. Int. Joint Conf. Neural Netw. (IJCNN), 2024.
- Ren Y, Siegel B, Yin R, et al., *[Neuromorphic detection and cooling of microparticle arrays](https://arxiv.org/pdf/2408.00661)*, arXiv, 2024.
- Li Z, Lu W, Lu Y, et al., *[An energy-efficient object detection system in iot with dynamic neuromorphic vision sensors](https://ieeexplore.ieee.org/abstract/document/10558689/)*, Proc. IEEE Int. Symposium Circuits Syst. (ISCAS), 2024.


### Edge GPUs
- Wang Z, Cladera F, Bisulco A, et al., *[EV-Catcher: High-speed object catching using low-latency event-based neural networks](https://arxiv.org/pdf/2304.07200)*, IEEE Robot. Autom. Lett. (RAL), 2022
- Yuan M, Zhang C, Wang Z, et al., *[Trainable spiking-yolo for low-latency and high-performance object detection](https://www.sciencedirect.com/science/article/pii/S0893608023007530)*, Neural Netw. (NN), 2023.
- Sanaullah, Koravuna S, Rückert U, et al., *[A spike vision approach for multi-object detection and generating dataset using multicore architecture on edge device](https://link.springer.com/chapter/10.1007/978-3-031-62495-7_24)*, Proc. Int. Conf. Eng. Appl. Neural Netw. (EANN), 2024.
- Chen D, Zhou L, Guo C, *[A low-latency dynamic object detection algorithm fusing depth and events](https://www.proquest.com/docview/3181428058?fromopenview=true&pq-origsite=gscholar)*, Drones, 2025.


### Edge TPUs
- Crafton B, Paredes A, Gebhardt E, et al., *[Hardware-algorithm co-design enabling efficient event-based object detection](https://ieeexplore.ieee.org/abstract/document/9458497/)*, Proc. IEEE Int. Conf. Artificial Intell. Circuits Syst. (AICAS), 2021.
- Lu Y, Shi Y, Li Z, et al., *[A real-time event vision sensor based object detection and tracking system for edge applications](https://ieeexplore.ieee.org/abstract/document/10849082)*, Proc. IEEE Asian Solid-State Circuits Conf.(ASSCC), 2024


### Neuromorphic Chips
- Ziegler A, Vetter K, Gossard T, et al., *[Spiking neural networks for fast-moving object detection on neuromorphic hardware devices using an event-based camera](https://arxiv.org/abs/2403.10677)*, arXiv, 2024. [Resource](https://cogsys-tuebingen.github.io/snn-edge-benchmark)
- Ziegler A, Vetter K, Gossard T, et al., *[Detection of fast-moving objects with neuromorphic hardware](https://arxiv.org/abs/2403.10677)*, arXiv, 2024
- Silva D A, Shymyrbay A, Smagulova K, et al., *[End-to-end edge neuromorphic object detection system](https://ieeexplore.ieee.org/abstract/document/10595906/)*, Proc. IEEE Int. Conf. Artificial Intell. Circuits Syst. (ISCAS), 2024.


<a name="application_scenarios"></a>
# Application Scenarios (Selected Works)

### Space Awareness
- Afshar S, Nicholson A P, Van Schaik A, et al., *[Event-based object detection and tracking for space situational awareness](https://ieeexplore.ieee.org/abstract/document/9142352/)*. IEEE} Sensors J., 2020.
- Salvatore N, Fletcher J, *[Learned event-based visual perception for improved space object detection](https://openaccess.thecvf.com/content/WACV2022/html/Salvatore_Learned_Event-Based_Visual_Perception_for_Improved_Space_Object_Detection_WACV_2022_paper.html)*, Proc. Int. IEEE Winter Conf. Applications Comput. Vis. (WACV). 2022.
- Jawaid M, Elms E, Latif Y, et al., *[Towards bridging the space domain gap for satellite pose estimation using event sensing](https://ieeexplore.ieee.org/abstract/document/10160531/)*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2023.
- Zhou X, Bei C., *[End-to-end space object detection method based on event camera](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12917/129170K/End-to-end-space-object-detection-method-based-on-event/10.1117/12.3011053.short)*, Proc. Int. Conf. Precision Instruments Opt. Eng., 2023.


### Autonomous Driving
- Chen G, Cao H, Conradt J, et al., *[Event-based neuromorphic vision for autonomous driving: A paradigm shift for bio-inspired visual sensing and perception]()*. IEEE Sig. Proc. Mag. (SPM), 2020.
- Li D, Tian Y, Li J, [SODFormer: Streaming object detection with transformer using events and frames](https://ieeexplore.ieee.org/abstract/document/10195232/), IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2023. [Code](https://github.com/dianzl/SODFormer)
- Gehrig D, Scaramuzza D, *[Low-latency automotive vision with event cameras](https://www.nature.com/articles/s41586-024-07409-w)*, Nature, 2024. [Code](https://github.com/uzh-rpg/dagr)

### Agile Drones
- Mitrokhin A, Fermüller C, Parameshwara C, et al., *[Event-based moving object detection and tracking](https://arxiv.org/pdf/1803.04523)*, Proc. IEEE Conf. Intell. Robot. Syst. (IROS), 2018.
- Andersen K F, Pham H X, Ugurlu H I, et al., *[Event-based navigation for autonomous drone racing with sparse gated recurrent network](https://ieeexplore.ieee.org/abstract/document/9838538)*, Proc. Eur. Control Conf. (ECC), 2022.
- Iaboni C, Kelly T, Abichandani P., *[NU-AIR: A neuromorphic urban aerial dataset for detection and localization of pedestrians and vehicles](https://link.springer.com/article/10.1007/s11263-025-02418-2)*, Int. J. Comput. Vis. (IJCV), 2025. [Dataset](https://bit.ly/nuair-data)


### Humanoid Robotics
- Iacono M, Weber S, Glover A, et al., *[Towards event-driven object detection with off-the-shelf deep learning](https://ieeexplore.ieee.org/abstract/document/8594119/)*, Proc. IEEE Conf. Intell. Robot. Syst. (IROS), 2018.
- Monforte M, Arriandiaga A, Glover A, et al., *[Where and when: Event-based spatiotemporal trajectory prediction from the iCub’s point-of-view](https://ieeexplore.ieee.org/abstract/document/9197373)*, Proc. IEEE Conf. Robot. Autom. (ICRA), 2020.


### Underwater Robotics
- Luo C, Wu J, Sun S, et al., *[Transcodnet: Underwater transparently camouflaged object detection via RGB and event frames collaboration](https://ieeexplore.ieee.org/abstract/document/10373055/)*, IEEE Robot. Autom. Lett. (RAL), 2023.
- Dadson N K N, Barbalata C, *[Marine Event Vision: Harnessing Event Cameras For Robust Object Detection In Marine Scenarios](https://openaccess.thecvf.com/content/WACV2025W/MaCVi/html/Dadson_Marine_Event_Vision_Harnessing_Event_Cameras_For_Robust_Object_Detection_WACVW_2025_paper.html)*, Proc. Int. IEEE Winter Conf. Applications Comput. Vis. Workshops. (WACVW), 2025.
- Chen Q, Wang H, Ming L, et al. *[Zebrafish Counting Using Event Stream Data](https://arxiv.org/abs/2504.13692)*, arXiv, 2025.


### Sports Equipment
- Nakabayashi T, Kondo A, Higa K, et al., *[Event-based high-speed ball detection in sports video](https://dl.acm.org/doi/abs/10.1145/3606038.3616164)*, Proc. Int. Workshop Multimedia Content Anal. Sports. 2023.
- Ziegler A, Vetter K, Gossard T, et al., *[Spiking neural networks for fast-moving object detection on neuromorphic hardware devices using an event-based camera](https://arxiv.org/abs/2403.10677)*, arXiv, 2024. [Resource](https://cogsys-tuebingen.github.io/snn-edge-benchmark)
- Ziegler A, Vetter K, Gossard T, et al., *[Detection of fast-moving objects with neuromorphic hardware](https://arxiv.org/abs/2403.10677)*, arXiv, 2024
- Ziegler A, Gossard T, Glover A, et al., *[An Event-Based Perception Pipeline for a Table Tennis Robot](https://arxiv.org/abs/2502.00749)*, arXiv, 2025.



<a name="new_directions"></a>
# New Directions

### 3D Object Detection
- <a name="cho2025ev"></a> Cho H, Kang J, Kim Y, et al., *[Ev-3DOD: Pushing the temporal boundaries of 3D object detection with event cameras](https://arxiv.org/abs/2502.19630)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025. [DSEC-3DOD Dataset](https://drive.google.com/drive/folders/1A6XhFxDlqcIgTi28G01fhXBQceaK5vjV?usp=drive_link), [Code](https://github.com/mickeykang16/Ev3DOD)


### Object Detection with Large Language Models
- <a name="liu2024eventgpt"></a> Liu S, Li J, Zhao G, et al., *[EventGPT: Event stream understanding with multimodal large language models](https://arxiv.org/pdf/2412.00832)*, Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025. [Project](https://xdusyl.github.io/eventgpt.github.io/)




___
<br>

<a name="modality"></a>
# Modality [All Publication List]

- ### Single-Modality

- *Updating*

- ### Multimodal Fusion

- *Updating*