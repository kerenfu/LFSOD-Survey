# Light Field Salient Object Detection: A Review and Benchmark
(This project site is under construction...)

# Light Field SOD
## Traditional Models
### Table
Table I: Overview of traditional LFSOD models

| No.  | year | models | pub.   | Title                                                        | Links         |
| ---- | ---- | ------ | ------ | ------------------------------------------------------------ | ------------- |
| 1    | 2014 | LFS    | CVPR   | Saliency Detection on Light Field                            | [Paper](https://sites.duke.edu/nianyi/files/2020/06/Li_Saliency_Detection_on_2014_CVPR_paper.pdf)/[Project](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/) |
| 2    | 2015 | WSC    | CVPR   | A Weighted Sparse Coding Framework for Saliency Detection    | [Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_A_Weighted_Sparse_2015_CVPR_paper.pdf)/[Project](https://www.researchgate.net/publication/294874666_Code_WSC) |
| 3    | 2015 | DILF   | IJCAI  | Saliency Detection with a Deeper Investigation of Light Field |[Paper](https://www.ijcai.org/Proceedings/15/Papers/313.pdf)/[Project](https://github.com/pencilzhang/lightfieldsaliency_ijcai15) |
| 4    | 2016 | RL     | ICASSP | Relative location for light field saliency detection         | [Paper](http://sites.nlsde.buaa.edu.cn/~shenghao/Download/publications/2016/11.Relative%20location%20for%20light%20field%20saliency%20detection.pdf)/Project |
| 5    | 2017 | BIF    | NPL    | A Two-Stage Bayesian Integration Framework for Salient Object  Detection on Light Field | [Paper](https://link.springer.com/article/10.1007/s11063-017-9610-x)/Project |
| 6    | 2017 | LFS    | TPMI   | Saliency Detection on Light Field                            | [Paper](https://ieeexplore.ieee.org/document/7570181)/[Project](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/) |
| 7    | 2017 | MA     | TOMM   | Saliency Detection on Light Field: A Multi-Cue Approach      | [Paper](http://www.linliang.net/wp-content/uploads/2017/07/ACMTOM_Saliency.pdf)/Project |
| 8    | 2018 | SDDF   | MTAP   | Accurate saliency detection based on depth feature of 3D images              | [Paper](https://link.springer.com/article/10.1007%2Fs11042-017-5052-8)/Project |
| 9    | 2018 | SGDC   | CVPR   | Salience Guided Depth Calibration for Perceptually Optimized  Compressive Light Field 3D Display | [Paper](https://ieeexplore.ieee.org/document/8578315/)/Project |
| 10   | 2020 | RDFD   | MTAP   | Region-based depth feature descriptor for saliency detection  on light field | [Paper](https://link.springer.com/article/10.1007%2Fs11042-020-08890-x)/Project |
| 11   | 2020 | DCA    | TIP    | Saliency Detection via Depth-Induced Cellular Automata on  Light Field | [Paper](https://ieeexplore.ieee.org/document/8866752)/Project |

## Deep Learning-based Models

### Picture

<img src="https://github.com/kerenfu/LFSOD-Survey/tree/main/pictures/networks.png" width="200" height="100" alt="Frameworks"/><br/>


Fig. 1 Frameworks of deep light field SOD models. (a) Late-fusion ([DLLF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Learning_for_Light_Field_Saliency_Detection_ICCV_2019_paper.pdf)). (b) Middle-fusion ([MoLF](https://papers.nips.cc/paper/8376-memory-oriented-decoder-for-light-field-salient-object-detection.pdf), [LFNet](https://github.com/OIPLab-DUT/AAAI2020-Exploit-and-Replace-Light-Field-Saliency)). (c) Knowledge distillation-based ([ERNet](https://www.aiide.org/ojs/index.php/AAAI/article/view/6860)). (d) Reconstruction-based ([DLSD](https://www.ijcai.org/Proceedings/2019/0127.pdf)). (e) Single-stream ([LFDCN](https://arxiv.org/pdf/1906.08331.pdf)). Note (a)-(c) utilize the focal stack and all-focus image, whereas (d)-(e) utilize the center-view image and micro-lens image array.}


### Table
Table II: Overview of deep learning based LFSOD models

| No.  | year | models | pub.   | Title                                                        | Links         |
| ---- | ---- | ------ | ------ | ------------------------------------------------------------ | ------------- |
| 1    | 2019 | DLLF   | ICCV   | Deep Learning for Light Field Saliency Detection             | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Learning_for_Light_Field_Saliency_Detection_ICCV_2019_paper.pdf)/[Project](https://github.com/OIPLab-DUT/ICCV2019_Deeplightfield_Saliency) |
| 2    | 2019 | DLSD   | IJCAI  | Deep Light-field-driven Saliency Detection from a Single View | [Paper](https://www.ijcai.org/Proceedings/2019/0127.pdf)/Project |
| 3    | 2019 | MoLF   | NIPS   | Memory-oriented Decoder for Light Field Salient Object  Detection | [Paper](https://papers.nips.cc/paper/8376-memory-oriented-decoder-for-light-field-salient-object-detection.pdf)/[Project](https://github.com/OIPLab-DUT/MoLF) |
| 4    | 2020 | ERNet  | AAAI   | Exploit and Replace: An Asymmetrical Two-Stream Architecture  for Versatile Light Field Saliency Detection | [Paper](https://www.aiide.org/ojs/index.php/AAAI/article/view/6860)/[Project](https://github.com/OIPLab-DUT/AAAI2020-Exploit-and-Replace-Light-Field-Saliency) |
| 5    | 2020 | LFNet  | TIP    | LFNet: Light Field Fusion Network for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9082882)/Project |
| 6    | 2020 | LFDCN   | TIP    | Light Field Saliency Detection with Deep Convolutional  Networks | [Paper](https://arxiv.org/pdf/1906.08331.pdf)/[Project](https://github.com/pencilzhang/MAC-light-field-saliency-net) |

## Other Review Works
### Table
Table III: Overview of related reviews and surveys to LFSOD

| No.  | year | models | pub.   | Title                                                        | Links         |
| ---- | ---- | ------ | ------ | ------------------------------------------------------------ | ------------- |
| 1    | 2015 | CS     | NEURO  | Light field saliency vs.2D saliency : A comparative study    | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003410?via%3Dihub)/Project |
| 2    | 2020 |RGBDS   | CVM    | RGB-D Salient Object Detection: A Survey                     | [Paper](https://arxiv.org/pdf/2008.00230.pdf)/[Project](https://github.com/taozh2017/RGBD-SODsurvey) |



# LFSOD Datasets

Overview of light field SOD datasets. About the abbreviations: MOP=Multiple-Object Proportion (The percentage of images regarding the entire dataset, which have more than one objects per image), FS=Focal Stacks, DE=Depth maps, MV=Multi-view images, ML=Micro-lens images, GT=Ground-truth, Raw=Raw light field data. FS, MV, DE, ML, GT and Raw indicate the data provided by the datasets. '✓' denotes the data forms provided in the original datasets, while '✔️' indicates the data forms generated by us. Original data forms as well as supplement data forms can be download at 'All Download'. You can also download original dataset in 'Original Link'.

| No.  | Dataset     | Year | Pub.     | Size | MOP  | FS   | MV   | DE   | ML   | GT   | Raw  | Download(Pass:lfso) | Original Link |
| ---- | ----------- | ---- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | ------------- |
| 1    | LFSD        | 2014 | CVPR     | 100  | 0.04 | ✓    | ✔️  |   ✓  | ✔️   |   ✓  |  ✓  | [All Download](https://pan.baidu.com/s/1gy2JSf8zNuvL0xZWZnc2tQ) | [Link](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/)    |
| 1    | HFUT        | 2017 | ACM TOMM | 255  | 0.29 | ✓ | ✓ | ✓ |    ✔️  |✓  |      | [All Download](https://pan.baidu.com/s/1ZJEhkG73bXS6001ImK0daA) | [Link](https://github.com/pencilzhang/MAC-light-field-saliency-net)    |
| 1    | DUT-LF      | 2019 | ICCV     | 1462 | 0.05 |✓  |      |✓  |      |✓  |      | [All Download](https://pan.baidu.com/s/1XJxjLCpTnKLSLPqWcnGyaw) | [Link](https://github.com/OIPLab-DUT/ICCV2019_Deeplightfield_Saliency)   |
| 1    | DUT-MV      | 2019 | IJCAI    | 1580 | 0.04 |      |✓  |      |      |✓  |      | [All Download](https://pan.baidu.com/s/1KTWXMsm5mdBIR-of4y4s6g) | [Link](https://github.com/OIPLab-DUT/IJCAI2019-Deep-Light-Field-Driven-Saliency-Detection-from-A-Single-View)     |
| 1    | Lytro Illum | 2020 | IEEE TIP | 640  | 0.15 |   ✔️   |    ✔️  |    ✔️  |✓  |✓  |✓  | [All Download](https://pan.baidu.com/s/1-oYq0_69_kFLrFp--emzhg) | [Link](https://github.com/pencilzhang/MAC-light-field-saliency-net)     |



# Benchmarking results
## Table


# Citation
Please cite our paper if you find the work useful: 

	@article{jiang2020light,
  	title={Light Field Salient Object Detection: A Review and Benchmark},
  	author={Jiang, Yao and Zhou, Tao and Ji, Ge-Peng and Fu, Keren and Zhao, Qijun and Fan, Deng-Ping},
  	journal={arXiv preprint arXiv:2010.04968},
  	year={2020}
	}


