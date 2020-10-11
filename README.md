# Light Filed Salient Objetc Detection: A Benchmark

# Light Field SOD:
## LF Models

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
| 12   | 2019 | DLLF   | ICCV   | Deep Learning for Light Field Saliency Detection             | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Learning_for_Light_Field_Saliency_Detection_ICCV_2019_paper.pdf)/[Project](https://github.com/OIPLab-DUT/ICCV2019_Deeplightfield_Saliency) |
| 13   | 2019 | DLSD   | IJCAI  | Deep Light-field-driven Saliency Detection from a Single View | [Paper](https://www.ijcai.org/Proceedings/2019/0127.pdf)/Project |
| 14   | 2019 | MoLF   | NIPS   | Memory-oriented Decoder for Light Field Salient Object  Detection | [Paper](https://papers.nips.cc/paper/8376-memory-oriented-decoder-for-light-field-salient-object-detection.pdf)/[Project](https://github.com/OIPLab-DUT/MoLF) |
| 15   | 2020 | ERNet  | AAAI   | Exploit and Replace: An Asymmetrical Two-Stream Architecture  for Versatile Light Field Saliency Detection | [Paper](https://www.aiide.org/ojs/index.php/AAAI/article/view/6860)/[Project](https://github.com/OIPLab-DUT/AAAI2020-Exploit-and-Replace-Light-Field-Saliency) |
| 16   | 2020 | LFNet  | TIP    | LFNet: Light Field Fusion Network for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9082882)/Project |
| 17   | 2020 | LFDCN   | TIP    | Light Field Saliency Detection with Deep Convolutional  Networks | [Paper](https://arxiv.org/pdf/1906.08331.pdf)/[Project](https://github.com/pencilzhang/MAC-light-field-saliency-net) |
| 18   | 2015 | CS     | NEURO  | Light field saliency vs.2D saliency : A comparative study    | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003410?via%3Dihub)/Project |
| 19   | 2020 |RGBDS   | CVM    | RGB-D Salient Object Detection: A Survey                     | [Paper](https://arxiv.org/pdf/2008.00230.pdf)/[Project](https://github.com/taozh2017/RGBD-SODsurvey) |

## LF Datasets

| No.  | Dataset     | Year | Pub.     | Size | MOP  | FS   | MV   | DE   | ML   | GT   | Raw  | Download |
| ---- | ----------- | ---- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- |
| 1    | LFSD        | 2014 | CVPR     | 100  | 0.04 | :heavy_check_mark:    |      | :heavy_check_mark:    |      | :heavy_check_mark:    | :heavy_check_mark:    | [Link](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/)     |
| 1    | HFUT        | 2017 | ACM TOMM | 255  | 0.29 | :heavy_check_mark:    | :heavy_check_mark:    | :heavy_check_mark:   |      | :heavy_check_mark:   |      | [Link](https://github.com/pencilzhang/MAC-light-field-saliency-net)    |
| 1    | DUT-LF      | 2019 | ICCV     | 1462 | 0.05 | :heavy_check_mark:    |      | :heavy_check_mark:    |      | :heavy_check_mark:    |      | [Link](https://github.com/OIPLab-DUT/ICCV2019_Deeplightfield_Saliency)   |
| 1    | DUT-MV      | 2019 | IJCAI    | 1580 | 0.04 |      | :heavy_check_mark:    |      |      |:heavy_check_mark:    |      | [Link](https://github.com/OIPLab-DUT/IJCAI2019-Deep-Light-Field-Driven-Saliency-Detection-from-A-Single-View)     |
| 1    | Lytro Illum | 2020 | IEEE TIP | 640  | 0.15 |      |      |      | :heavy_check_mark:    | :heavy_check_mark:    | :heavy_check_mark:   | [Link](https://github.com/pencilzhang/MAC-light-field-saliency-net)     |


# Benchmarking results

## PR curves

![PR curves](https://github.com/jiangyao-scu/LFSOD-Survey/blob/main/pictures/PR_curve.png)

## F-measure curves

![F-measure curves](https://github.com/jiangyao-scu/LFSOD-Survey/blob/main/pictures/Fmeasure_curve.png)


