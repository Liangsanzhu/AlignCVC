<div align="center">
<h2>AlignCVC: Aligning Cross-View Consistency for Single-Image-to-3D Generation</h2>

[Xinyue Liang](https://scholar.google.com/citations?user=R9PlnKgAAAAJ&hl=zh-CN)<sup>\*</sup> |
[Zhiyuan Ma](https://scholar.google.com/citations?user=F15mLDYAAAAJ&hl=en)<sup>\*</sup> | 
[Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ) | 
Yanjun Guo | 
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)

<br>

The Hong Kong Polytechnic University  

<br>

<sup>*</sup> These authors contributed equally.

<h3>📍  AAAI 2026</h3>

</div>

<div>
    <h4 align="center">
     <a href="https://liangsanzhu.github.io/aligncvc.github.io/" target='_blank'>
        <img src="https://img.shields.io/badge/💡-Project%20Page-gold">
        </a>
        <a href="https://arxiv.org/abs/2506.23150" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2506.23150-b31b1b.svg">
        </a>
         <a href="https://github.com/Liangsanzhu/AlignCVC/" target='_blank'>
        <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white">
        </a>
    </h4>
</div>

<p align="center">

<img src="assets/intro.png" alt="Visual Results">

</p>


## 👀 TODO
- [ ] Release inference code.
- [ ] Colab demo for convenient test.
- [ ] Release training code.


## 🌟 Overview Framework

<p align="center">

<img src="assets/pipeline.png" alt="AlignCVC Framework">

</p>


## 🔧 Dependencies and Installation

1. Clone and install threestudio dependencies
    ```bash
    git clone https://github.com/threestudio-project/threestudio.git
    cd threestudio
    pip install -e .
    ```

2. Download AlignCVC and replace the custom module
    ```bash
    cd ..
    git clone https://github.com/Liangsanzhu/AlignCVC.git
    cp -r AlignCVC/custom threestudio/
    # Or remove the original custom directory first, then copy
    # rm -rf threestudio/custom
    # cp -r AlignCVC/custom threestudio/
    ```

3. Navigate to the threestudio directory to start using
    ```bash
    cd threestudio
    ```

**Note**: The system and dataloader code are currently not available, coming soon.
    
## 💬 Contact:
If you have any problem, please feel free to contact me at xinyue.liang@connect.polyu.hk
### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@misc{liang2025aligncvcaligningcrossviewconsistency,
  title={AlignCVC: Aligning Cross-View Consistency for Single-Image-to-3D Generation}, 
  author={Xinyue Liang and Zhiyuan Ma and Lingchen Sun and Yanjun Guo and Lei Zhang},
  year={2025},
  eprint={2506.23150},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.23150}, 
}