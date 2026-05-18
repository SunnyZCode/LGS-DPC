# **LGS-DPC: Density peak clustering integrating local granular ball  information and spherical geodesic distance(LGS-DPC)**

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{LGS-DPC,
title = {LGS-DPC: Density peak clustering integrating local granular ball information and spherical geodesic distance},
journal = {Applied Soft Computing},
pages = {115479},
year = {2026},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2026.115479},
url = {https://www.sciencedirect.com/science/article/pii/S1568494626009270},
author = {Xingguo Zhang and Li Xu and Weikuan Jia},
keywords = {Granular ball computation, Density peaks clustering, K-nearest neighbors, Spherical geodesic distance},
abstract = {Granular ball (GB) density peaks clustering (GB-DP) is a fast-clustering method that applies the concept of coarse-granular GB computation to density peaks clustering (DPC), effectively addressing the slow running efficiency of DPC. However, studies have shown that GB-DP fails to accurately identify cluster centers when dealing with data that has uneven density distributions and cannot effectively handle manifold data. To address these issues, this paper proposes a new Density Peaks Clustering method that integrates local granular ball information and spherical geodesic distances (LGS-DPC). By redesigning the GB generation method, as well as the density and distance calculation approaches, the algorithm is better adapted to handle uneven density distributions and manifold data. First, a simplified adaptive neighbor search algorithm is used to achieve adaptive generation of GBs. Then, by combining the local density information of the GBs themselves and their nearest neighbors, a new density calculation formula is proposed to improve the handling of uneven density distributions and manifold data. Finally, spherical geodesic distances are introduced to replace traditional Euclidean distance calculations, allowing for more precise capture of the geometric features of manifold data. Experiments on synthetic and real datasets against eight advanced baselines show that LGS-DPC achieves a strong accuracy-efficiency trade-off. Compared with the strongest baseline, the average ACC/NMI gains are 5.5%/11.7% on synthetic datasets and 3.1%/1.6% on real datasets.}
}
