# Gemini Update  

We have iterated on Gemini, expanding it into a larger template. In the latest version, Gemini supports different process technologies, DRAM, D2D Interfaces, and packaging technologies. Currently, the latest Gemini supports:  

- **Two process technologies**: 7nm and 12nm  
- **Three D2D Interfaces**: XSR, USR, and UCIe  
- **Three types of DDR**: LPDDR5, GDDR6X, and HBM  
- **Three packaging technologies**: Organic Substrate (OS), Redistribution Layer(RDL), and Silicon Interposer(SI)  

For more details, please refer to the code.  

# How to Run the Project  

First, clone the entire project from GitHub:  

```bash
git clone https://github.com/SET-Scheduling-Project/GEMINI-HPCA2024.git
```
Then, navigate to the project directory:
```bash
cd GEMINI-HPCA2024
make
```
After building, the executable target will be generated at ./build/stschedule.

Next, install the required dependencies:
```bash
pip install -r requirements.txt
```
Now, you can run the DSE script, such as:
```bash
./72tops_dse13.sh
```
Wait for the execution to complete, and you will obtain the architecture exploration results. You can modify the DSE files to perform your own design space exploration experiments.
# Clarification
Many essential foundational components of Gemini are shared with the SET framework from the same project. We are currently updating the architecture and documentation of the SET framework. For urgent inquiries about Gemini, one can refer to the comments in SET (updated at 2025.2.1). Additionally, we plan to migrate and update these annotations from SET to Gemini and include comments related to code unique to Gemini.

For the calculation of the packaging cost in this article, we referred to the article "Chiplet Actuary: A Quantitative Cost Model and Multi-Chiplet Architecture Exploration" published in 2022 DAC and had relevant discussions with the author. We would like to thank the author of the article.

# Citations ###
```
@inproceedings{10.1145/3579371.3589048,
author = {Cai, Jingwei and Wei, Yuchen and Wu, Zuotong and Peng, Sen and Ma, Kaisheng},
title = {Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators},
year = {2023},
isbn = {9798400700958},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3579371.3589048},
doi = {10.1145/3579371.3589048},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
articleno = {13},
numpages = {17},
keywords = {tiled accelerators, neural networks, inter-layer scheduling, scheduling},
location = {Orlando, FL, USA},
series = {ISCA '23}
}

```
```
@inproceedings{cai2024gemini,
  title={Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators},
  author={Cai, Jingwei and Wu, Zuotong and Peng, Sen and Wei, Yuchen and Tan, Zhanhong and Shi, Guiming and Gao, Mingyu and Ma, Kaisheng},
  booktitle={2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  pages={156--171},
  year={2024},
  organization={IEEE}
}
```
