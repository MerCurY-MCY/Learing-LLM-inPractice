# LearningLLM-InPractice




## Introduction

本仓库专注于在实践中学习LLM基础知识。欢迎想要学习大模型知识的朋友们共同学习交流。

预计将**从头复现GPT-2这一经典的LLM模型**，并在本地机器消费卡上进行训练和推理。同时，会将所有代码进行详细注释和解读。

主要将实现：1.BPE、WordPiece分词模型；2.LayerNorm、MultiHeadAttention、TransformerBlock三个自注意力中的核心模块；3.完整的GPT1/2分词器、模型搭建，并加载参数完成推理；4.MSELoss、CrossEntropy等Loss、SGD、AdamW等Optimizer，并训练一个自己的小型类ChatGPT模型；5.Linux多进程、Pytorch分布式通信以及混合精度和分布式训练；6.全功能trainer开发。


## Todo List

- [x] 分词模型
- [x] LayerNorm、MultiHeadAttention、TransformerBlock
- [x] GPT-2 模型与推理框架
- [x] GPT-2 训练框架
- [ ] 混合精度和分布式训练技术
- [ ] 全功能trainer开发


## References


### GitHub Repositories

1. https://github.com/firechecking/CleanTransformer
### Academic Papers

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.



