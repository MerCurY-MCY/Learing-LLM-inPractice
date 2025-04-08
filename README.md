# LearningLLM-InPractice




## Introduction

本仓库专注于**在实践中学习LLM基础知识**。欢迎有志于学习大模型知识的朋友们follow，共同学习交流，一起进步。

本仓库将会**从头复现GPT2这一经典的LLM模型**，并在本地机器消费卡上进行训练和推理。同时，会将所有代码涉及到的**经典论文进行详细解读并同步上传**，在提升代码能力的同时学习扎实的原理知识，跟进LLM科研进展。

主要将实现：1.BPE、WordPiece分词模型；2.LayerNorm、MultiHeadAttention、TransformerBlock三个自注意力中的核心模块；3.完整的GPT1/2分词器、模型搭建，并加载参数完成推理；4.MSELoss、CrossEntropy等Loss、SGD、AdamW等Optimizer，并训练一个自己的小型类ChatGPT模型；5.Linux多进程、Pytorch分布式通信以及混合精度和分布式训练；6.全功能trainer开发。


## Todo List

- [x] 分词模型
- [x] LayerNorm、MultiHeadAttention、TransformerBlock
- [ ] GPT1/2结构搭建
- [ ] 类ChatGPT模型训练
- [ ] 混合精度和分布式训练
- [ ] 全功能trainer开发


## References


### GitHub Repositories

1. https://github.com/firechecking/CleanTransformer
### Academic Papers

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.



