# Atom-based_zero-shot_for_antigen-antibody
## abstract

We designed an atom-based prediction method for antibody-antigen compounds. Compared to the latest works, this work uses atom position information instead of residue position to predict the binding of antibodies and antigens.

The general steps are to extract the protein space and sequence data from the dataset of antibody-antigen pairs; after extracting the data, encode the antibody and antigen through the CDConv network to extract features; after feature extraction, use the idea of comparative learning to divide into positive and negative samples for processing; The final model performs a zero-shot attempt to infer the antibody or antigen with the highest binding degree.

## 项目灵感与参考：

 - 参考范老师的CDConv的神经网络模型处理feature的方式；
 - 参考paper dyMEAN-model 里抗原抗体数据信息以及提取方式；
 - 加入MLP提取氨基酸位置信息
 - 使用CLIP模型的思想；
 - 加入Zero-Shot的使用。
