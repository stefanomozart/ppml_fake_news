# Privacy-preserving fake news classification

This repository contains the code and datasets used for the experiments decribed in the paper "Privacy-preserving fake news classification", submited for the Applied Soft Computing Journal.


## Authors
* Stefano M P C Souza - Department of Electrical Engineering, School of Technology, Universidade de Brasília;
* Daniel G Silva - Department of Electrical Engineering, School of Technology, Universidade de Brasília;
* Anderson C A Nascimento - School of Engineering and Technology, University of Washington Tacoma.

## Abstract
Fake news is content shared on digital platforms and presented as legitimate information, which, however, enclose inaccurate, 
incorrect or false data. Theyare known to promote hate speech, manipulate public opinion and influencethe community on topics 
as relevant as national elections. 

In response to such a threat, several governments, political parties andlarge companies began to actively monitor manifestations 
of groups and individuals associated with views considered controversial or extreme. This kind of response raises genuine 
concerns with the protection of privacy and freedom of thought and belief of regular, law abiding citizens.

We propose the use of Privacy-preserving Machine Learning techniques for fake news detection. These techniques can be used to 
provide social media users with timely notice as to the potential quality of the content they are receiving or sharing,  
without exposing them to centralized or overreaching monitoring initiatives. We tested a few use cases of privacy-preserving 
model training and inference, and found the results to indicate the viability of this type of solution for practical use in 
the various digital platforms.

**Keywords:** fake news, privacy-preserving machine learning, securemulti-party computation


## Datasets:
* [FackCk.br](datasets/factck.br): J. a. Moreno, G. Bressan, Factck.br:  A new dataset to study fake news,in:  Proceedings of the 25th Brazillian Symposium on Multimedia andthe  Web,  WebMedia  ’19,  Association  for  Computing  Machinery,  NewYork, NY, USA, 2019, p. 525–527.  doi:10.1145/3323503.3361698;
* [Fake.br](datasets/fake.br): R.  A.  Monteiro,  R.  L.  S.  Santos,  T.  A.  S.  Pardo,  T.  A.  de  Almeida,E. E. S. Ruiz, O. A. Vale, Contributions to the study of fake news inportuguese:  New corpus and automatic detection results,  in:  Compu-tational Processing of the Portuguese Language, Springer InternationalPublishing, 2018, pp. 324–334;
* [Liar](datasets/liar): W. Y. Wang,  “liar,  liar pants on fire”:  A new benchmark dataset forfake news detection, arXiv preprint arXiv:1705.00648 (2017);
* [SBNC](datasets/sbnc): A.  Bharadwaj,  B.  Ashar,  P.  Barbhaya,  R.  Bhatia,  Z.  Shaikh, Source based fake news classification using machine learning (Aug 2020).
 
## Experiments
* [Classic NLP Preprocessing](experiments/classic_nlp.ipynb)
* [BERT based embedings](experiments/bert_embeddings.ipynb)
* [Classic ML](experiments/classic_ml.ipynb)
* [Clear text neural networks](experiments/clear_text_cnn.ipynb)
* [Privacy-preserving neural networks training](experiments/ppml_cnn_training.ipynb)
* [Privacy-preserving neural networks inference](experiments/ppml_cnn_inference.ipynb)

