Text Classification with Character Recognition using Convolutional Neurals Networks.

The task is to classify paragraph text input, where the text is obfuscated, so that it cannot be read.
Each paragraph belongs to a novel. 

Two ideas were evaluated. 
The ﬁrst one, was to use an LSTM Recurrent neural network, which have been shown to perform well 
in sequences of data, like a text. 
The second method, which is shown is this repository,
 was to use a Convolutional neural network, 
which also has been proven to perform well for text classiﬁcation. 
In Yoon Kim’s paper ”Convolutional Neural Network for Sentence Classiﬁcation”, 
it is showed that a simple CNN can perform really well on various benchmarks, using also 
pretrained word vectors. Although, in our case the fact that our data are obfuscated, 
makes the above method not viable. However, the patterns between the characters are preserved. 
For that reason we can use a CNN for classiﬁcation of the sentences on the character level, 
as shown on the paper ”Character-level Convolutional Networks for Text Classiﬁcation” authored by Xiang Zhang, 
Junbo Zhao, Yann LeCun.

