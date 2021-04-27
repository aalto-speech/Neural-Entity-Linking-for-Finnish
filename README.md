# Neural-Entity-Linking-for-Finnish
A neural entity linking system for Finnish langauge

The model is trained on a small manually annotated sports magazine data for Finnish.

It utilizes three different inputs: context, mention and candidate. The context is the surrounding text of the named entity. The mention is the named entity for which we need to retrieve the link. The candidate is the entity that the system compares in order to determine if it is the right one.
The candidate, besides the name, contains description and alias information.

The input is processed through a one layer BLSTM, after which the output is concatenated with the embeddings of the mention, generating a context-mention pair. That pair is processed through a fully-connected layer, dropout and a ReLU non-linearity. In the end, it is processed through anohter fully-connected layer and a dropout.

The context-mention pair is compared to the candidates, using cosine similarity.

It uses fastText and entity embeddings. The fastText embeddings can be obtained from here: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz.
The Entity embeddings are trained using the wiki2vec library on the Finnish Wikipedia dump. For more information about the training the entity embeddings, refer to this link: https://wikipedia2vec.github.io/wikipedia2vec/commands/. 
