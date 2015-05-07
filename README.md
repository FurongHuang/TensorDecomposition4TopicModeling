#Topic Modeling via Method of Moments

This code performs learning and inference of topic modeles via method of moments using tensor decomposition on a single machine. Specifically the code first performs a pre-processing which implements a whitening transformation for matrix/tensor orthogonalisation and dimensionality reduction and then does "Alternating Least Squares (ALS)". A full description of the code can be found [here](http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/TopicModeling.html).

You can also perform ALS tensor decomposition which is in the M3decomp folder. A full description of the tensor decomposition code can be found [here](http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/M3decomp.html).