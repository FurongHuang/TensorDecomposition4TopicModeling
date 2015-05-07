SRC=TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp
CFLAGS=-lsvd -lm
COPTZ=-O3 -g  -I.
SUPP_WARN=-Wno-write-strings
N_train = 10000
N_test = 100
Voca_size=100
Hidden_size=3
ALPHA0=0.01
IndexStart=1
DataFolder =../datasets/synthetic
Sample_train=samples_train.txt
Sample_test = samples_test.txt
ResultFolder= $(DataFolder)/result
Result_topic_eigenvalue = corpus_topic_weights.txt
Result_topic_eigenvector = topic_word_matrix.txt
Result_topic_inferred_membership = inferred_topic_weights_per_document.txt

Tensor_dimension = 3
Tensor_file=Tensor.txt
Result_tensor_eigenvalue = tensor_eigenvalue.txt
Result_tensor_eigenvector=tensor_eigenvector.txt

exe-topicmodel: TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp 
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

exe-tensor: M3decomp.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp 
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

runtopic:
	./exe-topicmodel $(N_train) $(N_test) $(Voca_size)  $(Hidden_size)  $(ALPHA0) $(IndexStart) $(DataFolder)/$(Sample_train)  $(DataFolder)/$(Sample_test)  $(ResultFolder)/$(Result_topic_eigenvalue)  $(ResultFolder)/$(Result_topic_eigenvector) $(ResultFolder)/$(Result_topic_inferred_membership)

runtensor:
	./exe-tensor $(Tensor_dimension)  $(Hidden_size)  $(IndexStart) $(DataFolder)/$(Tensor_file)  $(ResultFolder)/$(Result_tensor_eigenvalue)  $(ResultFolder)/$(Result_tensor_eigenvector) 

cleantopic:
	rm exe-topicmodel

cleantensor:
	rm exe-tensor
