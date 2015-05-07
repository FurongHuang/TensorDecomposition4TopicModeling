% created by Furong Huang, furongh@uci.edu
clear;clc;
%% Data Specs
n =10000;       % Sample Size
d =100;         % Vocabulary Size
k =3;           % Hidden Dimension
alpha0 =0.01;   % How mixed the topics are
n_test = 100;


alphaprime=abs(randn(1,k));
alpha=alpha0*alphaprime/sum(alphaprime);
Aprime=zeros(n,k);
Aprime_test=zeros(n_test,k);
for j=1:k
  Aprime(:,j)=randg(alpha(j),n,1);
  Aprime_test(:,j)=randg(alpha(j),n_test,1);
end
ListNoGood = sum(Aprime,2)==0;
Aprime(ListNoGood,:)=1;

ListNoGood = sum(Aprime_test,2)==0;
Aprime_test(ListNoGood,:)=1;

A=bsxfun(@rdivide,Aprime,sum(Aprime,2));
A_test=bsxfun(@rdivide,Aprime_test,sum(Aprime_test,2));

Bprime=abs(randn(k,d));
beta=bsxfun(@rdivide,Bprime,sum(Bprime,2));

expectedlen =10000;
len = 2+ poissrnd(expectedlen,1,n);
Counts=mnrnd(len',A*beta);

len_test = 2+ poissrnd(expectedlen,1,n_test);
Counts_test = mnrnd(len',A_test*beta);
%% write synthetic data in the bag of words format
% docID wordID counts
fid = fopen('datasets/synthetic/samples_train.txt','wt');
for index_doc = 1: n
    for index_word = 1:d
        currentCount = Counts(index_doc,index_word);
        if currentCount~=0
            fprintf(fid, '%d\t%d\t%d\t\n',index_doc,index_word,currentCount);
        end
    end
end
fclose(fid);

fid = fopen('datasets/synthetic/samples_test.txt','wt');
for index_doc = 1: n_test
    for index_word = 1:d
        currentCount = Counts_test(index_doc,index_word);
        if currentCount~=0
            fprintf(fid, '%d\t%d\t%d\t\n',index_doc,index_word,currentCount);
        end
    end
end
fclose(fid);
