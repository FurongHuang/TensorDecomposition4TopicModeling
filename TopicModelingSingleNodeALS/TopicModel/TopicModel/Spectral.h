//
//  Probability.h
//  CommunityDetection
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#ifndef __TopicModel__Spectral__
#define __TopicModel__Spectral__
#include "stdafx.h"
pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd(Eigen::MatrixXd A);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> >latenttree_svd(SparseMatrix<double> A);
double pinv_num(double pnum);
Eigen::VectorXd pinv_vector(Eigen::VectorXd pinvvec);
Eigen::SparseVector<double> pinv_vector(Eigen::SparseVector<double> pinvvec);
Eigen::MatrixXd pinv_matrix(Eigen::MatrixXd pmat);

Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat);
SparseMatrix<double> embedding_mat(long cols_num, int k);
SparseMatrix<double> random_embedding_mat(long cols_num, int k);
SparseMatrix<double> random_embedding_mat_dense(long cols_num, int k);
SparseMatrix<double> orthogonalize_cols(MatrixXd Y);
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> M);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X);
pair< SparseMatrix<double>, SparseVector<double> > SVD_Nystrom_columnSpace_sparse(SparseMatrix<double> A, SparseMatrix<double> B);
SparseMatrix<double> pinv_Nystrom_sparse(SparseMatrix<double> X);
SparseMatrix<double> pinv_aNystrom_sparse(SparseMatrix<double> X);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > pinv_Nystrom_sparse_component(SparseMatrix<double> X);

#endif