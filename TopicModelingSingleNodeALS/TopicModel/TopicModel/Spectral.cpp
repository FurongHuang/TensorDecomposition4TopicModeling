//
//  MathProbabilities.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2013 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#include "stdafx.h"

using namespace Eigen;
using namespace std;
extern int KHID;

pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd(Eigen::MatrixXd A)
{	// works with zero matrix too
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd u = svd.matrixU();
	Eigen::VectorXd s = svd.singularValues();
	Eigen::MatrixXd v = svd.matrixV();
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv;
	mv.first.first = u;
	mv.first.second = v;
	mv.second = s;
	return mv;
}
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> >latenttree_svd(SparseMatrix<double> A)
{
	MatrixXd A_dense = (MatrixXd)A;
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv_dense;
	mv_dense = latenttree_svd(A_dense);
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > mv;
	mv.first.first = mv_dense.first.first.sparseView();
	mv.first.second = mv_dense.first.second.sparseView();
	mv.second = mv_dense.second.sparseView();
	return mv;
}


// pseudo inverse
double pinv_num(double pnum)
{
	double pnum_inv = (fabs(pnum) > TOLERANCE) ? 1.0 / pnum : 0;
	return pnum_inv;
}

Eigen::VectorXd pinv_vector(Eigen::VectorXd pinvvec)
{
	Eigen::VectorXd singularValues_inv(pinvvec.size());
	for (int i = 0; i<pinvvec.size(); ++i) {
		singularValues_inv(i) = (fabs(pinvvec(i)) > TOLERANCE) ? 1.0 / pinvvec(i) : 0;
	}
	return singularValues_inv;
}
Eigen::SparseVector<double> pinv_vector(Eigen::SparseVector<double> pinvvec)
{
	Eigen::SparseVector<double> singularValues_inv;
	singularValues_inv.resize(pinvvec.size());

	for (int i = 0; i<pinvvec.size(); ++i) {
		singularValues_inv.coeffRef(i) = (fabs(pinvvec.coeff(i)) > TOLERANCE) ? 1.0 / pinvvec.coeff(i) : 0;
	}
	singularValues_inv.prune(TOLERANCE);
	return singularValues_inv;
}

Eigen::MatrixXd pinv_matrix(Eigen::MatrixXd pmat)
{
	MatrixXd pinvmat(pmat.cols(), pmat.rows());
	if (pmat.nonZeros() == 0){
		pinvmat = pmat.transpose();
	}
	else{

		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> U_L = latenttree_svd(pmat);
		Eigen::VectorXd singularValues_inv = pinv_vector(U_L.second);
		pinvmat = (U_L.first.second*singularValues_inv.asDiagonal()*U_L.first.first.transpose());
	}
	return pinvmat;
}

// sqrt
Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat)
{

	Eigen::MatrixXd sqrtmat;
	if (pinvmat.nonZeros() == 0){
		sqrtmat = pinvmat.transpose();
	}
	else{
		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> U_L = latenttree_svd(pinvmat);
		Eigen::VectorXd singularValues_sqrt = U_L.second.head(KHID);
		Eigen::MatrixXd left_sing_vec = U_L.first.first.leftCols(KHID);
		Eigen::MatrixXd right_sing_vec = U_L.first.second.leftCols(KHID);
		for (long i = 0; i < KHID; ++i) {
			singularValues_sqrt(i) = sqrt(U_L.second(i));
		}

		sqrtmat = (left_sing_vec*singularValues_sqrt.asDiagonal()*right_sing_vec.transpose());
	}
	return sqrtmat;
}

SparseMatrix<double> embedding_mat(long cols_num, int k)
{
	SparseMatrix<double> phi(cols_num, k);
	for (int i = 0; i < k; i++)
	{
		phi.coeffRef(i, i) = 1.0;
	}
	return phi;
}
//Nystrom for sparseSVD
SparseMatrix<double> random_embedding_mat(long cols_num, int k)
{	// phi.transpose() * phi is always diagonal 
	srand(clock());
	SparseMatrix<double> phi(cols_num, k);
	if (cols_num != k){
		for (int i = 0; i < cols_num; i++)
		{

			int r = rand() % k;							// randomly-------
			int r_sign = rand() % 5;
			double p_rand = (double)r_sign / (double)5.0;
			if (p_rand < 0.5)
				phi.coeffRef(i, r) = -1.0;
		}
	}
	else
	{
		for (int i = 0; i < cols_num; i++){
			phi.coeffRef(i, i) = 1.0;
		}
	}
	phi.makeCompressed();
	phi.prune(TOLERANCE);
	return phi;
}

SparseMatrix<double> random_embedding_mat_dense(long cols_num, int k)
{	// phi.transpose() * phi is always diagonal 
	srand(clock());
	MatrixXd randMat = MatrixXd::Random((int)cols_num, k);
	SparseMatrix<double> phi = randMat.sparseView();
	phi.makeCompressed();	phi.prune(TOLERANCE);
	phi = orthogonalize_cols(phi);
	return phi;
}

SparseMatrix<double> orthogonalize_cols(MatrixXd Y)
{

	unsigned int c = Y.cols();;
	unsigned int r = Y.rows();
	for (unsigned int j = 0; j < c; ++j)
	{
		for (unsigned int i = 0; i < j; ++i)
		{
			double dotij = Y.col(i).dot(Y.col(j));
			Y.col(j) -= dotij * Y.col(i);
		}

		double normsq = Y.col(j).dot(Y.col(j));
		double normsq_sqrt = sqrt(normsq);
		double scale = (normsq_sqrt > TOLERANCE) ? 1.0 / normsq_sqrt : 0.0;

		Y.col(j) *= scale;
	}
	return Y.sparseView();
}



pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> M)
{
	pair< SparseMatrix<double>, SparseVector<double> > USigma;
	int k_prime = 2 * KHID;
	SparseMatrix<double> C;
	if (M.rows() > 20 * KHID)
	{
		SparseMatrix<double> Q;
		if (M.cols() ==M.rows()){
			SparseMatrix<double> RandomMat(M.cols(), k_prime);	RandomMat.resize(M.cols(), k_prime);
			gaussian_matrix((unsigned int) M.cols(), (unsigned int)k_prime, RandomMat);
			Q = M * RandomMat; //Q.makeCompressed(); Q.prune(TOLERANCE);
			Q = orthogonalize_cols((MatrixXd)Q);			
			assert(M.cols() == Q.rows());
			C = M * Q;
		}
		else{
			C = M;
		}
				
		MatrixXd Z = (MatrixXd)C.transpose() * C;
		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> svd_Z = latenttree_svd(Z);
		MatrixXd V = (svd_Z.first.second).leftCols(KHID);
		SparseMatrix<double> V_sparse = V.sparseView();
		VectorXd S = svd_Z.second.head(KHID);
		USigma.second = S.cwiseSqrt().sparseView(); // S.array().sqrt();
		MatrixXd diag_inv_S_sqrt = pinv_vector(S.cwiseSqrt()).asDiagonal();
		SparseMatrix<double> diag_inv_S_sqrt_s = diag_inv_S_sqrt.sparseView();
		USigma.first = C * (V_sparse)* diag_inv_S_sqrt_s;
	}
	else
	{
		pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > mv = latenttree_svd(M);
		USigma.first = mv.first.first.leftCols(KHID);
		USigma.second = mv.second.head(KHID);
	}
	return USigma;
}

pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X){
	// result.first.first is U, result.first.second is V, result.second is L.
	SparseMatrix<double> RandomMat = embedding_mat(X.cols(), 2 * KHID);
	SparseMatrix<double> Gram = X.transpose()*(X*RandomMat);
	pair< SparseMatrix<double>, SparseVector<double> > V_L = SVD_symNystrom_sparse(Gram);
	VectorXd L_inv_vec = pinv_vector((VectorXd)V_L.second);
	L_inv_vec = L_inv_vec.cwiseSqrt();
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	SparseMatrix<double> U = X * (V_L.first) * L_inv_s;

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L;
	UV_L.first.first = U;
	UV_L.first.second = V_L.first;
	UV_L.second = V_L.second.cwiseSqrt();
	return UV_L;
}

pair< SparseMatrix<double>, SparseVector<double> > SVD_Nystrom_columnSpace_sparse(SparseMatrix<double> A, SparseMatrix<double> B){
	// right singular vectors of SVD of A' * B, where A \in R^(n * d_A), B \in R^(n * d_B);
	double norm_denom = 1.0 / (double)A.rows();
	pair< SparseMatrix<double>, SparseVector<double> > USigma;
	// Generate random_mat; 
	int k_prime = 2 * KHID;
	SparseMatrix<double> random_mat = (B.cols() > 20 * KHID) ? random_embedding_mat((long)B.cols(), k_prime) : random_embedding_mat_dense((long)B.cols(), k_prime);
	SparseMatrix<double> tmp = B * random_mat;
	SparseMatrix<double> Q = A.transpose() * tmp;  Q.makeCompressed(); Q.prune(TOLERANCE); tmp.resize(0, 0);
	Q = orthogonalize_cols(Q);
	SparseMatrix<double> tmp2 = B * Q;
	SparseMatrix<double> C = A.transpose() * tmp2;
	C = norm_denom * C;
	MatrixXd Z = (MatrixXd)C.transpose() * C;
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> svd_Z = latenttree_svd(Z);
	MatrixXd V = (svd_Z.first.second).leftCols(KHID);
	SparseMatrix<double> V_sparse = V.sparseView();
	VectorXd S = svd_Z.second.head(KHID);
	USigma.second = S.cwiseSqrt().sparseView(); // S.array().sqrt();
	MatrixXd diag_inv_S_sqrt = pinv_vector(S.cwiseSqrt()).asDiagonal();
	SparseMatrix<double> diag_inv_S_sqrt_s = diag_inv_S_sqrt.sparseView();
	USigma.first = C * (V_sparse)* diag_inv_S_sqrt_s;
	return USigma;

}

/////////////////////

SparseMatrix<double> pinv_Nystrom_sparse(SparseMatrix<double> X){
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans = pinv_Nystrom_sparse_component(X);
	SparseMatrix<double> X_pinv; X_pinv.resize(X.cols(), X.rows());
	X_pinv = V_invL_Utrans.first.first * V_invL_Utrans.second * V_invL_Utrans.first.second;
	return X_pinv;
}

SparseMatrix<double> pinv_aNystrom_sparse(SparseMatrix<double> X){
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L = SVD_asymNystrom_sparse(X);
	// result.first.first is U, result.first.second is V, result.second is L.

	///////////
	VectorXd L_inv_vec = pinv_vector((VectorXd)UV_L.second);
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	SparseMatrix<double> result = UV_L.first.second * L_inv_s * UV_L.first.first.transpose();
	return result;
}


pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > pinv_Nystrom_sparse_component(SparseMatrix<double> X)
{
	// Pinv_symNystrom_sparse_component computes the components of pinv(X). 
	// result.first.first is V, result.first.second is U', result.second is the diag(inv(L));
	// The pseudo inverse of the matrix is result.first.first * result.second * result.first.second. 

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L = SVD_asymNystrom_sparse(X);
	// UV_L.first.first is U, UV_L.first.second is V, UV_L.second is L.
	VectorXd L_inv_vec = pinv_vector((VectorXd)UV_L.second);
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > result;
	result.first.first = UV_L.first.second;
	result.first.second = UV_L.first.first.transpose();
	result.second = L_inv_s;
	return result;
}
