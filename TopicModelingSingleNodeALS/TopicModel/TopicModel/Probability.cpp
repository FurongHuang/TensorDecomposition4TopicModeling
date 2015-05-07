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
Eigen::MatrixXd normc(Eigen::MatrixXd phi)
{
	for (int i = 0; i < phi.cols(); i++)
	{
		phi.col(i).normalize();
	}

	return phi;
}
Eigen::SparseMatrix<double> normc(Eigen::SparseMatrix<double> phi)
{
	MatrixXd phi_f = normc((MatrixXd)phi);

	return phi_f.sparseView();
}
////////////////////////////////////////////////////////////

Eigen::VectorXd normProbVector(VectorXd P_vec)
{
	VectorXd P_norm = P_vec;
	if (P_vec == Eigen::VectorXd::Zero(P_vec.size())){
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec(row_idx) > 0) ? (P_positive + P_vec(row_idx)) : P_positive;
			P_negative = (P_vec(row_idx) > 0) ? P_negative : (P_negative + P_vec(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm(row_idx) = (P_norm(row_idx)<0) ? 0 : P_norm(row_idx);
		}
	}
	return P_norm;
}

Eigen::SparseVector<double> normProbVector(Eigen::SparseVector<double> P_vec)
{
	VectorXd P_dense_vec = (VectorXd)P_vec;
	Eigen::SparseVector<double> P_norm;
	if (P_dense_vec == VectorXd::Zero(P_vec.size()))
	{
		P_norm = P_vec;
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec.coeff(row_idx) > 0) ? (P_positive + P_vec.coeff(row_idx)) : P_positive;
			P_negative = (P_vec.coeff(row_idx) > 0) ? P_negative : (P_negative + P_vec.coeff(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm.coeffRef(row_idx) = (P_norm.coeff(row_idx)<0) ? 0 : P_norm.coeff(row_idx);
		}
	}
	P_norm.prune(TOLERANCE);
	return P_norm;
}

Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P)
{
	// each column is a probability simplex
	Eigen::MatrixXd P_norm(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		Eigen::VectorXd P_vec = P.col(col);
		P_norm.col(col) = normProbVector(P_vec);
	}
	return P_norm;
}

Eigen::SparseMatrix<double> normProbMatrix(Eigen::SparseMatrix<double> P)
{
	// each column is a probability simplex
	Eigen::SparseMatrix<double> P_norm;
	P_norm.resize(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		//SparseVector<double> A_col_sparse = A_sparse.block(0, i, A_sparse.rows(),1);
		SparseVector<double> P_vec = P.block(0, col, P.rows(), 1);
		SparseVector<double> P_vec_norm;
		P_vec_norm.resize(P_vec.size());
		P_vec_norm = normProbVector(P_vec);
		for (int id_row = 0; id_row < P.rows(); id_row++)
		{
			P_norm.coeffRef(id_row, col) = P_vec_norm.coeff(id_row);
		}
	}
	P_norm.makeCompressed();
	P_norm.prune(TOLERANCE);
	return P_norm;
}

void KhatrioRao(VectorXd A, VectorXd B, VectorXd& output){
	//    cout << "A: " << endl << A << endl;
	//    cout << "B: " << endl << B << endl;
	MatrixXd Out(B.size(), A.size());
	Out = B * A.transpose();
	fflush(stdout);
	//    cout << "Out: "<< endl << Out << endl;
	output = Map<MatrixXd>(Out.data(), A.size()*B.size(), 1);
	//    cout << "KatrioRao output: "<< endl << output << endl;
}

void KhatrioRao(MatrixXd A, MatrixXd B, MatrixXd& output){

	long col_a = A.cols();
	long col_b = B.cols();
	if (col_a != col_b) {
		//  printf("Error : Input matrices must have the same number of columns.");
		fflush(stdout);
		exit(1);
	}
	//    MatrixXd output(A.rows() * B.rows(), col_a);
	for (long id_col = 0; id_col < col_a; id_col++) {
		VectorXd a = A.col(id_col);
		VectorXd b = B.col(id_col);
		VectorXd tmpout(a.size()*b.size());
		KhatrioRao(a, b, tmpout);
		output.col(id_col) = tmpout;
	}
	//    cout << "output: "<< endl << output << endl;
}

void Multip_KhatrioRao(VectorXd T, MatrixXd C_old, MatrixXd B_old, VectorXd& out){
	for (int i = 0; i< C_old.cols(); i++) {
		VectorXd tmpc = C_old.col(i);
		VectorXd tmpb = B_old.col(i);
		double tmpout;
		Multip_KhatrioRao(T, tmpc, tmpb, tmpout);
		out[i] = tmpout;
	}
	//    cout << "out: "<< endl << out << endl;
}
void Multip_KhatrioRao(VectorXd T, VectorXd C_old, VectorXd B_old, double& out){
	VectorXd longvec(C_old.size()*B_old.size());
	KhatrioRao(C_old, B_old, longvec);
	out = (double)T.dot(longvec);
	//    cout << "Multip KatrioRao one element of the first row: "<< endl << out << endl;
}

void Multip_KhatrioRao(MatrixXd T, MatrixXd C_old, MatrixXd B_old, MatrixXd& out){
	for (int i = 0; i < T.rows(); ++i){
		VectorXd tmpT = T.row(i);
		VectorXd tmpout(T.rows());
		Multip_KhatrioRao(tmpT, C_old, B_old, tmpout);
		out.row(i) = tmpout.transpose();
	}
}