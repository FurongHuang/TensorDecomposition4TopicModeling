//============================================================================
// Name        : TopicModel.cpp
// Type        : main
// Created by  : Furong Huang on 9/25/13
// Version     :
// Copyright   : Copyright (c) 2013 Furong Huang.
//               All rights reserved
// Description : Single Node ALS Topic Modeling
//============================================================================

#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int NX;
int NX_test;
int NA;
int KHID;
bool M2_yes;
bool M3_yes;
double alpha0;
int DATATYPE;
int main(int argc, const char * argv[])
{

	NX = furong_atoi(argv[1]);
	NX_test = furong_atoi(argv[2]);
	NA = furong_atoi(argv[3]);
	KHID = furong_atoi(argv[4]);
	alpha0 = furong_atof(argv[5]);
	DATATYPE = furong_atoi(argv[6]);
	//===============================================================================================================================================================
	// User Manual:
	// (1) Data specs
	// NX is the training sample size
	// NX_test is the test sample size
	// NA is the vocabulary size
	// KHID is the number of topics you want to learn
	// alpha0 is the mixing parameter, usually set to < 1
	// DATATYPE denotes the index convention.
	// -> DATATYPE == 1 assumes MATLAB index which starts from 1,DATATYPE ==0 assumes C++ index which starts from 0 .
	// e.g.  10000 100 500 3 0.01 1
	const char* FILE_GA = argv[7];
	const char* FILE_GA_test = argv[8];
	// (2) Input files
	// $(SolutionDir)\datasets\$(CorpusName)\samples_train.txt
	// $(SolutionDir)\datasets\$(CorpusName)\samples_test.txt
	// e.g. $(SolutionDir)datasets\synthetic\samples_train.txt $(SolutionDir)datasets\synthetic\samples_test.txt
	const char* FILE_alpha_WRITE = argv[9];
	const char* FILE_beta_WRITE = argv[10];
	const char* FILE_hi_WRITE = argv[11];
	// (3) Output files
	// FILE_alpha_WRITE denotes the filename for estimated topic marginal distribution
	// FILE_beta_WRITE denotes the filename for estimated topic-word probability matrix
	// FILE_hi_WRITE denote the estimation of topics per document for the test data.
	// The format is:
	// $(SolutionDir)\datasets\$(CorpusName)\result\alpha.txt
	// $(SolutionDir)\datasets\$(CorpusName)\result\beta.txt
	// $(SolutionDir)\datasets\$(CorpusName)\result\hi.txt
	// e.g. $(SolutionDir)datasets\synthetic\result\alpha.txt $(SolutionDir)datasets\synthetic\result\beta.txt $(SolutionDir)datasets\synthetic\result\hi.txt

	const char* FILE_M2;
	const char* FILE_M3;
	if (argc < 14){
		M2_yes = false;
		M3_yes = false;

	} else if (argc < 16) {
		M3_yes = false;
		if (string(argv[12]).compare("true") == 0){
		       M2_yes = true;
		       FILE_M2 = argv[13]; //M2 file location
		}else if (string(argv[12]).compare("false") == 0){
		       M2_yes = false;
		} else {
		       //error
		      M2_yes = false;
		}

	} else if (argc == 16) {
		if (string(argv[12]).compare("true") == 0){
			M2_yes = true;
			FILE_M2 = argv[13]; //M2 file location
		}else if (string(argv[12]).compare("false") == 0){
			M2_yes = false;
		} else {
			//error
			M2_yes = false;
		}
		if (string(argv[14]).compare("true") == 0){
			M3_yes = true;
			FILE_M3 = argv[15]; //M3 file location
		} else if (string(argv[14]).compare("false") == 0) {
			M3_yes = false;
		} else {
			//error
			M3_yes = false;
		}
	}
	// (4) set of inputs that indicate whether we have direct access to the second and third order moments
	// M2_yes (boolean) indicates if we have the second order moment M2 (this is the 12th input argument)
	// set to "true" if you have M2. It is set to "false" by default
	// FILE_M2 is the filename containing the second order moment (the 13th input argument)
	// M3_yes (boolean) indicates if we have the third order moment M3 (this is the 14th input argument)
	// set to "true" if you have M3. It is set to "false" by default (the 15th input argument)
	// FILE_M3 is the filename containing the third order moment
	//==============================================================================================================================================================
	TIME_start = clock();
	SparseMatrix<double> Gx_a(NX, NA);	Gx_a.resize(NX, NA);
	Gx_a.makeCompressed();
	Gx_a = read_G_sparse((char *)FILE_GA, "Word Counts Training Data", NX, NA);
	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.10e (Seconds)\n", time_readfile);

	SparseMatrix<double> input_M2(Gx_a.cols(), 2 * KHID);
	input_M2.resize(Gx_a.cols(), 2 * KHID);
	if (M2_yes){
		input_M2 = read_G_sparse((char*) FILE_M2, "second order moment", Gx_a.cols(), 2 * KHID);
	}

	MatrixXd input_M3(KHID, KHID * KHID);
	input_M3 = MatrixXd::Zero(KHID, KHID*KHID);
	if (M3_yes){
		input_M3 = read_G_sparse((char*) FILE_M3, "second order moment", KHID, KHID * KHID);
	}

	cout << "(1) Whitening--------------------------" << endl;
	TIME_start = clock();
	SparseMatrix<double> W(NA, KHID); W.resize(NA, KHID); W.makeCompressed();
	VectorXd mu_a(NA);
	SparseMatrix<double> Uw(NA, KHID);  Uw.resize(NA, KHID); Uw.makeCompressed();
	SparseMatrix<double> diag_Lw_sqrt_inv_s(KHID, KHID); diag_Lw_sqrt_inv_s.resize(NA, KHID); diag_Lw_sqrt_inv_s.makeCompressed();
	VectorXd Lengths(NX);
	second_whiten_topic(Gx_a, W, mu_a, Uw, diag_Lw_sqrt_inv_s, Lengths, M2_yes, input_M2);

	// whitened datapoints
	SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();	VectorXd Data_a_mu = W.transpose() * mu_a;

	TIME_end = clock();
	double time_whitening = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.10e (Seconds)\n", time_whitening);

	cout << "(1.5) Matricization---------------------" << endl;
	MatrixXd T(KHID, KHID * KHID);
	if (!M3_yes){
		T = MatrixXd::Zero(KHID, KHID*KHID);
		Compute_M3_topic((MatrixXd)Data_a_G, Data_a_mu, Lengths, T);
	} else if (M3_yes){
		T = input_M3;
	}

	cout << "(2) Tensor decomposition----------------" << endl;
	TIME_start = clock();
	VectorXd lambda(KHID);
	MatrixXd phi_new(KHID, KHID);

    
    VectorXd current_lambda(KHID);
    MatrixXd current_phi(KHID, KHID);
    double err_min = 1000;double current_err=1000;
    int restart_num = 0;
    int whichRun = 0;
    while(restart_num<3){
        cout << "Running ALS " << restart_num << endl;
        current_err = tensorDecom_batchALS(T,current_lambda,current_phi);
        if(current_err <err_min){
            cout << "replace current eigenvalue and eigenvectors with this run"<< endl;
            whichRun = restart_num;
            lambda = current_lambda;
            phi_new = current_phi;
            err_min = current_err;
        }
        restart_num +=1;
    }
    cout << "FINAL ERROR (" << whichRun << "-th run)" <<" : " << err_min << endl;

	TIME_end = clock();
	double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by ALS = %5.10e (Seconds)\n", time_stpm);


	cout << "(3) Unwhitening-----------" << endl;
	TIME_start = clock();
	MatrixXd Inv_Lambda = (pinv_vector(lambda)).asDiagonal();
	SparseMatrix<double> inv_lam_phi = (Inv_Lambda.transpose() * phi_new.transpose()).sparseView();
	SparseMatrix<double> pi_tmp1 = inv_lam_phi * W.transpose();
	VectorXd alpha(KHID);
	MatrixXd beta(NA, KHID);
	Unwhitening(lambda, phi_new, Uw, diag_Lw_sqrt_inv_s, alpha, beta);

	TIME_end = clock();
	double time_post = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for post processing = %5.10e (Seconds)\n", time_post);

	cout << "(4) Writing results----------" << endl;
	write_alpha((char *)FILE_alpha_WRITE, alpha);
	write_beta((char *)FILE_beta_WRITE, beta);


	// decode
	cout << "(5) Decoding-----------" << endl;
	TIME_start = clock();

	SparseMatrix<double> Gx_a_test(NX_test, NA); Gx_a_test.resize(NX_test, NA);
	Gx_a_test.makeCompressed();
	Gx_a_test = read_G_sparse((char *)FILE_GA_test, "Word Counts Test Data", NX_test, NA);
	double nx_test = (double)Gx_a_test.rows();
	double na = (double)Gx_a_test.cols();
	VectorXd OnesPseudoA = VectorXd::Ones(na);
	VectorXd OnesPseudoX = VectorXd::Ones(nx_test);
	VectorXd lengths_test = Gx_a_test * OnesPseudoA;
	lengths_test = lengths_test.cwiseMax(3.0*OnesPseudoX);
	int inference = decode(alpha, beta, lengths_test, Gx_a_test, (char*)FILE_hi_WRITE);
	TIME_end = clock();
	double time_decode = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for decoding = %5.10e (Seconds)\n", time_decode);


	cout << "(6) Program over------------" << endl;
	printf("\ntime taken for execution of the whole program = %5.10e (Seconds)\n", time_whitening + time_stpm + time_post);
	return 0;
}
