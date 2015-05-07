//
//  MathProbabilities.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2013 {Furong Huang} <{furongh@uci.edu}>
* Copyright (C) 2013 {Forough Arabshahi} <farabsha@uci.edu>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#include "stdafx.h"
#include <stdint.h>
extern double alpha0;
extern int KHID;
extern int NX;
extern int NA;
extern int NB;
extern int NC;
extern const char * FILE_Gb_a;
extern const char * FILE_Gc_a;
using namespace Eigen;
using namespace std;

int furong_atoi(string word)
{
	int lol = atoi(word.c_str()); /*c_str is needed to convert string to const char*
								  previously (the function requires it)*/
	return lol;
}
double furong_atof(string word)
{
	double lol = atof(word.c_str()); /*c_str is needed to convert string to const char*
									 previously (the function requires it)*/
	return lol;
}
static uint32_t kiss_x = 123456789, kiss_y = 234567891, kiss_z = 345678912, kiss_w = 456789123, kiss_c = 0;

/* Implementation of a 32-bit KISS generator which uses no multiply instructions */
static uint32_t kiss32()
{
	int t;
	kiss_y ^= (kiss_y << 5);
	kiss_y ^= (kiss_y >> 7);
	kiss_y ^= (kiss_y << 22);
	t = kiss_z + kiss_w + kiss_c;
	kiss_z = kiss_w;
	kiss_c = t < 0;
	kiss_w = t & 2147483647;
	kiss_x += 1411392427;
	return kiss_x + kiss_y + kiss_w;
}

static double urandom(){
	return .0000000002328306089594001f *kiss32() + FLT_EPSILON;
	//return .00000000023283064370807971f*kiss32();
}

//This is faster than Box-Muller but not benchmarked against inverse cdf
static double gaussian_deviate() {
	double u, v, x, x2;
	do{
		u = urandom();
		v = 1.71552776992141359294f*urandom() - .85776388496070679647f;
		x = v / u;
		x2 = x*x;
	} while (x2>6 - 8 * u + 2 * u*u && (x2>2 / u - 2 * u || x2>-4 * log(u)));
	return x;
}


void gaussian_matrix(unsigned int p,unsigned int k, SparseMatrix<double> & Omega_sparse)
{
	MatrixXd Omega = MatrixXd::Zero(p, k);// zeros(p, k);

	for (unsigned int i = 0; i < p; ++i){
		for (unsigned int j = 0; j < k; ++j){
			Omega(i, j) = gaussian_deviate();
		}
	}
	Omega_sparse = Omega.sparseView();
//	return Omega;
}

void accumulate_M_mul_S(SparseMatrix<double> Gx_a, SparseMatrix<double> RandomMat, \
                        SparseMatrix<double> & M2_a, VectorXd & mu_a, VectorXd &Lengths){
    double nx = (double)Gx_a.rows();
    double na = (double)Gx_a.cols();
    VectorXd OnesPseudoA = VectorXd::Ones(na);
    VectorXd OnesPseudoX = VectorXd::Ones(nx);
    VectorXd lengths = Gx_a * OnesPseudoA;
    Lengths = lengths.cwiseMax(3.0*OnesPseudoX);
    
#ifdef NORMALIZE
    double inv_nx = 1.0 / nx;
    
#else
    double inv_nx = 1.0;
#endif
    VectorXd lengthSquareInv = Lengths.cwiseProduct(Lengths - OnesPseudoX).cwiseInverse();
    //	MatrixXd lengthSquareInvMat(lengthSquareInv.size(),Gx_a.cols());
    MatrixXd lengthSquareInvMat(lengthSquareInv.size(),RandomMat.cols());
    //	lengthSquareInvMat = lengthSquareInv.replicate(1, Gx_a.cols());
    lengthSquareInvMat = lengthSquareInv.replicate(1, RandomMat.cols());
    //	SparseMatrix<double> Gx_aSquareNorm = Gx_a.cwiseProduct(lengthSquareInvMat.sparseView());
    SparseMatrix<double> Gx_aProjected = Gx_a*RandomMat;
    SparseMatrix<double> Gx_aSquareNorm = Gx_aProjected.cwiseProduct(lengthSquareInvMat.sparseView());
    lengthSquareInvMat.resize(0,0);
    VectorXd lengthInv = Lengths.cwiseInverse();
    mu_a = inv_nx * (Gx_a.transpose()* lengthInv);
    //1 .  compute M2 Main Term
    double para_main = inv_nx * (alpha0 + 1);
    // ct \otimes ct term :
    SparseMatrix<double>  sec_ord_mom = Gx_a.transpose() * (Gx_aSquareNorm);
    // diagonal(ct) term :
    
    VectorXd tempvect = OnesPseudoX.cwiseProduct(lengthSquareInv);
    VectorXd sec_ord_momDiagVec = Gx_a.transpose() * tempvect;
    vector<Triplet<double> > triplets_sparse;
    for (int i = 0; i < sec_ord_momDiagVec.size(); ++i){
        triplets_sparse.push_back(Triplet<double>(i, i, sec_ord_momDiagVec(i)));
    }
    SparseMatrix<double> sec_ord_momDiag(sec_ord_momDiagVec.size(), sec_ord_momDiagVec.size());
    sec_ord_momDiag.setFromTriplets(triplets_sparse.begin(), triplets_sparse.end());
    SparseMatrix<double> M2 = sec_ord_mom - sec_ord_momDiag*RandomMat; M2 = para_main * M2;
    //2 .  compute M2 Shift Term
    double para_shift = alpha0;
    SparseVector<double> mu_a_sparse = mu_a.sparseView();
    SparseMatrix<double> shiftTerm = mu_a_sparse * (mu_a_sparse.transpose()*RandomMat); shiftTerm = para_shift * shiftTerm;
    M2_a = M2 - shiftTerm;	M2_a.makeCompressed(); M2_a.prune(TOLERANCE);
}
// set of whitening matrix
void second_whiten_topic(SparseMatrix<double> Gx_a,  \
                         SparseMatrix<double> &W, VectorXd &mu_a, SparseMatrix<double> &Uw, SparseMatrix<double> &diag_Lw_sqrt_inv_s, VectorXd &Lengths, bool M2_yes, SparseMatrix<double> &input_M2)
{
    
    
    SparseMatrix<double> RandomMat(Gx_a.cols(), 2 * KHID);	RandomMat.resize(Gx_a.cols(), 2 * KHID);
    gaussian_matrix((unsigned int) Gx_a.cols(), (unsigned int)2 * KHID, RandomMat);
    SparseMatrix<double> M2(Gx_a.cols(), 2 * KHID); M2.resize(Gx_a.cols(), 2 * KHID);
    if (!M2_yes) {
        accumulate_M_mul_S(Gx_a, RandomMat, M2, mu_a, Lengths);
        
        SparseMatrix<double> Q = orthogonalize_cols((MatrixXd)M2);
        accumulate_M_mul_S(Gx_a, Q, M2, mu_a, Lengths);
    }else if(M2_yes) {
        M2 = input_M2;
    }
    
    
    pair< SparseMatrix<double>, SparseVector<double> > Vw_Lw = SVD_symNystrom_sparse(M2);
    Uw = Vw_Lw.first.leftCols(KHID);
    VectorXd Lw = (VectorXd)Vw_Lw.second;
    Lw = pinv_vector(Lw.head(KHID).cwiseSqrt());
    
    MatrixXd diag_Lw_sqrt_inv = Lw.asDiagonal();
    diag_Lw_sqrt_inv_s = diag_Lw_sqrt_inv.sparseView();
    W.resize(Gx_a.cols(), KHID);
    W = Uw * diag_Lw_sqrt_inv_s;
    
    W.makeCompressed(); W.prune(TOLERANCE);
    
}
static int
isspaceornull(char c)
{
	return isspace(c) || c == '\0';
}

static double**
zeros(unsigned int r,
unsigned int c)
{
	double** rv = (double**)malloc(r * sizeof(double*));
	assert(rv != NULL);

	for (unsigned int i = 0; i < r; ++i)
	{
		rv[i] = (double*)calloc(c, sizeof(double));
		assert(rv[i] != NULL);
	}

	return rv;
}

static int
accumulate_matricization(MatrixXd &Ta, VectorXd Wc, unsigned int length, double a0, VectorXd m1, unsigned int kstart, unsigned int kend, unsigned int k)
{
	if (length > 2)
	{

		double scale2fac = a0 * (a0 + 1.0)
			/ (2.0 * length * (length - 1));

		double scale3fac = (a0 + 1.0) * (a0 + 2.0)
			/ (2.0 * length * (length - 1) * (length - 2));

		for (unsigned int i = kstart; i <= kend; ++i)
		{
			for (unsigned int j = 0; j < k; ++j)
			{
				for (unsigned int l = 0; l < k; ++l)
				{
					/* Wc ( Wc \odot Wc )^\top */
					/*topic shift scale3fac first term*/

					Ta(i - kstart,k*j + l) += scale3fac * Wc[i] * Wc[l] * Wc[j];

					/*dirichelet second order term (new term!!!)*/
					Ta(i - kstart,k*j + l) -= scale2fac * Wc[i] * m1[l] * Wc[j];
					Ta(i - kstart,k*j + l) -= scale2fac * Wc[i] * Wc[l] * m1[j];
					Ta(i - kstart,k*j + l) -= scale2fac * m1[i] * Wc[l] * Wc[j];
				}

				/* - \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_i \odot e_j)^\top
				- \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_j \odot e_i)^\top
				- \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_j \odot e_j)^\top
				*/
				/*tpic shift scale3fac 2nd, 3rd, 4th term*/
				Ta(i - kstart,k*i + j) -= scale3fac * Wc[i] * Wc[j];
				Ta(i - kstart,k*j + i) -= scale3fac * Wc[i] * Wc[j];
				Ta(i - kstart,k*j + j) -= scale3fac * Wc[i] * Wc[j];

				/* - \sum_{j=1}^d Wc_j e_j (e_j \odot m1)^\top
				- \sum_{j=1}^d Wc_j e_j (m1 \odot e_j)^\top
				- \sum_{j=1}^d Wc_j m1 (e_j \odot e_j)^\top */
				/*this is problematic*/
				Ta(i - kstart,k*i + j) += scale2fac * Wc[i] * m1[j];
				Ta(i - kstart,k*j + i) += scale2fac * Wc[i] * m1[j];
				Ta(i - kstart,k*j + j) += scale2fac * m1[i] * Wc[j];
			}

			/* + 2 \sum_{i=1}^d Wc_i e_i (e_i \odot e_i)^\top */
			/*topic shift scale3fac fifth term */
			Ta(i - kstart,k*i + i) += 2.0 * scale3fac * Wc[i];
		}
	}

	return (length > 2);
}


void Compute_M3_topic(MatrixXd whitenedData, VectorXd whitenedMean, VectorXd Lengths, MatrixXd &Ta){
	unsigned int k = KHID;
	unsigned int kstart = 0;	unsigned int kend = k-1;	assert(kend < k);
	unsigned int krange = 1 + (kend - kstart);

	

	for (unsigned long long int examples = 0; examples < NX; ++examples){
		VectorXd currentDataPoint = whitenedData.col(examples);
		accumulate_matricization(Ta, currentDataPoint, (unsigned int)Lengths(examples), alpha0, whitenedMean, kstart, kend, k);
	}

	Ta /= NX;
	double alpha0sq = alpha0 * alpha0;
	for (unsigned int i = kstart; i <= kend; ++i){
		for (unsigned int j = 0; j < k; ++j){
			for (unsigned int l = 0; l < k; ++l){
				Ta(i - kstart, k*j + l) += alpha0sq * whitenedMean[i] * whitenedMean[l] * whitenedMean[j];
			}
		}		
	}
	
}

void update_mode_oneiteration(MatrixXd rhs, MatrixXd C_old, MatrixXd B_old, MatrixXd &A_new){
	MatrixXd TobeInvert_tmp(KHID, KHID);
	TobeInvert_tmp = (C_old.transpose() * C_old).cwiseProduct(B_old.transpose()*B_old);
	MatrixXd TobeInvert(KHID, KHID);
	TobeInvert = pinv_matrix(TobeInvert_tmp);
	
	A_new = rhs * TobeInvert;
}
double tensorDecom_batchALS(MatrixXd T, VectorXd & lambda, MatrixXd & A_new){
    bool fail;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID)); MatrixXd B_random(MatrixXd::Random(KHID, KHID)); MatrixXd C_random(MatrixXd::Random(KHID, KHID));
    srand( (unsigned)time(NULL) );
	A_random.setRandom(); B_random.setRandom(); C_random.setRandom();
	HouseholderQR<MatrixXd> qr_A(A_random); HouseholderQR<MatrixXd> qr_B(B_random); HouseholderQR<MatrixXd> qr_C(C_random);
	A_new = qr_A.householderQ(); MatrixXd B_new = qr_A.householderQ(); MatrixXd C_new = qr_A.householderQ();
	MatrixXd A_old = A_new; MatrixXd B_old = B_new; MatrixXd C_old = C_new;
	long iteration = 1;
	double error;
	MatrixXd rhs(KHID, KHID);
    MatrixXd T_est(T.rows(),T.cols());
	while (true)
	{
        // update mode A
        Multip_KhatrioRao(T, C_old, B_old, rhs);
        update_mode_oneiteration(rhs, C_old, B_old, A_new);
        
        lambda = ((A_new.array().pow(2)).colwise().sum()).pow(1.0 / 2.0);
        // convergence check
        if (iteration > MINITER)
        {
            tensorReconstruct(T_est, A_new, B_new, C_new, lambda);
            error = (T_est - T).norm() ;// max(T.norm(), TOLERANCE);            
            if (error < 1e-4)
            {
//                cout << "err: " << error << endl;
                A_new = normc(A_new);
                fail = 0;
                break;
            }
            else
                if (iteration > MAXITER)
            {
                fail = 1;
                break;
            }
            
        }
        A_new = normc(A_new);
		A_old = A_new;
		// update mode B
		Multip_KhatrioRao(T, C_old, A_old, rhs);
		update_mode_oneiteration(rhs, C_old, A_old, B_new);
		B_new = normc(B_new);
		B_old = B_new;
		// update mode C
		Multip_KhatrioRao(T, B_old, A_old, rhs);
		update_mode_oneiteration(rhs, B_old, A_old, C_new);
		C_new = normc(C_new);
		C_old = C_new;
		
		iteration++;
	}
    return error;
}

void tensorReconstruct(MatrixXd &T, MatrixXd A, MatrixXd B, MatrixXd C,VectorXd lambda){
    if (B.cols() != C.cols()) {
        //  printf("Error : Input matrices must have the same number of columns.");
        fflush(stdout);
        exit(1);
    }
    for (int i =0; i< lambda.size(); ++i){
        lambda(i) = pow(lambda(i),1/3);
    }
    MatrixXd Lambda_diag = lambda.asDiagonal();
    A =A * Lambda_diag;
    B =B * Lambda_diag;
    C =C * Lambda_diag;
    MatrixXd output(B.rows()*C.rows(),B.cols());
    KhatrioRao(C, B, output);
    T = A*output.transpose();
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void tensorDecom_alpha0_topic(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, VectorXd Lengths, VectorXd &lambda, MatrixXd & phi_new)
{
	double error; MatrixXd phi_old;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	double inv_x = 1.0 / ((double)D_a_mat.cols());
	double para_main = (alpha0 + 1.0)*(alpha0 + 2.0) / 2.0 * inv_x;
	double para_shift1 = -alpha0 *(alpha0 + 1.0) / 2.0 * inv_x;
	double para_shift0 = alpha0*alpha0;
	// find norm terms
	VectorXd OnesPseudoX = VectorXd::Ones(NX);
	VectorXd lengthInv0 = Lengths.cwiseInverse();
	VectorXd lengthInv1 = (Lengths - OnesPseudoX).cwiseInverse();
	VectorXd lengthInv2 = (Lengths - 2.0* OnesPseudoX).cwiseInverse();
	// VectorXd lengthSquareInv = lengthInv1.cwiseProduct(lengthInv2);
	VectorXd lengthSquareInv = lengthInv0.cwiseProduct(lengthInv1);
	VectorXd lengthCubInv = lengthInv2.cwiseProduct(lengthSquareInv);
	// VectorXd lengthCubInv = Lengths.cwiseProduct(lengthSquareInv);


	MatrixXd lengthInv0Mat = (lengthInv0.transpose()).replicate(D_a_mat.rows(), 1);
	MatrixXd lengthInv1Mat = (lengthInv1.transpose()).replicate(D_a_mat.rows(), 1);
	MatrixXd lengthInv2Mat = (lengthInv2.transpose()).replicate(D_a_mat.rows(), 1);

	MatrixXd lengthSquareInvMat = (lengthSquareInv.transpose()).replicate(D_a_mat.rows(), 1);
	MatrixXd lengthCubInvMat = (lengthCubInv.transpose()).replicate(D_a_mat.rows(), 1);

	SparseMatrix<double> Gx_aNorm0 = D_a_mat.cwiseProduct(lengthInv0Mat.sparseView());
	SparseMatrix<double> Gx_aNorm1 = D_a_mat.cwiseProduct(lengthInv1Mat.sparseView());
	SparseMatrix<double> Gx_aNorm2 = D_a_mat.cwiseProduct(lengthInv2Mat.sparseView());

	SparseMatrix<double> Gx_aSquareNorm = D_a_mat.cwiseProduct(lengthSquareInvMat.sparseView());
	SparseMatrix<double> Gx_aCubNorm = D_a_mat.cwiseProduct(lengthCubInvMat.sparseView());
	

	//


	phi_new = qr.householderQ();
	lambda = VectorXd::Zero(KHID);
	SparseMatrix<double> pair_aa = Gx_aNorm0 * Gx_aNorm1.transpose(); MatrixXd Pair_aa = (MatrixXd)pair_aa;

	A_random.resize(0, 0);
	long iteration = 1;
	while (true)
	{
		long iii = iteration % NX;
		phi_old = phi_new;
		for (int index_k = 0; index_k < KHID; index_k++){
			VectorXd curr_eigenvec = phi_old.col(index_k);
			VectorXd OrthoCost = tensor_form_orthcost_topic(ORTH_COST_WEIGHT, phi_old, index_k);
			VectorXd CorreRewd_0 = para_main * tensor_form_main_topic(Gx_aNorm0, Gx_aNorm1, Gx_aNorm2, Gx_aSquareNorm, Gx_aCubNorm, curr_eigenvec);
			VectorXd CorreRewd_1 = para_shift1 * tensor_form_shift1_topic(Pair_aa, D_a_mu, curr_eigenvec);
			VectorXd CorreRewd_2 = para_shift0*tensor_form_shift0_topic(D_a_mu, curr_eigenvec);
			VectorXd CorreRewd = CorreRewd_0 + CorreRewd_1 + CorreRewd_2;

			phi_new.col(index_k) = CorreRewd - OrthoCost;
			//para_main * tensor_form_main_topic(Gx_aNorm0, Gx_aNorm1, Gx_aNorm2, Gx_aSquareNorm, Gx_aCubNorm, curr_eigenvec) \
				+ para_shift1 * tensor_form_shift1_topic(Pair_aa, D_a_mu, curr_eigenvec)\
				+ para_shift0*tensor_form_shift0_topic(D_a_mu, curr_eigenvec)\
				- tensor_form_orthcost_topic(ORTH_COST_WEIGHT, phi_old, index_k);
//			cout << "OrthoCost: " << endl << OrthoCost << endl;
//			cout << "CorreRewd: " << endl << CorreRewd << endl;
//			cout << "CorreRewd_0: " << endl << CorreRewd_0 << endl;
			// Add the orthogonality cost term

		}
		lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
		phi_new = normc(phi_new);

		if (iteration < MINITER){}
		else
		{
			error = (phi_new - phi_old).norm();
			cout << "error: " << error << endl;
			if (error < TOLERANCE || iteration > MAXITER)
			{
				cout << " coverged iteration: " << iteration << endl;
				break;
			}
		}

		iteration++;
	}
	
}


VectorXd tensor_form_orthcost_topic(double theta, MatrixXd curr_eigenmat, int curr_ind){
	VectorXd curr_eigenvec = curr_eigenmat.col(curr_ind);
	VectorXd eigenvec_sum = VectorXd::Zero(curr_eigenvec.size());
	for (int indx = 0; indx < curr_eigenmat.cols(); ++indx){
		VectorXd this_eigenvec = curr_eigenmat.col(indx);
		double value = curr_eigenvec.dot(this_eigenvec);
		value = value*value;
		eigenvec_sum = eigenvec_sum + value* this_eigenvec;
	}
	eigenvec_sum = theta * eigenvec_sum;

	return eigenvec_sum;
}

VectorXd tensor_form_main_topic(SparseMatrix<double> Gx_aNorm0, SparseMatrix<double> Gx_aNorm1, \
	 SparseMatrix<double> Gx_aNorm2, SparseMatrix<double> Gx_aSquareNorm, SparseMatrix<double> Gx_aCubNorm, VectorXd curr_eigenvec){
	// T(.,vi,vi) = 1/n sum [1/l(l-1)(l-2)]  { (ct'*vi)^2 ct - }
	// find norm terms
	VectorXd D_b_f = Gx_aNorm1.transpose() * curr_eigenvec; // vector of dimension NX
	VectorXd D_c_f = Gx_aNorm2.transpose() * curr_eigenvec;
	VectorXd D_bc_f = D_b_f.cwiseProduct(D_c_f);		// vector of dimensionn NX, coefficient for each sample... 
	// shifts1
	MatrixXd eigenMat = curr_eigenvec.replicate(1, Gx_aSquareNorm.cols()); //D_a_mat.cols() = NX;
	SparseMatrix<double> EigenMat = eigenMat.sparseView();
	EigenMat = EigenMat.cwiseProduct(Gx_aSquareNorm);
	VectorXd D_bc_fShift1 = EigenMat.transpose()* curr_eigenvec;
	// shifts2
	VectorXd D_bc_fShift2 = 2.0* ((curr_eigenvec.transpose() * Gx_aNorm0) * EigenMat.transpose()).transpose();
	
	// shifts3
	VectorXd OnesPseudoX = VectorXd::Ones(NX);
	VectorXd sumD_a_mat = Gx_aCubNorm * OnesPseudoX;
	MatrixXd eigenSquare = curr_eigenvec.cwiseProduct(curr_eigenvec).asDiagonal();
	VectorXd D_bc_fShift3 = 2.0* (eigenSquare * sumD_a_mat); 
	return Gx_aNorm0*D_bc_f - Gx_aNorm0* D_bc_fShift1 - D_bc_fShift2 + D_bc_fShift3;
}

VectorXd tensor_form_shift1_topic(MatrixXd Pair_aa, VectorXd D_a_mu, VectorXd curr_eigenvec){
	double coeff_uA = curr_eigenvec.transpose() * Pair_aa * curr_eigenvec;
	double coeff_uB = curr_eigenvec.transpose() * D_a_mu;
	VectorXd coeff_A = coeff_uA * D_a_mu;
	VectorXd coeff_B = coeff_uB * (Pair_aa * curr_eigenvec);
	//topic shifts
	return coeff_A + coeff_B + coeff_B;
}
VectorXd tensor_form_shift0_topic(VectorXd D_a_mu,  VectorXd curr_eigenvec){
	double coeff1 = curr_eigenvec.transpose()*D_a_mu;
	return (coeff1*coeff1)* D_a_mu;
}


void tensorDecom_alpha0_online(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new)
{
	double error;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	MatrixXd phi_old;

	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	phi_new = qr.householderQ();
	lambda = VectorXd::Zero(KHID);
	A_random.resize(0, 0);
	long iteration = 1;
	cout << "phi_new: " << phi_new << endl;
	while (true)
	{
		long iii = iteration % NX;
		VectorXd D_a_g = D_a_mat.col((int)iii);//
		VectorXd D_b_g = D_b_mat.col((int)iii);
		VectorXd D_c_g = D_c_mat.col((int)iii);
		double learningrate = min(1e-9, 1.0 / sqrt((double)iteration));

		phi_old = phi_new;
		phi_new = Diff_Loss(D_a_g, D_b_g, D_c_g, D_a_mu, D_b_mu, D_c_mu, phi_old, learningrate);
		cout << "phi_new: " << phi_new << endl;
		///////////////////////////////////////////////
		if (iteration < MINITER)
		{
		}
		else
		{
			error = (normc(phi_new) - normc(phi_old)).norm();
			cout << "error: " << error << endl;
			if (error < TOLERANCE || iteration > MAXITER)
			{

				break;

			}
		}

		iteration++;
	}
	lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
	phi_new = normc(phi_new);
	cout << "lambda: " << lambda << endl;
	cout << "phi_new: " << endl << phi_new << endl;
}

MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, MatrixXd phi, double learningrate)
{
	MatrixXd New_Phi;
	double theta = 10000;

	MatrixXd myvectors = MatrixXd::Zero(KHID, KHID);
	cout << "phi: " << endl << phi << endl;
	cout << "learningrate: " << learningrate << endl;
	cout << "Data_a_g: " << Data_a_g.transpose() << endl;
	cout << "Data_b_g: " << Data_b_g.transpose() << endl;
	cout << "Data_c_g: " << Data_c_g.transpose() << endl;
	for (int index_k = 0; index_k < KHID; index_k++)
	{
		VectorXd curr_eigenvec = phi.col(index_k);
		VectorXd SquareTerm = (curr_eigenvec.transpose()*phi).array().pow(2).transpose();
		MatrixXd The_first_term_noSum = phi * SquareTerm.asDiagonal();
		VectorXd vector_term1 = (3.0*theta) * The_first_term_noSum.rowwise().sum();
		cout << "vector_term1: " << vector_term1.transpose() << endl;
		VectorXd vector_term2 = -3.0 * The_second_term(Data_a_g, Data_b_g, Data_c_g, Data_a_mu, Data_b_mu, Data_c_mu, curr_eigenvec);
		cout << "vector_term2: " << vector_term2.transpose() << endl;
		myvectors.col(index_k) = vector_term1 + vector_term2;

	}
	cout << "myvectors: " << endl << myvectors << endl;
	New_Phi = phi - myvectors*learningrate;
	cout << "New_Phi: " << endl << New_Phi << endl;
	return New_Phi;
}

VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi)
{
	// phi is a VectorXd
	double para0 = (alpha0 + 1.0)*(alpha0 + 2.0) / 2.0;
	double para1 = alpha0*alpha0;
	double para2 = -alpha0 *(alpha0 + 1.0) / 2.0;

	VectorXd Term1 = para0*(phi.dot(Data_a_g))*(phi.dot(Data_b_g)) * Data_c_g;
	VectorXd Term2 = para1*(phi.dot(Data_a_mu))*(phi.dot(Data_b_mu))*Data_c_mu;
	VectorXd Term31 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_g))*Data_c_mu;
	VectorXd Term32 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_mu))*Data_c_g;
	VectorXd Term33 = para2*(phi.dot(Data_a_mu))*(phi.dot(Data_b_g))*Data_c_g;
	VectorXd output = Term1 + Term2 + Term31 + Term32 + Term33;
	cout << "(phi.dot(Data_a_g)): " << (phi.dot(Data_a_g)) << endl;
	cout << "(phi.dot(Data_b_g)): " << (phi.dot(Data_b_g)) << endl;
	cout << "Data_c_g: " << Data_c_g.transpose() << endl;
	cout << "output: " << output.transpose() << endl;
	return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Unwhitening(VectorXd lambda, MatrixXd eigenvec, SparseMatrix<double> Uw, SparseMatrix<double> diag_Lw_sqrt_inv, VectorXd & alpha, MatrixXd &R_A)
{
	alpha = normProbVector(lambda.array().pow(-2));	alpha = alpha / alpha.sum(); alpha = alpha0 * alpha;

	MatrixXd Lambda_diag = lambda.asDiagonal();
	MatrixXd diag_Lw_sqrt = MatrixXd::Zero(diag_Lw_sqrt_inv.cols(), diag_Lw_sqrt_inv.rows()); 
	for (int i = 0; i < diag_Lw_sqrt_inv.rows(); i++){
		diag_Lw_sqrt(i, i) = (fabs(diag_Lw_sqrt_inv.coeff(i, i)) < 1e-6) ? 0 : 1.0 / diag_Lw_sqrt_inv.coeff(i, i);
	}
	MatrixXd Uw_dense = (MatrixXd)Uw;
	R_A = Uw_dense * diag_Lw_sqrt;
	R_A = R_A * eigenvec;
	R_A = R_A * Lambda_diag;

	for (int j = 0; j < R_A.cols(); j++)
	{
		R_A.col(j) = R_A.col(j) / (R_A.col(j).cwiseAbs().sum());
		VectorXd tmp1(NA); VectorXd tmp2(NA);
		normProbVectorJohn(R_A.col(j), tmp1);
		double err_tmp1 = (tmp1 - R_A.col(j)).norm();
		normProbVectorJohn(-R_A.col(j), tmp2);
		double err_tmp2 = (tmp2 + R_A.col(j)).norm();
		R_A.col(j) = (err_tmp1 < err_tmp2) ? tmp1 : tmp2;
	}
}

void normProbVectorJohn(VectorXd V, VectorXd & P_norm){
	double eps = 1e-5;
	double z = 1.0;
	const int len = V.size();

	VectorXd U = V;
	sort(U.data(), U.data() + U.size());
	reverse(U.data(), U.data() + U.size());

	VectorXd cums = U;
	for (int j = 1; j<U.size(); ++j)
		cums(j) += cums(j - 1);

	VectorXd Onevec = VectorXd::Ones(len);
	VectorXd Index = Onevec;
	for (int j = 1; j<Onevec.size(); ++j)
		Index(j) += Index(j - 1);


	Index = Index.cwiseInverse();
	VectorXd InterVec = Index.cwiseProduct(cums - Onevec);
	VectorXd TobefindMax = U - InterVec;
	long maxIndex;
	for (long i = len - 1; i > -1; i--){
		if (TobefindMax(i) > 0){
			maxIndex = i;
			break;
		}
	}
	double theta = InterVec(maxIndex);
	VectorXd W = V - theta* Onevec;

	for (int i = 0; i < len; i++){
		P_norm(i) = (W(i)>0) ? W(i) : 0.0;

	}
}



int decode(VectorXd alpha, MatrixXd beta, VectorXd DOCLEN, SparseMatrix<double> corpus,char* filename){
	

	double ll_sum = 0;
	int Maxiter = 10;
	fstream f(filename, ios::out);
	f << "-1\t" << "DOCLEN" << "\t" <<"topic estimation" << endl;
	for (int data_i = 0; data_i < corpus.rows(); data_i++){
		vector<int> my_doc_word; 
		vector<double> my_doc_count; 
		VectorXd h(KHID);
		// find out my_doc_word and my_doc_count
		SparseVector<double> my_document = corpus.block(data_i, 0, 1, corpus.cols());

		for (SparseVector<double>::InnerIterator it(my_document); it; ++it)
		{
			my_doc_count.push_back(it.value()); 
			my_doc_word.push_back(it.index());
		}

		double ll = 0;
		lda_inference(beta, my_doc_word, my_doc_count, DOCLEN(data_i), alpha, h, ll, Maxiter);
		f << data_i << "\t" << DOCLEN(data_i) ;
		for (long id = 0; id < h.size(); id++) {
			f << "\t" << h(id);
		}
		f << endl;

		ll_sum += ll;
	}
	VectorXd dummy(KHID);
	dummy = VectorXd::Ones(KHID);
	dummy = -dummy;
	double sumDOCLEN = DOCLEN.sum();

//	f << "-1\t" << sumDOCLEN << "\t" << ll_sum << "\t" << dummy.transpose() << endl;


	double pp = exp(-ll_sum / sumDOCLEN);

	f << "-1\t" << sumDOCLEN << "\t" << dummy.transpose() << endl;
	f << "number of words in total : " << sumDOCLEN << endl;
	f << "log likelihood: " << ll_sum << endl;
	f << "perplexity score for k =" << KHID << ", alpha0 =" << alpha0 << ": " << pp << endl;

	f.close();
	return 0;
}


void lda_inference(MatrixXd R, vector<int> my_doc_word, vector<double> my_doc_count, double total, VectorXd alpha, VectorXd& h, double& likelihood, int Maxiter)
{

	static int num_topics = R.cols();
	double TOL = 1e-4; double SMOOTH = 1e-4; VectorXd SmoothVec = VectorXd::Ones(num_topics); SmoothVec = SMOOTH * SmoothVec;
	double alpha0 = alpha.sum();
//	double total = std::accumulate(my_doc_count.begin(), my_doc_count.end(), 0);

	long d = R.rows();


	double converged = 1;
	double phisum = 0;
	// likelihood = 0;
	double likelihood_old = 0;
	vector<double> oldphi(num_topics);
	int k, n, var_iter;
	vector<double> digamma_gam(num_topics);


	double* var_gamma; // initialize the size
	var_gamma = (double*)malloc(sizeof(double)* num_topics);
	assert(var_gamma != NULL);
	double** phi; // initialize the size
	phi = (double**)malloc(sizeof(double*)*my_doc_count.size());
	assert(phi != NULL);
	for (n = 0; n < my_doc_count.size(); ++n){
		phi[n] = (double*)malloc(sizeof(double)* num_topics);
		assert(phi[n] != NULL);
	}
	// compute posterior dirichlet

	for (k = 0; k < num_topics; k++)
	{
		var_gamma[k] = alpha[k] + (total / ((double)num_topics));
		digamma_gam[k] = digamma(var_gamma[k]);
		for (n = 0; n < my_doc_word.size(); n++)
			phi[n][k] = 1.0 / num_topics;
	}
	var_iter = 0;

	while ((converged > TOL) && ((var_iter < Maxiter) || (Maxiter == -1)))
	{
		var_iter++;
		for (n = 0; n < my_doc_word.size(); n++)
		{
			int word_id = my_doc_word[n];
			VectorXd currrowR = (R.block(my_doc_word[n], 0, 1, k)).transpose();//(VectorXd)R.block(0, word_id, R.rows(), 1);
			//			cout << "currrowR: " << currrowR << endl;
			currrowR = vec_log(currrowR + SmoothVec);
			//			cout << "log currrowR: " << currrowR << endl;
			double * rowR = currrowR.data();
			//			cout << " log currrowR vals: " << rowR[0] << "," << rowR[1] << "," << rowR[2] << endl;
			phisum = 0;
			for (k = 0; k < num_topics; k++)
			{
				oldphi[k] = phi[n][k];
				phi[n][k] = digamma_gam[k] + rowR[k];

				if (k > 0)
					phisum = log_sum(phisum, phi[n][k]);
				else
					phisum = phi[n][k]; // note, phi is in log space
			}

			for (k = 0; k < num_topics; ++k)
			{
				phi[n][k] = exp(phi[n][k] - phisum);
				var_gamma[k] = var_gamma[k] + my_doc_count[n] * (phi[n][k] - oldphi[k]);
				digamma_gam[k] = digamma(var_gamma[k]);
			}
		}

		likelihood = compute_likelihood(R, my_doc_word, my_doc_count, alpha, phi, var_gamma);// this is like vlb in Nikos' code
		assert(!isnan(likelihood));
		converged = (likelihood_old - likelihood) / likelihood_old;
		likelihood_old = likelihood;

	}
	for (k = 0; k < num_topics; ++k)
	{
		h[k] = var_gamma[k];
	}
	h = h / h.sum();
}


/*
* compute likelihood bound
*
*/

double compute_likelihood(MatrixXd R, vector<int> my_doc_word, vector<double> my_doc_count, VectorXd alpha, double** phi, double* var_gamma)
{
	static int num_topics = R.cols();
	double SMOOTH = 1e-4; VectorXd SmoothVec = VectorXd::Ones(R.rows()); SmoothVec = SMOOTH*SmoothVec;
	double alpha0 = alpha.sum();
	double likelihood = 0, digsum = 0, var_gamma_sum = 0; vector<double> dig(num_topics);
	int k, n;
	double sumlgammaAlpha = 0;
	for (k = 0; k < num_topics; k++)
	{
		sumlgammaAlpha += lgamma(alpha[k]);
		dig[k] = digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
	}
	digsum = digamma(var_gamma_sum);


	likelihood = lgamma(alpha0) - sumlgammaAlpha - (lgamma(var_gamma_sum));

	for (k = 0; k < num_topics; k++)
	{
		likelihood += (alpha[k] - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);



		VectorXd currrowR = (R.block(0, k, R.rows(), 1));//(VectorXd)R.block(0, word_id, R.rows(), 1) this is a column
		currrowR = vec_log(currrowR + SmoothVec);
		double * rowR = currrowR.data();

		for (n = 0; n < my_doc_word.size(); n++)
		{
			if (phi[n][k] > 0)
			{
				likelihood += my_doc_count[n] * (phi[n][k] * ((dig[k] - digsum) - log(phi[n][k]) + rowR[my_doc_word[n]]));
			}
		}
	}
	return(likelihood);
}

VectorXd eigen_gamma(VectorXd a){
	VectorXd gamma_a(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		gamma_a(i) = exp(lgamma(a(i)));
		//		cout << "gamma of " << a(i) << ": " << gamma_a(i) << endl;
	}
	return gamma_a;
}

VectorXd vec_log(VectorXd a){
	VectorXd log_a(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		log_a(i) = log(a(i));
		//		cout << "a(i): " << a(i) << "log a(i):" << log_a(i) << endl;
	}
	return log_a;
}

/*
* given log(a) and log(b), return log(a + b)
*
*/

double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b)
	{
		v = log_b + log(1 + exp(log_a - log_b));
	}
	else
	{
		v = log_a + log(1 + exp(log_b - log_a));
	}
	return(v);
}

/*
* taylor approximation of first derivative of the log gamma function
*
*/

double digamma(double x)
{
	double p;
	x = x + 6;
	p = 1 / (x*x);
	p = (((0.004166666666667*p - 0.003968253986254)*p +
		0.008333333333333)*p - 0.083333333333333)*p;
	p = p + log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6);
	return p;
}


double log_gamma(double x)
{
	double z = 1 / (x*x);

	x = x + 6;
	z = (((-0.000595238095238*z + 0.000793650793651)
		*z - 0.002777777777778)*z + 0.083333333333333) / x;
	z = (x - 0.5)*log(x) - x + 0.918938533204673 + z - log(x - 1) -
		log(x - 2) - log(x - 3) - log(x - 4) - log(x - 5) - log(x - 6);
	return z;
}

void estimate_h_ll(SparseMatrix<double> R, vector<long> my_doc_word, vector<double> my_doc_count, VectorXd alpha, VectorXd& h, double& ll, int Maxiter)
{
	double TOL = 1e-4;
	long d = R.rows();
	int k = R.cols();

	double SMOOTH = 1e-4;
	VectorXd SmoothVec = VectorXd::Ones(k);
	SmoothVec = SMOOTH*SmoothVec;
	VectorXd h_old(k);
	VectorXd h_new = VectorXd::Ones(k);
	h_new = h_new / k;
	for (int iter = 0; iter < Maxiter; iter++)
	{
		h_old = h_new;
		VectorXd S = VectorXd::Zero(k);
		for (unsigned int id = 0; id < my_doc_count.size(); id++)
		{
			long word_id = my_doc_word[id];
			VectorXd currrow = (R.block(word_id, 0, 1, k)).transpose();//(VectorXd)R.block(0, word_id, R.rows(), 1);
			if (currrow.sum() == 0){
				currrow = SmoothVec;
			}
			VectorXd phi(k);
			phi = h_old.cwiseProduct(currrow);
			if (phi.sum() == 0){
				phi = SmoothVec;
			}
			phi = phi / phi.sum();
			S = S + (my_doc_count[id]) * phi;

		}

		h_new = S;


		if ((h_old - h_new).norm() / max(h_new.norm(), TOLERANCE) < TOL)
			break;

	}
	// assign h
	h = (h_new + alpha);
	h = h / h.sum();
	// calculate ll ;
	ll = 0;
	for (unsigned int id = 0; id < my_doc_count.size(); id++)
	{
		long word_id = my_doc_word[id];
		VectorXd currrow = (R.block(word_id, 0, 1, k)).transpose();//(VectorXd)R.block(0, word_id, R.rows(), 1);
		double tmp = h.dot(currrow);
		ll = ll + (my_doc_count[id]) * log(max(tmp, SMOOTH));

	}

	ll = ll + lgamma(alpha.sum()) - (eigen_gamma(alpha)).sum();
	VectorXd alphaminus1 = alpha - VectorXd::Ones(k);
	VectorXd hlog = vec_log(h);
	VectorXd alphatimesh = alphaminus1.cwiseProduct(hlog);
	ll = ll + alphatimesh.sum();


}


