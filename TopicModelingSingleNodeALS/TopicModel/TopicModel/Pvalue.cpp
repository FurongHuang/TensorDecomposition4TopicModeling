//
//  Pvalue.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#include "stdafx.h"

using namespace Eigen;
using namespace std;

// Furong's function for calculating p-values
double CalculateMean(double value[], long len) // calculating sample mean of an array of values
{
	double sum = 0;
	for (long i = 0; i<len; i++)
	{
		sum += value[i];
	}
	return (double)(sum / len);
}

double CalculateVariance(double value[], long len) // calculating variance of an array of values
{
	double mean = CalculateMean(value, len);
	double temp = 0;
	for (long i = 0; i < len; i++)
	{
		temp += (value[i] - mean)*(value[i] - mean);
	}
	return (double)(temp / len);
}

double CalculateSampleVariance(double value[], long len) // calculating sample variance of an array of values
{
	double mean = CalculateMean(&*value, len);
	double temp = 0;
	for (long i = 0; i < len; i++)
	{
		temp += (value[i] - mean)*(value[i] - mean);
	}
	return (double)(temp / (len - 1));

}

double Calculate_StandardDeviation(double value[], long len) // calculating standard deviation of an array of values
{
	return sqrt(CalculateVariance(&*value, len));
}

double Calculate_SampleStandardDeviation(double value[], long len) // calculating sample standard deviation of an array of values
{
	return sqrt(CalculateSampleVariance(&*value, len));
}

double Calculate_Covariance(double x[], double y[], long len) // calculating cross-covariance of two arrays of values
{
	double x_mean = CalculateMean(x, len);
	double y_mean = CalculateMean(y, len);
	double summation = 0;
	for (long i = 0; i<len; i++)
	{
		summation += (x[i] - x_mean)*(y[i] - y_mean);
	}
	return (double)(summation / len);
}

double Calculate_Correlation(double x[], double y[], long len) // calculating correlation coefficient between two arrays of values
{
	double covariance = Calculate_Covariance(&*x, &*y, len);
	double correlation = covariance / (Calculate_StandardDeviation(&*x, len)*Calculate_StandardDeviation(&*y, len));
	return (correlation);
}

double Calculate_Tstat(double x[], double y[], long len) // calculating t-statistic for hypothesis testing
{
	double r = Calculate_Correlation(&*x, &*y, len);
	double t = r * sqrt((len - 2) / (1 - r*r));
	return t;
}

// FUNCTION betacf(a,b,x)
double betacf(double a, double b, double x)
{
	double betacfvalue;
	int MAXIT = 100000;
	double EPS_thisfunc = 3e-7;
	double FPMIN = 1e-30;
	int m, m2;
	double aa, c, d, del, h, qab, qam, qap;
	qab = a + b;
	qap = a + 1.0;
	qam = a - 1.0;
	c = 1.0;
	d = 1.0 - qab*x / qap;
	if (fabs(d)<FPMIN)
	{
		d = FPMIN;
	}

	d = 1.0 / d;
	h = d;
	m = 0;
	do{
		m += 1;
		m2 = 2 * m;
		aa = m*(b - m)*x / ((qam + m2)*(a + m2));
		d = 1.0 + aa*d;
		if (fabs(d) < FPMIN)
		{
			d = FPMIN;
		}
		c = 1.0 + aa / c;
		if (fabs(c) < FPMIN)
		{
			c = FPMIN;
		}
		d = 1.0 / d;
		h = h*d*c;
		aa = -(a + m)*(qab + m)*x / ((a + m2)*(qap + m2));
		d = 1.0 + aa*d;
		if (fabs(d)<FPMIN)
		{
			d = FPMIN;
		}
		c = 1.0 + aa / c;
		if (fabs(c)<FPMIN)
		{
			c = FPMIN;

		}
		d = 1.0 / d;
		del = d*c;
		h = h*del;
		double theta = fabs(del - 1.0);
		if (theta<EPS_thisfunc || theta == EPS_thisfunc)
		{
			goto STOP;
		}
	} while (m <= MAXIT);

	printf("a or b too big or MAXIT too small in betacf\n");
	printf("Redo: \n");
	system("pause");
	betacfvalue = betacf(a, b, x);
STOP: betacfvalue = h;
	return betacfvalue;
}

double gammln(double xx)
{
	double x, tmp, ser;
	static double cof[6] = { 76.18009173, -86.50532033, 24.01409822,
		-1.231739516, 0.120858003e-2, -0.536382e-5 };
	int j;

	x = xx - 1.0;
	tmp = x + 5.5;
	tmp -= (x + 0.5)*log(tmp);
	ser = 1.0;
	for (j = 0; j <= 5; j++) {
		x += 1.0;
		ser += cof[j] / x;
	}
	return -tmp + log(2.50662827465*ser);
}

double betainc(double a, double b, double x)
{
	double betaivalue, bt;
	if (x == 0 || x > 1)
	{
//		sleep(2);
		printf("bad argument x in betainc");
	}
	if (x == 0 || x == 1)
	{
		bt = 0;
	}
	else
	{
		bt = exp(gammln(a + b) - gammln(a) - gammln(b) + a*log(x) + b*log(1.0 - x));
	}
	if (x<(a + 1.0) / (a + b + 2.0))
	{
		betaivalue = bt * betacf(a, b, x) / a;
		return betaivalue;
	}
	else
	{
		betaivalue = 1.0 - bt * betacf(b, a, 1.0 - x) / b;
		return betaivalue;
	}
}

double Calculate_Pvalue(double x[], double y[], long len) // calculating p values
{
	double p;
	double t = -fabs(Calculate_Tstat(&*x, &*y, len));
	double n = (double)(len - 2);
	double normcutoff = 1e7;
	// Initialize P.
	p = NAN;
	bool nans = (isnan(t) || n <= 0);
	if (n == 1)
	{
		p = 0.5 + atan(t) / M_PI;
		//return (2*p);
	}
	if (n > normcutoff)
	{
		p = 0.5 * erfc(-t / sqrt(2.0));
		//return (2*p);
	}
	if (n != 1 && n <= normcutoff && !nans)
	{
		double    temptemp = n / (n + t*t);
		p = 0.5* betainc(0.5*n, 0.5, temptemp);
		if (t >0)
		{
			p = 1 - p;
			//return (2*p);
		}

	}
	else
	{
		p = 0.5;
	}
	return (2 * p);
}