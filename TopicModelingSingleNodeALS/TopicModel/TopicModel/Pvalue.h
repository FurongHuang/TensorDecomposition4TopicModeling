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
#pragma once
#include "stdafx.h"

using namespace Eigen;
using namespace std;
#ifndef __TopicModel__Pvalue__
#define __TopicModel__Pvalue__
#include "stdafx.h"
double CalculateMean(double value[], long len);
double CalculateVariance(double value[], long len);
double CalculateSampleVariance(double value[], long len);
double Calculate_StandardDeviation(double value[], long len);
double Calculate_SampleStandardDeviation(double value[], long len);
double Calculate_Covariance(double x[], double y[], long len);
double Calculate_Correlation(double x[], double y[], long len);
double Calculate_Tstat(double x[], double y[], long len);
double betacf(double a, double b, double x);
double gammln(double xx);
double betainc(double a, double b, double x);
double Calculate_Pvalue(double x[], double y[], long len);


#endif