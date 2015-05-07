// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifndef __TopicModel__stdafx__
#define __TopicModel__stdafx__
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <list>
#include <vector>
#include <set>
#include <queue>
#include <map>

#include <iterator>

#include <math.h>
#include <algorithm>

#include "../../dependency/Eigen/Dense"
#include "../../dependency/Eigen/Sparse"
#include "../../dependency/Eigen/Core"
#include "../../dependency/Eigen/OrderingMethods"
#include "../../dependency/Eigen/SparseQR"

#include <ctime>
#include <sys/time.h>   // this is for linux machine
//#include <windows.h> // this is for windows machine



//reference additional headers your program requires here
#include "IOfn.h"
#include "Util.h"

#include "Probability.h"
#include "Spectral.h"
#include "Pvalue.h"



#define LEARNRATE 1e-9
#define MINITER 100
#define MAXITER 500
#define TOLERANCE 1e-6
#define NORMALIZE
#define ORTH_COST_WEIGHT 1000
#endif




//#define LEARNRATE 1e-9 // learning rate for tensor decomposition
//#define MINITER 10000 // minimum number of iterations
//#define MAXITER 100000 // maximum number of iterations
//#define PVALUE_TOLE 0.01 // pvalue tolerance
// #define KTRUE 6003//159//6003




