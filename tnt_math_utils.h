////////////////////////////////////////////////////////////////////////////////
//                         HYPERJAMA MATH UTILITIES                           //
//                         By Alex Christensen                                //
//                         based on NIST's jama_lu.h                          //
//                         http://math.nist.gov/tnt/                          //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2012 Alex Christensen                                        //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// all copies or substantial portions of the Software.                        //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Changes:                                                                   //
//                                                                            //
// includes <complex>                                                         //
// added hyperjama_conj, _axpy, _dot, _dotc, _nrm2 functions                  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

/* needed for fabs, sqrt() below */
#include <cmath>

#include <complex>

namespace TNT
{
/**
	@returns hypotenuse of real (non-complex) scalars a and b by 
	avoiding underflow/overflow
	using (a * sqrt( 1 + (b/a) * (b/a))), rather than
	sqrt(a*a + b*b).
*/
template <class Real>
Real hypot(const Real &a, const Real &b)
{
	
	if (a== 0)
		return abs(b);
	else
	{
		Real c = b/a;
		return fabs(a) * sqrt(1 + c*c);
	}
}

#ifdef _WIN32
#pragma message("Note: Visual Studio requires these flags in the command line to parallelize and vectorize (Project>Properties>Configuration Properties>C/C++>Command Line)")
#pragma message("/openmp /arch:SSE2 /DHYPERJAMA_SSE2")
#else
#pragma message("Note: G++ requires these flags to parallelize and vectorize")
#pragma message("-fopenmp -DHYPERJAMA_SSE2")
#endif

#define HYPERJAMA_BLAS

//conjugate complex elements, do nothing to non-complex elements
template <class T> inline              T  hyperjama_conj(             T  x){return      x ;}
template <class T> inline std::complex<T> hyperjama_conj(std::complex<T> x){return conj(x);}

//saxpy, caxpy, daxpy, zaxpy template
template <class Real>
void _axpy(const Real* x, Real* y, int n, Real a){
	for(int i=0;i<n;i++)
		y[i]+=a*x[i];
}

//sdot, cdot, ddot, zdot template
template <class T>
T _dot(int n, const T* x, const T* y){
	T sum(0);
	for(int i=0;i<n;i++)
		sum+=x[i]*y[i];
	return sum;
}

//cdotc, zdotc template (sdot,ddot also)
template <class T>
inline T _dotc(int n, const T* x, const T* y){
	T sum(0);
	for(int i=0;i<n;i++)
		sum+=hyperjama_conj(x[i])*y[i];
	return sum;
}

//snrm2, dnrm2, scnrm2, dznrm2 template
//
//this is unoptimized, but not used in the innermost loops
template <class T>
inline T _nrm2(const T* x, int n){
	T norm(0);
	for(int i=0;i<n;i++){
		if(x[i]!=T(0)){
			T c(abs(norm)/abs(x[i]));
			norm=abs(x[i])*sqrt(1+abs(c)*abs(c));
		}
	}
	return norm;
}

//overloaded functions that use sse and sse2 intrinsic functions
//optimized for processors with hyperthreading, which benefit from
//grouped read and operation commands
#ifdef HYPERJAMA_SSE2
#include <emmintrin.h>

//saxpy
inline void _axpy(const float* x, float* y, int n, float a){

	if(((size_t)x)%16!=((size_t)y)%16||((size_t)x)%sizeof(float)!=0)
		for(int i=0;i<n;i++)
			y[i]+=a*x[i];
	else{
		//process unaligned portions
		while(((size_t)x)%16){
			if(n<=0)
				return;
			y[0]+=a*x[0];
			x++;
			y++;
			n--;
		}

		__m128 sum0;
		__m128 sum1;
		__m128 reg0,reg1,reg2,reg3;
		__m128 areg=_mm_set1_ps(a);
		__m128 prod;

		//add floats 8 at a time
		while(n>=8){
			//read floats into MMX registers (8 from each array)
			reg0=_mm_load_ps(x  );
			reg1=_mm_load_ps(x+4);
			reg2=_mm_load_ps(y  );
			reg3=_mm_load_ps(y+4);

			//add floats
			prod=_mm_mul_ps(reg0,areg);
			sum0=_mm_add_ps(prod,reg2);
			prod=_mm_mul_ps(reg1,areg);
			sum1=_mm_add_ps(prod,reg3);

			//put float sums into y
			_mm_store_ps(y  ,sum0);
			_mm_store_ps(y+4,sum1);

			x+=8;
			y+=8;
			n-=8;
		}

		//add beyond the last multiple of 8
		for(int i=0;i<n;i++)
			y[i]+=a*x[i];
	}
}

//daxpy
inline void _axpy(const double* x, double* y, int n, double a){

	if(((size_t)x)%16!=((size_t)y)%16||((size_t)x)%sizeof(double)!=0)
		for(int i=0;i<n;i++)
			y[i]+=a*x[i];
	else{
		//process unaligned portions
		while(((size_t)x)%16){
			if(n<=0)
				return;
			y[0]+=a*x[0];
			x++;
			y++;
			n--;
		}

		__m128d sum0;
		__m128d sum1;
		__m128d reg0,reg1,reg2,reg3;
		__m128d areg=_mm_set1_pd(a);
		__m128d prod;

		//add doubles 4 at a time
		while(n>=8){
			//read floats into MMX registers (4 from each array)
			reg0=_mm_load_pd(x  );
			reg1=_mm_load_pd(x+2);
			reg2=_mm_load_pd(y  );
			reg3=_mm_load_pd(y+2);

			//add floats
			prod=_mm_mul_pd(reg0,areg);
			sum0=_mm_add_pd(prod,reg2);
			prod=_mm_mul_pd(reg1,areg);
			sum1=_mm_add_pd(prod,reg3);

			//put float sums into y
			_mm_store_pd(y  ,sum0);
			_mm_store_pd(y+2,sum1);

			x+=4;
			y+=4;
			n-=4;
		}

		//add beyond the last multiple of 4
		for(int i=0;i<n;i++)
			y[i]+=a*x[i];
	}
}

//sdot
inline float _dot(int n, const float* X, const float* Y){

	//if one is aligned and one unaligned, perform the non-SSE code
	if(((size_t)X)%16!=((size_t)Y)%16||((size_t)X)%sizeof(float)!=0){
		float sum=0;
		for(int i=0;i<n;i++)
			sum+=X[i]*Y[i];
		return sum;
	}
	else{

		//to add before aligned sections and after the last multiple of 8
		float sum=0;

		//process unaligned section of array
		while(((size_t)X)%16){
			if(n<=0)
				return sum;
			sum+=X[0]*Y[0];
			Y++;
			X++;
			n--;
		}

		//find aligned memory on the stack to put the sums
		float sums[8];
		float* pSums=sums;
		if(((size_t)pSums)%16!=0)
			pSums=(float*)((((size_t)pSums)&(~15))+16);
		pSums[0]=0;
		pSums[1]=0;
		pSums[2]=0;
		pSums[3]=0;

		__m128 sum0=_mm_setzero_ps();
		__m128 sum1=_mm_setzero_ps();
		__m128 reg0,reg1,reg2,reg3,reg4,reg5;

		//add floats 8 at a time
		while(n>=8){

			//read floats into MMX registers (8 from each array)
			reg0=_mm_load_ps(X  );
			reg1=_mm_load_ps(X+4);
			reg2=_mm_load_ps(Y  );
			reg3=_mm_load_ps(Y+4);

			//multiply floats together
			reg4=_mm_mul_ps(reg0,reg2);
			reg5=_mm_mul_ps(reg1,reg3);

			//add to sums
			sum0=_mm_add_ps(sum0,reg4);
			sum1=_mm_add_ps(sum1,reg5);

			X+=8;
			Y+=8;
			n-=8;
		}

		//move the sums from the xmm registers to the stack
		sum0=_mm_add_ps(sum0,sum1);
		_mm_store_ps(pSums,sum0);

		//add beyond where the inner loop stopped
		for(int i=0;i<n;i++)
			sum+=X[i]*Y[i];

		return sum+pSums[0]+pSums[1]+pSums[2]+pSums[3];
	}
}

//ddot
inline double _dot(int n, const double* X, const double* Y){

	//if one is aligned and one unaligned, perform the non-SSE2 code
	if(((size_t)X)%16!=((size_t)Y)%16||((size_t)X)%sizeof(double)!=0){
		double sum=0;
		for(int i=0;i<n;i++)
			sum+=X[i]*Y[i];
		return sum;
	}
	else{

		//to add before aligned sections and after the last multiple of 8
		double sum=0;

		//process unaligned section of array
		while(((size_t)X)%16){
			if(n<=0)
				return sum;
			sum+=X[0]*Y[0];
			Y++;
			X++;
			n--;
		}

		//find aligned memory on the stack to put the sums
		double sums[4];
		double* pSums=sums;
		if(((size_t)pSums)%16!=0)
			pSums=(double*)((((size_t)pSums)&(~15))+16);
		pSums[0]=0;
		pSums[1]=0;

		__m128d sum0=_mm_setzero_pd();
		__m128d sum1=_mm_setzero_pd();
		__m128d reg0,reg1,reg2,reg3,reg4,reg5;

		//add doubles 4 at a time
		while(n>=4){

			//read doubles into MMX registers (4 from each array)
			reg0=_mm_load_pd(X  );
			reg1=_mm_load_pd(X+2);
			reg2=_mm_load_pd(Y  );
			reg3=_mm_load_pd(Y+2);

			//multiply doubles together
			reg4=_mm_mul_pd(reg0,reg2);
			reg5=_mm_mul_pd(reg1,reg3);

			//add to sums
			sum0=_mm_add_pd(sum0,reg4);
			sum1=_mm_add_pd(sum1,reg5);

			X+=4;
			Y+=4;
			n-=4;
		}

		//move the sums from the xmm registers to the stack
		sum0=_mm_add_pd(sum0,sum1);
		_mm_store_pd(pSums,sum0);

		//add beyond where the inner loop stopped
		for(int i=0;i<n;i++)
			sum+=X[i]*Y[i];

		return sum+pSums[0]+pSums[1];
	}
}

//conjugated dot products are the same as non-conjugated dot products for real numbers
inline float  _dotc(int n, const float  *x, const float  *y){return _dot(n,x,y);}
inline double _dotc(int n, const double *x, const double *y){return _dot(n,x,y);}

#endif

} /* TNT namespace */



#endif
/* MATH_UTILS_H */
