////////////////////////////////////////////////////////////////////////////////
//                         HYPERJAMA QR DECOMPOSITION                         //
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
// include <omp.h> and <cmath>                                                //
// constructor optimized                                                      //
// changed constants to template constructors in getHouseholder, getR, getQ,  //
//   isFullRank, and solve functions                                          //
// added hyperjama_conj to getQ and solve functions to work with std::complex //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef JAMA_QR_H
#define JAMA_QR_H

#include "tnt_array1d.h"
#include "tnt_array2d.h"
#include "tnt_math_utils.h"

#include <omp.h>
#include <cmath>

namespace JAMA
{

/** 
<p>
	Classical QR Decompisition:
   for an m-by-n matrix A with m >= n, the QR decomposition is an m-by-n
   orthogonal matrix Q and an n-by-n upper triangular matrix R so that
   A = Q*R.
<P>
   The QR decompostion always exists, even if the matrix does not have
   full rank, so the constructor will never fail.  The primary use of the
   QR decomposition is in the least squares solution of nonsquare systems
   of simultaneous linear equations.  This will fail if isFullRank()
   returns 0 (false).

<p>
	The Q and R factors can be retrived via the getQ() and getR()
	methods. Furthermore, a solve() method is provided to find the
	least squares solution of Ax=b using the QR factors.  

   <p>
	(Adapted from JAMA, a Java Matrix Library, developed by jointly 
	by the Mathworks and NIST; see  http://math.nist.gov/javanumerics/jama).
*/

template <class Real>
class QR {


   /** Array for internal storage of decomposition.
   @serial internal array storage.
   */
   
   TNT::Array2D<Real> QR_;

   /** Row and column dimensions.
   @serial column dimension.
   @serial row dimension.
   */
   int m, n;

   /** Array for internal storage of diagonal of R.
   @serial diagonal of R.
   */
   TNT::Array1D<Real> Rdiag;


public:
	
/**
	Create a QR factorization object for A.

	@param A rectangular (m>=n) matrix.
*/
	QR(const TNT::Array2D<Real> &A)		/* constructor */
	{
  		m = A.dim1();
		n = A.dim2();

		//copy the transpose of the input matrix for better cache alignment
		Array2D<Real> QRT(n,m);
		for(int row=0;row<m;row++)
			for(int col=0;col<n;col++)
				QRT[col][row]=A[row][col];

		Rdiag = TNT::Array1D<Real>(n);

		// Main loop.

		const int cpuCacheLineSize=64;
		const int maxBlockSize=cpuCacheLineSize/sizeof(Real);
		Real* norms=new Real[maxBlockSize];

		int blockSize;
		for (int k = 0; k < n; k+=blockSize) {
			blockSize=min(n-k,maxBlockSize);

			for(int bi=0;bi<blockSize;bi++){

				// Compute 2-norm of k+bi-th row of the transpose to the right of the diagonal
				norms[bi]=_nrm2(QRT[k+bi]+k+bi,m-k-bi);

				if(norms[bi]!=Real(0)){

					//pick the better of two reflectors, to 1 or -1, whichever is closer
					//this is wierd syntax that also works with std::complex
					if(abs(QRT[k+bi][k+bi]-Real(1))>abs(QRT[k+bi][k+bi]+Real(1)))
						norms[bi]*=-1;
					for(int i=k+bi;i<m;i++)
						QRT[k+bi][i]/=norms[bi];
					QRT[k+bi][k+bi]+=Real(1);

					// Apply transformation to remaining rows of the transpose within the block
					for (int j = k+bi+1; j < k+blockSize; j++) {
					   Real s=_dotc(m-k-bi,QRT[k+bi]+k+bi,QRT[j]+k+bi);
					   s = -s/QRT[k+bi][k+bi];
					   _axpy(QRT[k+bi]+k+bi,QRT[j]+k+bi,m-k-bi,s);
					}
				}
				Rdiag[k+bi] = -norms[bi];
			}

			//apply transformations from block to remaining rows of the transpose below the block in parallel
#pragma omp parallel for
			for (int j = k+blockSize; j < n; j++) {
				for(int bi=0;bi<blockSize;bi++){
					if(norms[bi]!=Real(0)){
						Real s=_dotc(m-k-bi,QRT[k+bi]+k+bi,QRT[j]+k+bi);
						s = -s/QRT[k+bi][k+bi];
						_axpy(QRT[k+bi]+k+bi,QRT[j]+k+bi,m-k-bi,s);
					}
				}
			}
		}
		delete [] norms;

		//calculate the transpose again
		QR_=Array2D<Real>(m,n);
		for(int row=0;row<m;row++)
			for(int col=0;col<n;col++)
				QR_[row][col]=QRT[col][row];
   }


/**
	Flag to denote the matrix is of full rank.

	@return 1 if matrix is full rank, 0 otherwise.
*/
	int isFullRank() const		
	{
      for (int j = 0; j < n; j++) 
	  {
         if (Rdiag[j] == Real(0))
            return 0;
      }
      return 1;
	}
	
	


   /** 
   
   Retreive the Householder vectors from QR factorization
   @returns lower trapezoidal matrix whose columns define the reflections
   */

   TNT::Array2D<Real> getHouseholder (void)  const
   {
   	  TNT::Array2D<Real> H(m,n);

	  /* note: H is completely filled in by algorithm, so
	     initializaiton of H is not necessary.
	  */
      for (int i = 0; i < m; i++) 
	  {
         for (int j = 0; j < n; j++) 
		 {
            if (i >= j) {
               H[i][j] = QR_[i][j];
            } else {
               H[i][j] = Real(0);
            }
         }
      }
	  return H;
   }



   /** Return the upper triangular factor, R, of the QR factorization
   @return     R
   */

	TNT::Array2D<Real> getR() const
	{
      TNT::Array2D<Real> R(n,n);
      for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
            if (i < j) {
               R[i][j] = QR_[i][j];
            } else if (i == j) {
               R[i][j] = Rdiag[i];
            } else {
               R[i][j] = Real(0);
            }
         }
      }
	  return R;
   }
	
	



   /** 
   	Generate and return the (economy-sized) orthogonal factor
   @param     Q the (ecnomy-sized) orthogonal factor (Q*R=A).
   */

	TNT::Array2D<Real> getQ() const
	{
	  int i=0, j=0, k=0;

	  TNT::Array2D<Real> Q(m,n);
      for (k = n-1; k >= 0; k--) {
         for (i = 0; i < m; i++) {
            Q[i][k] = Real(0);
         }
         Q[k][k] = Real(1);
         for (j = k; j < n; j++) {
            if (QR_[k][k] != Real(0)) {
               Real s(0);
               for (i = k; i < m; i++) {
                  s += QR_[i][k]*hyperjama_conj(Q[i][j]);
               }
               s = hyperjama_conj(-s/QR_[k][k]);
               for (i = k; i < m; i++) {
                  Q[i][j] += s*QR_[i][k];
               }
            }
         }
      }
	  return Q;
   }


   /** Least squares solution of A*x = b
   @param B     m-length array (vector).
   @return x    n-length array (vector) that minimizes the two norm of Q*R*X-B.
   		If B is non-conformant, or if QR.isFullRank() is false,
						the routine returns a null (0-length) vector.
   */

   TNT::Array1D<Real> solve(const TNT::Array1D<Real> &b) const
   {
   	  if (b.dim1() != m)		/* arrays must be conformant */
	  	return TNT::Array1D<Real>();

	  if ( !isFullRank() )		/* matrix is rank deficient */
	  {
	  	return TNT::Array1D<Real>();
	  }

	  TNT::Array1D<Real> x = b.copy();

      // Compute Y = conjugate transpose(Q)*b
      for (int k = 0; k < n; k++) 
	  {
            Real s(0); 
            for (int i = k; i < m; i++) 
			{
               s += QR_[i][k]*hyperjama_conj(x[i]);
            }
            s = hyperjama_conj(-s/QR_[k][k]);
            for (int i = k; i < m; i++) 
			{
               x[i] += s*QR_[i][k];
            }
      }
      // Solve R*X = Y;
      for (int k = n-1; k >= 0; k--) 
	  {
         x[k] /= Rdiag[k];
         for (int i = 0; i < k; i++) {
               x[i] -= x[k]*QR_[i][k];
         }
      }


	  /* return n x nx portion of X */
	  TNT::Array1D<Real> x_(n);
	  for (int i=0; i<n; i++)
			x_[i] = x[i];

	  return x_;
   }

   /** Least squares solution of A*X = B
   @param B     m x k Array (must conform).
   @return X     n x k Array that minimizes the two norm of Q*R*X-B. If
   						B is non-conformant, or if QR.isFullRank() is false,
						the routine returns a null (0x0) array.
   */

   TNT::Array2D<Real> solve(const TNT::Array2D<Real> &B) const
   {
   	  if (B.dim1() != m)		/* arrays must be conformant */
	  	return TNT::Array2D<Real>(0,0);

	  if ( !isFullRank() )		/* matrix is rank deficient */
	  {
	  	return TNT::Array2D<Real>(0,0);
	  }

      int nx = B.dim2(); 
	  TNT::Array2D<Real> X = B.copy();
	  int i=0, j=0, k=0;

      // Compute Y = conjugate transpose(Q)*B
      for (k = 0; k < n; k++) {
         for (j = 0; j < nx; j++) {
            Real s(0); 
            for (i = k; i < m; i++) {
               s += QR_[i][k]*hyperjama_conj(X[i][j]);
            }
            s = hyperjama_conj(-s/QR_[k][k]);
            for (i = k; i < m; i++) {
               X[i][j] += s*QR_[i][k];
            }
         }
      }
      // Solve R*X = Y;
      for (k = n-1; k >= 0; k--) {
         for (j = 0; j < nx; j++) {
            X[k][j] /= Rdiag[k];
         }
         for (i = 0; i < k; i++) {
            for (j = 0; j < nx; j++) {
               X[i][j] -= X[k][j]*QR_[i][k];
            }
         }
      }


	  /* return n x nx portion of X */
	  TNT::Array2D<Real> X_(n,nx);
	  for (i=0; i<n; i++)
	  	for (j=0; j<nx; j++)
			X_[i][j] = X[i][j];

	  return X_;
   }


};


}
// namespace JAMA

#endif
// JAMA_QR__H

