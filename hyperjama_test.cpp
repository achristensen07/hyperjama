////////////////////////////////////////////////////////////////////////////////
//                         HYPERJAMA TIME AND STABILITY TEST                  //
//                         By Alex Christensen                                //
//                         uses on NIST's jama C++ headers                    //
//                         http://math.nist.gov/tnt/                          //
//                         or HyperJAMA's headers                             //
//                         http://www.hyperjama.net/                          //
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

#include <complex>
#include <iostream>

#include "jama_lu.h"
#include "jama_qr.h"
#include "jama_cholesky.h"
#include <time.h>

void matmultTest(int size,std::ostream& os){

	//initialize two random matrices
	Array2D<float > as(size,size);
	Array2D<float > bs(size,size);
	Array2D<double> ad(size,size);
	Array2D<double> bd(size,size);
	for(int r=0;r<size;r++){
		for(int c=0;c<size;c++){
			as[r][c]=((float )rand())/RAND_MAX;
			bs[r][c]=((float )rand())/RAND_MAX;
			ad[r][c]=((double)rand())/RAND_MAX;
			bd[r][c]=((double)rand())/RAND_MAX;
		}
	}

	//time the matmult operation
	clock_t t1=clock();
	Array2D<float > cs=matmult(as,bs);
	clock_t t2=clock();
	Array2D<double> cd=matmult(ad,bd);
	clock_t t3=clock();

	//check a random row and column
	int row=rand()%size;
	int col=rand()%size;
	float  sums=0;
	double sumd=0;
	for(int i=0;i<size;i++){
		sums+=as[row][i]*bs[i][col];
		sumd+=ad[row][i]*bd[i][col];
	}
	float  errors=fabs(sums-cs[row][col]);
	double errord=fabs(sumd-cd[row][col]);

	//print report
	os<<"mult\t"<<size<<"\t"<<((double)(t2-t1))/CLOCKS_PER_SEC<<"\t"<<((double)(t3-t2))/CLOCKS_PER_SEC<<"\t"<<errors<<"\t"<<errord<<std::endl;
}

void luTest(int size,std::ostream& os){

	//allocate matrices and vectors
	Array2D<float>  As(size,size);
	Array2D<double> Ad(size,size);
	Array1D<float>  bs(size);
	Array1D<double> bd(size);

	//initialize random matrices and vectors
	for(int i=0;i<size;i++){
		bs[i]=((float )rand())/RAND_MAX;
		bd[i]=((double)rand())/RAND_MAX;
		for(int j=0;j<size;j++){
			As[i][j]=((float )rand())/RAND_MAX;
			Ad[i][j]=((double)rand())/RAND_MAX;
		}
	}

	//time lu factorizations of the matrices
	clock_t t1=clock();
	JAMA::LU<float> lus(As);
	clock_t t2=clock();
	JAMA::LU<double> lud(Ad);
	clock_t t3=clock();

	//calculate the error (norm(abs(Ax-b),1))
	Array1D<float>  xs=lus.solve(bs);
	Array1D<double> xd=lud.solve(bd);
	float  errors=0;
	double errord=0;
	for(int i=0;i<size;i++){
		float  Axis=0;
		double Axid=0;
		for(int j=0;j<size;j++){
			Axis+=As[i][j]*xs[j];
			Axid+=Ad[i][j]*xd[j];
		}
		errors+=fabs(Axis-bs[i]);
		errord+=fabs(Axid-bd[i]);
	}

	//print report
	os<<"lu\t"<<size<<"\t"<<((double)(t2-t1))/CLOCKS_PER_SEC<<"\t"<<((double)(t3-t2))/CLOCKS_PER_SEC<<"\t"<<errors<<"\t"<<errord<<std::endl;
}

void choleskyTest(int size,std::ostream& os){

	//allocate matrices and vectors
	Array2D<float>  As(size,size);
	Array2D<double> Ad(size,size);
	Array1D<float>  bs(size);
	Array1D<double> bd(size);

	//initialize random spd matrices and vectors
	for(int i=0;i<size;i++){
		bs[i]=((float )rand())/RAND_MAX;
		bd[i]=((double)rand())/RAND_MAX;
		for(int j=0;j<i;j++){
			As[i][j]=((float )rand())/RAND_MAX;
			Ad[i][j]=((double)rand())/RAND_MAX;
			As[j][i]=As[i][j];
			Ad[j][i]=Ad[i][j];
		}
		As[i][i]=size+((float )rand())/RAND_MAX;
		Ad[i][i]=size+((double)rand())/RAND_MAX;
	}

	//time cholesky factorizations of the matrices
	clock_t t1=clock();
	JAMA::Cholesky<float> chols(As);
	clock_t t2=clock();
	JAMA::Cholesky<double> chold(Ad);
	clock_t t3=clock();

	//calculate the error (norm(abs(Ax-b),1))
	Array1D<float>  xs=chols.solve(bs);
	Array1D<double> xd=chold.solve(bd);
	float  errors=0;
	double errord=0;
	for(int i=0;i<size;i++){
		float  Axis=0;
		double Axid=0;
		for(int j=0;j<size;j++){
			Axis+=As[i][j]*xs[j];
			Axid+=Ad[i][j]*xd[j];
		}
		errors+=fabs(Axis-bs[i]);
		errord+=fabs(Axid-bd[i]);
	}

	//print report
	os<<"chol\t"<<size<<"\t"<<((double)(t2-t1))/CLOCKS_PER_SEC<<"\t"<<((double)(t3-t2))/CLOCKS_PER_SEC<<"\t"<<errors<<"\t"<<errord<<std::endl;
}

void qrTest(int size,std::ostream& os){

	//allocate matrices and vectors
	Array2D<float>  As(size,size);
	Array2D<double> Ad(size,size);
	Array1D<float>  bs(size);
	Array1D<double> bd(size);

	//initialize random matrices and vectors
	for(int i=0;i<size;i++){
		bs[i]=((float )rand())/RAND_MAX;
		bd[i]=((double)rand())/RAND_MAX;
		for(int j=0;j<size;j++){
			As[i][j]=((float )rand())/RAND_MAX;
			Ad[i][j]=((double)rand())/RAND_MAX;
		}
	}

	//time qr factorizations of the matrices
	clock_t t1=clock();
	JAMA::QR<float> qrs(As);
	clock_t t2=clock();
	JAMA::QR<double> qrd(Ad);
	clock_t t3=clock();

	//calculate the error (norm(abs(Ax-b),1))
	Array1D<float>  xs=qrs.solve(bs);
	Array1D<double> xd=qrd.solve(bd);
	float  errors=0;
	double errord=0;
	for(int i=0;i<size;i++){
		float  Axis=0;
		double Axid=0;
		for(int j=0;j<size;j++){
			Axis+=As[i][j]*xs[j];
			Axid+=Ad[i][j]*xd[j];
		}
		errors+=fabs(Axis-bs[i]);
		errord+=fabs(Axid-bd[i]);
	}

	//print report
	os<<"qr\t"<<size<<"\t"<<((double)(t2-t1))/CLOCKS_PER_SEC<<"\t"<<((double)(t3-t2))/CLOCKS_PER_SEC<<"\t"<<errors<<"\t"<<errord<<std::endl;
}

//functions for testing complex matrices
#ifdef HYPERJAMA_BLAS
template <class T>
Array2D<T> conjugateTranspose(const Array2D<T>& A)
{
	Array2D<T> At(A.dim2(),A.dim1());
	for(int i=0;i<A.dim1();i++)
		for(int j=0;j<A.dim2();j++)
			At[j][i]=hyperjama_conj(A[i][j]);
	return At;
}

template <class T>
bool isSmall(const Array2D<T>& A)
{
	for(int i=0;i<A.dim1();i++)
		for(int j=0;j<A.dim2();j++)
			if(abs(A[i][j])>0.001)
				return false;
	return true;
}
template <class T>
bool isSmall(const Array1D<T>& A)
{
	for(int i=0;i<A.dim1();i++)
		if(abs(A[i])>0.001)
			return false;
	return true;
}
#endif

template <class T>
Array1D<T> matVecMultiply(const Array2D<T>& A,const Array1D<T>& x)
{
	assert(A.dim2()==x.dim1());

	Array1D<T> b(A.dim1());

	for(int i=0;i<A.dim1();i++){
		T sum(0);
		for(int j=0;j<A.dim2();j++)
			sum+=A[i][j]*x[j];
	}

	return b;
}

void complexTest(int size,std::ostream& os)
{
#ifdef HYPERJAMA_BLAS
	Array2D<std::complex<double> >Az(size,size);
	Array2D<std::complex<double> >Bz(size,size);
	Array1D<std::complex<double> >bz(size);

	//make a hermitian positive definite matrix Az (to work with cholesky)
	for(int i=0;i<size;i++){
		for(int j=i+1;j<size;j++){
			Az[i][j]=std::complex<double>(((double)rand())/RAND_MAX,((double)rand())/RAND_MAX);
			Az[j][i]=hyperjama_conj(Az[i][j]);
		}
		Az[i][i]=std::complex<double>(size*2+((double)rand())/RAND_MAX,0);
	}

	//make random b vectors and matrices
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++)
			Bz[i][j]=std::complex<double>(((double)rand())/RAND_MAX,((double)rand())/RAND_MAX);
		bz[i]=std::complex<double>(((double)rand())/RAND_MAX,((double)rand())/RAND_MAX);
	}

	JAMA::LU      <std::complex<double> >LUz(Az);
	JAMA::Cholesky<std::complex<double> >Chz(Az);
	JAMA::QR      <std::complex<double> >QRz(Az);

	Array1D<std::complex<double> >xz1=LUz.solve(bz);
	Array1D<std::complex<double> >xz2=QRz.solve(bz);
	Array1D<std::complex<double> >xz3=Chz.solve(bz);
	Array2D<std::complex<double> >xz4=LUz.solve(Bz);
	Array2D<std::complex<double> >xz5=QRz.solve(Bz);
	Array2D<std::complex<double> >xz6=Chz.solve(Bz);

	if(!isSmall(TNT::matmult(LUz.getL(),LUz.getU())-Az))
		os<<"complex failed1"<<std::endl;
	else if(!isSmall(TNT::matmult(Chz.getL(),conjugateTranspose(Chz.getL()))-Az))
		os<<"complex failed2"<<std::endl;
	else if(!isSmall(TNT::matmult(QRz.getQ(),QRz.getR())-Az))
		os<<"complex failed3"<<std::endl;
	else if(!isSmall(matVecMultiply(Az,xz3)))
		os<<"complex failed4"<<std::endl;
	else if(!isSmall(TNT::matmult(Az,xz4)-Bz))
		os<<"complex failed5"<<std::endl;
	else if(!isSmall(TNT::matmult(Az,xz5)-Bz))
		os<<"complex failed6"<<TNT::matmult(Az,xz5)-Bz<<std::endl;
	else if(!isSmall(TNT::matmult(Az,xz6)-Bz))
		os<<"complex failed7"<<std::endl;
	else if(!isSmall(xz1-xz2)||
			!isSmall(xz3-xz1)||
			!isSmall(xz5-xz4)||
			!isSmall(xz6-xz4))
		os<<"complex failed8"<<std::endl;
	else
#endif
		os<<"complex success"<<std::endl;
}

int main(int argc, char *argv[]){

	srand((unsigned int)time(NULL));

	//print header
	std::cout<<"test\tsize\tfloat time\tdouble time\tfloat error\tdouble error"<<std::endl;

	//perform tests
	for(double size=4;size<=1024;size*=1.111){
		choleskyTest((int)size,std::cout);
		luTest      ((int)size,std::cout);
		qrTest      ((int)size,std::cout);
		matmultTest ((int)size,std::cout);
	}

	//check complex accuracy (if using hyperjama headers)
	complexTest(30,std::cout);
	return 0;
}
