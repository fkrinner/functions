// This file contains a number of definitions for amplitude parameterizations. 
// Header only, because of templates.
#ifndef BREITWIGNERS_CILLY_BO
#define BREITWIGNERS_CILLY_BO

#include <cmath> 
#include <vector>
#include <complex>
#include <iostream>
#include <string>
// double mPi=1.3957018;///\pm0.00035MeV // Particle Data Booklet 2012

#ifdef ADOL_ON // Some function on std::complex<adouble>, needed for automatic differentiation.
#include <adolc/adolc.h>  
std::complex<adouble> log(std::complex<adouble> z){
	adouble re = std::real(z);
	adouble im = std::imag(z);

	adouble newReal = pow(re*re+im+im,.5);
	adouble newImag = atan2(im,re);

	return std::complex<adouble>(newReal,newImag);
};
std::complex<adouble> sqrt(std::complex<adouble> z){
	return pow(z,0.5);
};
adouble abs(std::complex<adouble> z){
	adouble squared = std::real(z*std::conj(z));
	return pow(squared,0.5);
};
adouble sqrt(adouble x){
	return pow(x,0.5);
};
#endif//ADOL_ON

const double PION_MASS 	= 0.139;
const double PI  	= 3.141592653589793238463;

//////////////////////////  SOME COMMON DEFINITIONS  //////////////////////////////////////////////////////////////////////
template< typename xdouble> xdouble breakupMomentumReal(xdouble M2, xdouble m12, xdouble m22){ // Real breakup momentum: sqrt(lambda(M,m1,m2))/(2M) or 0.
	xdouble lambda= M2*M2 + m12*m12 + m22*m22 - 2*M2*m12 -2*M2*m22 - 2*m12*m22;
	if ( lambda >= 0. ){
		return sqrt(lambda)/(2*M2);
	}else{
		std::cerr << "breitWigners.h: Error: Found 0 > q^2("<<M2<<","<<m12<<","<<m22<<") = "<<lambda/(4*M2*M2)<<". Sub-threshold decay."<<std::endl;
		return 0.;
	};
};

template<typename xdouble> xdouble barrierFactor(xdouble q, int L){
	double pr = 0.1973;
	xdouble z=q*q/pr/pr;
	xdouble res;
	if (L == 0){
		res=1.;
	}else if (L==1){
		res=sqrt(2*z/(z+1));
	}else if (L==2){
		res=sqrt(13*z*z/((z-3)*(z-3)+9*z));
	}else if (L==3){
		res=sqrt(277*z*z*z/(z*(z-15)*(z-15)+9*pow(2*z-5,2)));
	}else if (L==4){
		res=sqrt(12746*pow(z,4)/(pow(z*z-45*z+105,2)+25*z*pow(2*z-21,2)));
	} else {
		std::cerr << "breitWigners.h: Error: Barrier factors not defined for L =" << L <<std::endl;
		res =0.;
	};
	return res;
};

template<typename xdouble>  // Some different definition for Barrier factors... used by Dimas program.
xdouble fdl(xdouble P, xdouble R, int L){
	xdouble X = P*R;
	if (L==0){
		return 1.;
	}else if(L==1){
		return 1. + X*X;
	}else if(L==2){
		return 9. + 3.*X*X + X*X*X*X;
	}else if(L==3){
		return 225. + 45.*X*X + 6.*X*X*X*X * X*X*X*X*X*X;	
	}else if(L==4){
		return 11025. + 1575.*X*X + 135.*X*X*X*X + 10*X*X*X*X*X*X + X*X*X*X*X*X*X*X;
	}else{
		std::cerr<<"breitWigners.h: Error: 'fdl(...)' not defined for 4 < L = "<< L << std::endl;
		return 1.;
	};
};

template<typename xdouble>
xdouble psl(xdouble m, xdouble m1, xdouble m2, xdouble R, int L){
	xdouble ampor = m1+m2;
	if (m > ampor){
		xdouble E = (m*m + m1*m1 - m2 * m2)/(2*m);
		xdouble P = pow((E*E-m1*m1)*(E*E-m1*m1),.25);
		xdouble f = fdl<xdouble>(P,R,L);
		return pow(P,2*L+1)/f;
	}else{
		return 0.;
	};
};


template<typename xdouble>
xdouble bowler_integral_table(xdouble m){
	double points[]={
0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.1313E-09, 0.8940E-07, 0.6175E-06, 0.2020E-05, 0.4786E-05, 0.9458E-05, 0.1664E-04, 0.2701E-04, 0.4131E-04, 0.6037E-04, 0.8510E-04, 0.1165E-03, 0.1558E-03, 0.2040E-03, 0.2628E-03, 0.3335E-03, 0.4179E-03, 0.5178E-03, 0.6355E-03, 0.7732E-03, 0.9337E-03, 0.1120E-02, 0.1335E-02, 0.1583E-02, 0.1867E-02, 0.2194E-02, 0.2568E-02, 0.2995E-02, 0.3483E-02, 0.4039E-02, 0.4673E-02, 0.5396E-02, 0.6220E-02, 0.7160E-02, 0.8233E-02, 0.9458E-02, 0.1086E-01, 0.1246E-01, 0.1430E-01, 0.1641E-01, 0.1884E-01, 0.2163E-01, 0.2484E-01, 0.2853E-01, 0.3277E-01, 0.3759E-01, 0.4306E-01, 0.4917E-01, 0.5591E-01, 0.6322E-01, 0.7100E-01, 0.7913E-01, 0.8752E-01, 0.9604E-01, 0.1046E+00, 0.1132E+00, 0.1218E+00, 0.1302E+00, 0.1386E+00, 0.1469E+00, 0.1551E+00, 0.1631E+00, 0.1711E+00, 0.1790E+00, 0.1867E+00, 0.1944E+00, 0.2020E+00, 0.2095E+00, 0.2169E+00, 0.2243E+00, 0.2315E+00, 0.2387E+00, 0.2458E+00, 0.2529E+00, 0.2599E+00, 0.2668E+00, 0.2737E+00, 0.2805E+00, 0.2873E+00, 0.2940E+00, 0.3007E+00, 0.3073E+00, 0.3138E+00, 0.3204E+00, 0.3269E+00, 0.3333E+00, 0.3397E+00, 0.3461E+00, 0.3525E+00, 0.3587E+00, 0.3650E+00, 0.3713E+00, 0.3775E+00, 0.3837E+00, 0.3898E+00, 0.3959E+00, 0.4020E+00, 0.4081E+00, 0.4141E+00, 0.4201E+00, 0.4261E+00, 0.4320E+00, 0.4380E+00, 0.4439E+00, 0.4498E+00, 0.4556E+00, 0.4615E+00, 0.4673E+00, 0.4731E+00, 0.4790E+00, 0.4847E+00, 0.4905E+00, 0.4962E+00, 0.5019E+00, 0.5076E+00, 0.5134E+00, 0.5189E+00, 0.5246E+00, 0.5303E+00, 0.5359E+00, 0.5415E+00, 0.5471E+00, 0.5526E+00, 0.5582E+00, 0.5638E+00, 0.5693E+00, 0.5749E+00, 0.5804E+00, 0.5858E+00, 0.5914E+00, 0.5968E+00, 0.6023E+00, 0.6077E+00, 0.6132E+00, 0.6186E+00, 0.6241E+00, 0.6294E+00, 0.6348E+00, 0.6403E+00, 0.6456E+00, 0.6510E+00, 0.6563E+00, 0.6617E+00, 0.6671E+00, 0.6724E+00, 0.6777E+00, 0.6830E+00, 0.6882E+00, 0.6936E+00, 0.6990E+00, 0.7041E+00, 0.7095E+00, 0.7149E+00, 0.7199E+00, 0.7252E+00, 0.7305E+00, 0.7356E+00, 0.7410E+00, 0.7462E+00, 0.7514E+00, 0.7567E+00, 0.7619E+00, 0.7668E+00, 0.7723E+00, 0.7774E+00, 0.7826E+00, 0.7878E+00, 0.7930E+00, 0.7982E+00, 0.8033E+00, 0.8084E+00, 0.8135E+00, 0.8188E+00, 0.8239E+00, 0.8291E+00, 0.8340E+00, 0.8393E+00, 0.8444E+00, 0.8493E+00, 0.8547E+00, 0.8597E+00, 0.8649E+00, 0.8700E+00, 0.8750E+00, 0.8800E+00, 0.8851E+00, 0.8903E+00, 0.8953E+00, 0.9005E+00, 0.9054E+00, 0.9105E+00, 0.9156E+00, 0.9205E+00, 0.9256E+00, 0.9308E+00, 0.9358E+00, 0.9408E+00, 0.9458E+00, 0.9507E+00, 0.9560E+00, 0.9609E+00, 0.9659E+00, 0.9711E+00, 0.9760E+00, 0.9808E+00, 0.9860E+00, 0.9909E+00, 0.9960E+00, 0.1001E+01, 0.1006E+01, 0.1011E+01, 0.1016E+01, 0.1021E+01, 0.1026E+01, 0.1031E+01, 0.1036E+01, 0.1041E+01, 0.1046E+01, 0.1051E+01, 0.1056E+01, 0.1061E+01, 0.1066E+01, 0.1071E+01, 0.1076E+01, 0.1081E+01, 0.1085E+01, 0.1090E+01, 0.1096E+01, 0.1100E+01, 0.1105E+01, 0.1110E+01, 0.1115E+01, 0.1120E+01, 0.1125E+01, 0.1130E+01, 0.1135E+01, 0.1140E+01, 0.1145E+01, 0.1150E+01, 0.1154E+01, 0.1160E+01, 0.1164E+01, 0.1169E+01, 0.1174E+01, 0.1179E+01, 0.1184E+01, 0.1189E+01, 0.1194E+01, 0.1199E+01, 0.1204E+01, 0.1208E+01, 0.1214E+01, 0.1218E+01, 0.1223E+01, 0.1228E+01, 0.1233E+01, 0.1238E+01, 0.1243E+01, 0.1248E+01, 0.1253E+01, 0.1257E+01, 0.1262E+01, 0.1267E+01, 0.1272E+01, 0.1277E+01, 0.1282E+01, 0.1287E+01, 0.1292E+01, 0.1296E+01, 0.1301E+01, 0.1306E+01, 0.1311E+01, 0.1316E+01, 0.1321E+01, 0.1326E+01, 0.1330E+01, 0.1336E+01, 0.1340E+01, 0.1345E+01, 0.1350E+01, 0.1355E+01, 0.1359E+01, 0.1365E+01, 0.1369E+01, 0.1374E+01, 0.1379E+01, 0.1384E+01, 0.1389E+01, 0.1394E+01, 0.1398E+01, 0.1404E+01, 0.1408E+01, 0.1412E+01, 0.1418E+01, 0.1422E+01, 0.1427E+01, 0.1432E+01, 0.1437E+01, 0.1442E+01, 0.1447E+01, 0.1451E+01, 0.1457E+01, 0.1461E+01, 0.1466E+01, 0.1472E+01, 0.1475E+01, 0.1480E+01, 0.1486E+01, 0.1490E+01, 0.1495E+01, 0.1500E+01, 0.1504E+01, 0.1510E+01, 0.1514E+01, 0.1518E+01, 0.1524E+01, 0.1529E+01, 0.1534E+01, 0.1538E+01, 0.1542E+01, 0.1549E+01, 0.1552E+01, 0.1557E+01, 0.1562E+01, 0.1567E+01, 0.1573E+01, 0.1577E+01, 0.1581E+01, 0.1586E+01, 0.1592E+01, 0.1595E+01, 0.1601E+01, 0.1605E+01, 0.1610E+01, 0.1616E+01, 0.1619E+01, 0.1625E+01, 0.1630E+01, 0.1634E+01, 0.1639E+01, 0.1644E+01, 0.1648E+01, 0.1654E+01, 0.1658E+01, 0.1663E+01, 0.1668E+01, 0.1672E+01, 0.1678E+01, 0.1682E+01, 0.1687E+01, 0.1692E+01, 0.1696E+01, 0.1701E+01, 0.1708E+01, 0.1710E+01, 0.1716E+01, 0.1721E+01, 0.1724E+01, 0.1726E+01};
	double xmin = 0.;
	double xmax = 4.;
	double step = 0.01;
	if (m > xmin and m < xmax){
		int nStep = 0;
		while (xmin+(nStep+1)*step < m){ // nStep is the last integer, where the mass is smaller than the input mass.
			nStep+=1;
		};
		double upper = points[nStep+1];
		double lower = points[nStep];
		double mUpper = xmin + (nStep+1)*step;
		double mLower = xmin + nStep*step;
		xdouble x = (m - mLower)/(mUpper-mLower);
		return (1-x)*lower + x*upper;
	}else{
		return 0.;
	};
};

//////////////////////////////////////  BREIT WIGNER DEFINITIONS  //////////////////////////////////////////////////////////////


template< typename xdouble> std::complex<xdouble> bw(double m, std::vector<xdouble> &param, int model, int L=0){ 	// Declare parametrization also in 'getNpars()'. Otherwise, the chi2 doesn't know 
	if (-1==model) {											// the number of parameters. It will then work with 20. Any function, with more than 20
		return std::complex<xdouble>(1.,0.);								// parameters which was not declared in 'getNpars()' will crash.
	};													// First paramters have to be given, then constants (the according numbers are 
	if (0==model) { //Simple Breit-Wigner// Reproduces Dimas program					// to be strored in in 'getNpars()' and 'getNparsNonConst()'
		xdouble m0 = param[0];
		xdouble G0 = param[1];
//		std::cout << "Params are "<< m0 << ' ' << G0 << endl;
		std::complex<xdouble> denominator = std::complex<xdouble>(m0*m0-m*m,-m0*G0);
		std::complex<xdouble> value = std::complex<xdouble>(m0*G0)/denominator;
//		std::cout<<"EVAL SIMPLE BW("<< m <<", "<< m0 <<", "<< G0 <<"): "<< value <<endl;
		return value;
	};
	if (1==model){ //Mass Dependent Breit-Wigner (one decay channel)
		xdouble m0 = param[0];
		xdouble G0 = param[1];
		xdouble mPi= param[2];
		xdouble mIso=param[3];

		xdouble q0 = breakupMomentumReal<xdouble>(m0*m0,mPi*mPi,mIso*mIso);
		xdouble q  = breakupMomentumReal<xdouble>(m* m ,mPi*mPi,mIso*mIso);
		xdouble Fl = barrierFactor<xdouble>(q,L);
		xdouble Fl0= barrierFactor<xdouble>(q0,L);

		xdouble G  = G0* m0/m * q*Fl*Fl/q0/Fl0/Fl0; //G0 * m0/m q*Fl^2/(q0*Fl0^2)
		std::complex<xdouble> denominator = std::complex<xdouble>(m0*m0-m*m,-m0*G);
		return std::complex<xdouble>(m0*G0,0.)/denominator;	
	};
	if (2==model){ //Mass dependent Breit-Wigner (two decay channels) // Copy Dimas code, do not use Stephan's slides// Reproduces Dimas program
		xdouble m0 = param[0];
		xdouble G0 = param[1];
		xdouble mPi= param[2];
		xdouble mIso1 = param[3];
		xdouble mIso2 = param[4];
		xdouble X  = param[5];

		xdouble R = 5.;

		xdouble psl1 = psl<xdouble>(m, mPi, mIso1, R, L);
		xdouble psl2 = psl<xdouble>(m, mPi, mIso2, R, L);

		xdouble psl10= psl<xdouble>(m0, mPi, mIso1, R, L);
		xdouble psl20= psl<xdouble>(m0, mPi, mIso2, R, L);

		xdouble G = G0 * m0/m * ((1-X) * psl1/psl10 + X * psl2/psl20);

		std::complex<xdouble> denominator  = std::complex<xdouble>(m0*m0-m*m,-m0*G);
		return std::complex<xdouble>(m0*G0,0)/denominator;
	};
	if (3==model){ // Vandermeulen Phase Space// Reproduces Dimas program

		xdouble alpha = param[0];
		xdouble mPi   = param[1];
		xdouble mIso  = param[2];

		xdouble ampor = mPi + mIso;
		std::complex<xdouble> value;

		if ( m > ampor){
			double S = m*m;
			xdouble E = (S + mPi * mPi - mIso*mIso)/(2*m);
			xdouble PSQ = E*E - mPi*mPi;
			value = std::complex<xdouble>(exp(alpha*PSQ),0.);
		}else{
			value = std::complex<xdouble>(1.,0.);			
		};
		return value;
	};
	if (4==model){ // Valera, Dorofeev Background
		xdouble alpha = param[0];
		xdouble beta  = param[1];
		xdouble m0 = param[2];		
		return std::complex<xdouble>(pow((m-m0)/0.5,alpha)*exp(-beta*(m-m0-0.5)),0);
	};
	if (5==model){ // Bowler parametrization// Reproduces Dimas program
		xdouble m0 = param[0];
		xdouble G0 = param[1];

		xdouble G = G0*	bowler_integral_table<double>(m)/bowler_integral_table<xdouble>(m0) * m0/m;
		std::complex<xdouble> denominator = std::complex<xdouble>(m0*m0-m*m,-m0*G);

		return std::complex<xdouble>(sqrt(m0*G0),0)/denominator;
	};
	if (6==model){ // Flatte (as in the 3 charged pion release-note from Sep2013
		xdouble m0 = param[0];
		xdouble g1 = param[1];
		xdouble g2 = param[2];
		xdouble mPi= param[3];
		xdouble mK = param[4];
		
		xdouble qpp= breakupMomentumReal<xdouble>(m*m,mPi*mPi,mPi*mPi);
		xdouble qKK= breakupMomentumReal<xdouble>(m*m,mK*mK,mK*mK);

		std::complex<xdouble> denominator = std::complex<xdouble>(m0*m0-m*m,-(g1*qpp*qpp + g2*qKK*qKK));
		return std::complex<xdouble>(1,0)/denominator;
	};
	if (9==model){	//Simple gaus for test		
		xdouble m0 = param[0];
		xdouble sig= param[1];

		return std::complex<xdouble>(exp(-(m-m0)*(m-m0)/2/sig/sig),0);

	};

	if (10==model){ //Polynomial c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4
		xdouble c0 = param[0];
		xdouble c1 = param[1];
		xdouble c2 = param[2];
		xdouble c3 = param[3];
		xdouble c4 = param[4];

		xdouble ret = c4*m*m*m*m + c3*m*m*m + c2*m*m + c1*m + c0;
		return std::complex<xdouble>(ret,0);
	};
	if (22==model){ //Mass dependent Breit-Wigner (two decay channels) // From Stephan Schmeing's slides
		xdouble m0 = param[0];
		xdouble G0 = param[1];
		xdouble mPi= param[2];
		xdouble mIso1 = param[3];
		xdouble mIso2 = param[4];
		xdouble X  = param[5];

		xdouble q1 = breakupMomentumReal<xdouble>(m*m,mPi*mPi,mIso1*mIso1);
		xdouble q10= breakupMomentumReal<xdouble>(m0*m0,mPi*mPi,mIso1*mIso1);
		xdouble q2 = breakupMomentumReal<xdouble>(m*m,mPi*mPi,mIso2*mIso2);
		xdouble q20= breakupMomentumReal<xdouble>(m0*m0,mPi*mPi,mIso2*mIso2);
		xdouble Fl1= barrierFactor<xdouble>(q1,L);
		xdouble Fl10=barrierFactor<xdouble>(q10,L);
		xdouble Fl2= barrierFactor<xdouble>(q2,L);
		xdouble Fl20=barrierFactor<xdouble>(q20,L);	

		xdouble G  = G0 * m0/m* ((1-X)*q1*Fl1*Fl1/q10/Fl10/Fl10 + X* q2*Fl2*Fl2/q20/Fl20/Fl20);
		std::complex<xdouble> denominator = std::complex<xdouble>(m0*m0-m*m,-m0*G);
		return std::complex<xdouble>(m0*G0,0.)/denominator;	
	};
	if (101==model){ //t'-dependent background // Reproduces Dimas program
		xdouble b     = param[0];
		xdouble c0    = param[1];
		xdouble c1    = param[2];
		xdouble c2    = param[3];
		xdouble m0    = param[4];
		xdouble mPi   = param[5];
		xdouble mIso  = param[6];
		xdouble tPrime= param[7];

		xdouble PSQ = 0.;
		xdouble mpor = mPi + mIso;		
		if (m > mpor){
			xdouble E = (m*m +mPi*mPi - mIso*mIso)/(2*m);
			PSQ = E*E - mPi*mPi;
		};
		return std::complex<xdouble>(pow(m-0.5,b)*exp(PSQ*(c0+c1*tPrime+c2*tPrime*tPrime)),0.);
	};
	if((model >= 510 and model <= 512) or (model >=610 and model <=612)){ // p-vector-formalism given by Michael Pennington (converted to C++)
		xdouble E = m;		// Mass
		
		xdouble m_Pi = param[0];	// Pion-mass
		xdouble m_K  = param[1];	// Kaon-mass

		xdouble Pi = 3.141592653592;	// Pi

		xdouble S1   = 0.26261;            	// Parameters which MAY NOW be touched.
		xdouble F11  = 0.38949;      
		xdouble F21  = 0.24150;          
		xdouble S2   = 1.0811;            
		xdouble F12  = 0.33961;           
		xdouble F22  =-0.78538;            
		xdouble C110 = 0.14760;           
		xdouble C111 = 0.62181E-01;       
		xdouble C112 = 0.29465E-01;      
		xdouble C120 = 0.10914;           
		xdouble C121 =-0.17912;           
		xdouble C122 = 0.10758;           
		xdouble C220 =-0.27253;           
		xdouble C221 = 0.79442;           
		xdouble C222 =-0.49529; 
		if (model >= 610 and model <= 612){
			S1   = param[0];            	// Parameters which may NOT (!!!) be touched.
			F11  = param[1];      
			F21  = param[2];          
			S2   = param[3];            
			F12  = param[4];           
			F22  = param[5];            
			C110 = param[6];           
			C111 = param[7];       
			C112 = param[8];      
			C120 = param[9];           
			C121 = param[10];           
			C122 = param[11];           
			C220 = param[12];           
			C221 = param[13];           
			C222 = param[14];   
			m_Pi = param[15];
			m_K = param[16];
		};
		xdouble S0   = 0.41000*m_Pi*m_Pi;	// Adler zero at (S-S0)
		xdouble eps  = 1E-5;

		xdouble S=E*E;
		std::complex<xdouble> S_c=std::complex<xdouble>(S,eps);
		xdouble rho1=sqrt(1-(4*m_Pi*m_Pi/S));
		std::complex<xdouble> rho1_c = sqrt(std::complex<xdouble>(1.,0.)-std::complex<xdouble>(4*m_Pi*m_Pi,0.)/S_c);
		std::complex<xdouble> rho2_c = sqrt(std::complex<xdouble>(1.,0.)-std::complex<xdouble>(4*m_K*m_K,0.)/S_c);
		std::complex<xdouble> F1_c   = rho1_c/Pi * log((rho1_c+std::complex<xdouble>(1.,0.))/(rho1_c-std::complex<xdouble>(1.,0.)));
		std::complex<xdouble> F2_c   = rho2_c/Pi * log((rho2_c+std::complex<xdouble>(1.,0.))/(rho2_c-std::complex<xdouble>(1.,0.)));

		xdouble rho2=0.;
		if (S>4*m_K*m_K){
			rho2 = abs(rho2_c);
		};

		xdouble Q2 = S/(4*m_K*m_K) - 1;
		xdouble Q4 = Q2*Q2;

		xdouble APL11 = F11*F11/((S1-S)*(S1-S0))+F12*F12/((S2-S)*(S2-S0));
		xdouble AQL11 = C110+C111*Q2+C112*Q4;
		xdouble APL12 = F11*F21/((S1-S)*(S1-S0))+F12*F22/((S2-S)*(S2-S0));
		xdouble AQL12 = C120+C121*Q2+C122*Q4;
		xdouble APL22 = F21*F21/((S1-S)*(S1-S0))+F22*F22/((S2-S)*(S2-S0));
		xdouble AQL22 = C220+C221*Q2+C222*Q4;

		xdouble ALL11 = (APL11+AQL11)*(S-S0)/(4*m_K*m_K);
		xdouble ALL12 = (APL12+AQL12)*(S-S0)/(4*m_K*m_K);
		xdouble ALL22 = (APL22+AQL22)*(S-S0)/(4*m_K*m_K);

		xdouble DET = ALL11*ALL22 - ALL12*ALL12;
		
		std::complex<xdouble> DEN_c = std::complex<xdouble>(1.,0.) + F1_c*ALL11 + F2_c*ALL22 + F1_c*F2_c*DET;

		std::complex<xdouble> T11_c = (ALL11+F2_c*DET)/DEN_c;
		std::complex<xdouble> T12_c = ALL12/DEN_c;
		std::complex<xdouble> T22_c = (ALL22+F1_c*DET)/DEN_c;

		std::complex<xdouble> T11_red_c = T11_c/std::complex<xdouble>((S-S0),0); // Remove Adler zero
		std::complex<xdouble> T12_red_c = T12_c/std::complex<xdouble>((S-S0),0); // Remove Adler zero
		std::complex<xdouble> T22_red_c = T22_c/std::complex<xdouble>((S-S0),0); // Remove Adler zero
		std::complex<xdouble>bw_c;
		if (510==model or 610==model){			// Pi Pi --> Pi Pi
			bw_c=T11_red_c;
		};
		if (511==model or 611==model){			// K  K  --> Pi Pi  (and Pi Pi --> K  K )
			bw_c=T12_red_c;
		};
		if (512==model or 612==model){			// K  K   --> K  K  
			bw_c=T22_red_c;
		};
		return(bw_c);
	};
	if(1000==model or 1100==model){ // Fixed Flatte as in Dimas program BWF0FLATTE in $PWASYS/phys/breitwigners/bw.f (called by PWA when itype_f0_980 = 2)
		double wPi = 0.1395675; // Attention: Constants are fixed here
		double wK  = 0.493677;
		xdouble par1 = 0.965; // Mass
		xdouble par2 = 0.165; // Width
		xdouble par4 = 4.21;  //
		if (1100==model){
			par1 = param[0]; // Mass
			par2 = param[1]; // Width
			par4 = param[2];  //
//			std::cout << par1 <<" "<<par2<<" "<<par3<<" "<<par4<<std::endl;
		};
		xdouble par3 = par2*par4;
		if (m <= 2*wPi){
			return std::complex<xdouble>(0.,0.);
		};
		xdouble pPi = 2. * (0.5*pow((m*m-4*wPi*wPi)*(m*m-4*wPi*wPi),0.25))/m;
		xdouble pK  = 2. * (0.5*pow((m*m-4*wK *wK )*(m*m-4*wK *wK ),0.25))/m;
		xdouble A;
		xdouble B;
		if (m*m <= 4*wK* wK){
			A = par1*par1 - m*m + par3 * pK;
			B = par2 * pPi;
//			std::cout << "caseA";
		}else{
			A = par1*par1 - m*m;
			B = par2 * (pPi + par4 * pK);
//			std::cout<< "caseB";
		};
		xdouble BWR = A/(A*A + B*B);
		xdouble BWI = B/(A*A + B*B);
//		std::cout << m << ": pK "<< abs(m*m-4*wK *wK ) <<"   pPi "<< abs(m*m-4*wPi*wPi)<<std::endl;
		return std::complex<xdouble>(BWR,BWI);
	};
	if(1001==model or 1101==model){ // (PiPi)_S wave as in Dimas PWA program (with itype_eps = 1)
		int N = 1; // Channel ??

		std::complex<xdouble> ALFA = std::complex<xdouble>(0.,0.);
		if(1101==model){
			ALFA = std::complex<xdouble>(param[0],0.);
		};
		double S0   = -0.0074;
		double S1   =  0.9828;
		double FC11 =  0.1968;
		double FC12 = -0.0154;
		double A11  =  0.1131;
		double A12  =  0.0150;
		double A22  = -0.3216;
		double C011 =  0.0337;
		double C111 = -0.3185;
		double C211 = -0.0942;
		double C311 = -0.5927;
		double C411 =  0.;
		double C012 = -0.2826; 
		double C112 =  0.0918;
		double C212 =  0.1669;
		double C312 = -0.2082;
		double C412 = -0.1386;
		double C022 =  0.3010;
		double C122 = -0.5140;
		double C222 =  0.1176;
		double C322 =  0.5204;
		double C422 =  0.;

		double WPI = 0.1395675;
		double WKC = 0.493646;   //  ! PI+- MASS, K+- MASS
		double WK0 = 0.497671;
		double WK  = 0.5*(WKC+WK0); // ! K0 MASS, K MEAN MASS

		double S  = m*m;
		if( N==1 and S<=(4.*WPI*WPI)){
			return std::complex<xdouble>(0.,0.);
		};
		if ( N==2 and S<=(4.*WKC*WKC)){
			return std::complex<xdouble>(0.,0.);
		};
		if(S == S1){
			return std::complex<xdouble>(0.,0.);
		};
		double R1 = sqrt(1. - 4.*WPI*WPI/S); // PI+-
		double RR2  = 0.;
		double RI2  = 0.;
		double RKC  = pow((1. - 4.*WKC*WKC/S)*(1. - 4.*WKC*WKC/S),0.25); // K+-
		double RK0  = pow((1. - 4.*WK0*WK0/S)*(1. - 4.*WK0*WK0/S),0.25); // K0
		if(S>(4.*WKC*WKC)){
			RI2  = RKC;
		}else{
			RR2  = - RKC;
			RKC  = 0.; // PHYSICAL PHASE SPACE
		};
		if(S>(4.*WK0*WK0) ){
			RI2  = RI2 + RK0;
		}else{
			RR2  = RR2 - RK0;
			RK0  = 0.;
		};
		RR2  = 0.5*RR2; // K MEAN
		RI2  = 0.5*RI2;
		//First compute sum of poles (one pole for M solution).
		//Misprint in the sign was corrected !!!
		xdouble M11 = A11/(S-S0);
		xdouble M12 = 0.;
		xdouble M22 = A22/(S-S0);
		//Add polinomial background.
		double X    = S/(4.*WK*WK) -1.;
		M11 = M11 + ((((C411*X+C311)*X)+C211)*X+C111)*X+C011;
		M22 = M22 + ((((C422*X+C322)*X)+C222)*X+C122)*X+C022;
		//ADD REAL PART OF K PHASE SPACE TO THE M-MATRIX BELOW THRESHOLD.
		M22 = M22 + RR2;
		double R2   = RI2;
		//COMPUTE F = (M - i*R)^-1
		xdouble DETMX = M11*M22 - M12*M12 - R1*R2;
		xdouble DETMY = -M11*R2 - M22*R1;
		std::complex<xdouble> DETF  = std::complex<xdouble>(DETMX/(DETMX*DETMX+DETMY*DETMY),-DETMY/(DETMX*DETMX+DETMY*DETMY));
		std::complex<xdouble> F11 = std::complex<xdouble>(M22,-R2) * DETF;
		std::complex<xdouble> F12 = std::complex<xdouble>(-M12,  0.) * DETF;
		std::complex<xdouble> F22 = std::complex<xdouble>(M11,-R1) * DETF;
		std::complex<xdouble> F21 = F12;
		// F MATRIX READY, COMPUTE COUPLING COEFFICIENTS (?)
		if(N == 1){
			return (std::complex<xdouble>(1.,0.)-ALFA)*F11 + ALFA*F21;
		};
		if(N == 2){
			return (std::complex<xdouble>(1.,0.)-ALFA)*F12 + ALFA*F22;
		};
		return std::complex<xdouble>(0.,0.);
	};
	if(1002==model or 1102==model){ // Rho used bui COMPASS PWA with itype_rho = 1
		xdouble wPi  = 0.13956755;
		xdouble wRho = 0.776794614;//0.7685;
		xdouble GRho = 0.152598585;//0.1507;
		if(1102==model){
			wRho = param[0];
			GRho = param[1];
		};

		xdouble R1   = 4.94;

		xdouble S = m*m;
		xdouble pabs = sqrt(std::complex<xdouble>(1-4*wPi*wPi/S,0.)).real();
		
		xdouble E = m/2;
		xdouble E0= wRho/2;

		xdouble P0 = sqrt(E0*E0 - wPi*wPi);
		xdouble P  = sqrt(std::complex<xdouble>(E*E-wPi*wPi,0.)).real();

		xdouble FD10 = 1.+ P0*P0*R1*R1;
		xdouble FD1  = 1.+ P*P*R1*R1;
		
		xdouble G = GRho * (P*P*P/P0/P0/P0) * FD10/FD1;
		xdouble C = sqrt(G*wRho*wRho/P)/P;
		
		xdouble A = wRho*wRho - S;
		xdouble B = wRho * G;
		xdouble DEN = A*A+B*B;
		
		return std::complex<xdouble>(pabs*C*A/DEN,pabs*C*B/DEN);

	};
	std::cerr << "breitWigners.h: Error: Invalid number '" << model << "' for fit-function, use vanishing amplitude: 0. + 0. i."<<std::endl;
	return std::complex<xdouble>(0.,0.);
};

//////////////////////////////////////////  NUMBER OF PARAMETERS NEEDED BY THE DEFINITIONS ABOVE  ///////////////////////////

int getNpars(int model){ 	// Currently has to match the function 'bw(...,model)'.
	if (-1==model){			// Constant function.
		return 0;
	};
	if (0==model){			// Simple Breit-Wigner
		return 2;
	};
	if (1==model){			// Mass dependent Breit-Wigner (one channel )
		return 4;		
	};
	if (2==model){			// Mass dependent Breit-Wigner (two channels)
		return 6;
	};
	if (3==model){			// e^{-\alpha q^2} non-resonant component //Vandermeulen Phase Space
		return 3;
	};
	if (4==model){			// Valera, Dorofeev Background
		return 3;
	};
	if (5==model){			// Bowler parametrization
		return 2;
	};
	if (6==model){			// Flatte
		return 5;
	};
	if (9==model){			// Simple gaus
		return 2;
	};
	if (10==model){			// Polynomial up to 4th order
		return 5;
	};
	if (22==model){	
		return 6.;
	};
	if (101==model){		// t' dependent background
		return 8;
	};
	if (model>=510 and model <= 512){// Different matrix elements of a p-vector approach (Given by Michael Pennington).
		return 2;
	};
	if (model>=610 and model <= 612){// Different matrix elements of a p-vector approach (Given by Michael Pennington). (Parameters are touched !!!!!)
		return 17;
	};
	if (1000==model){
		return 0;
	};
	if (1100==model){
		return 3;
	};
	if (1001==model){
		return 0;
	};
	if (1101==model){
		return 1;
	};
	if (1002==model){
		return 0;
	};
	if (1102==model){
		return 2;
	};
	std::cout << "breitWigners.h: Warning: Invalid number '"<< model <<"' for fit-function. Use 20 parameters to be sure." << std::endl;
	return 20;
};
//////////////////////////////////////////  NUMBER OF NON CONST PARAMETERS NEEDED BY THE DEFINITIONS ABOVE  ///////////////////////////

int getNparsNonConst(int model){ 	// Currently has to match the function 'bw(...,model)'.
	if (-1==model){			// Constant function.
		return 0;
	};
	if (0==model){			// Simple Breit-Wigner
		return 2;
	};
	if (1==model){			// Mass dependent Breit-Wigner (one channel )
		return 2;		
	};
	if (2==model){			// Mass dependent Breit-Wigner (two channels)
		return 2;
	};
	if (3==model){			// e^{-\alpha q^2} non-resonant component //Vandermeulen Phase Space
		return 1;
	};
	if (4==model){			// Valera, Dorofeev Background
		return 2;
	};
	if (5==model){			// Bowler parametrization
		return 2;
	};
	if (6==model){			// Flatte
		return 5;
	};
	if (9==model){			// Simple gaus
		return 2;
	};
	if (10==model){			// Polynomial up to 4th order
		return 5;
	};
	if (22==model){	
		return 2.;
	};
	if (101==model){		// t' dependent background
		return 4;
	};
	if (model>=510 and model <= 512){// Different matrix elements of a p-vector approach (Given by Michael Pennington).
		return 0;
	};
	if (model>=610 and model <= 612){// Different matrix elements of a p-vector approach (Given by Michael Pennington). (Parameters are touched !!!!!)
		return 15;
	};
	if (1000==model){
		return 0;
	};
	if (1100==model){
		return 3;
	};
	if (1001==model){
		return 0;
	};
	if (1101==model){
		return 1;
	};
	if(1002==model){
		return 0;
	};
	if(1102==model){
		return 2;
	};
	std::cout << "breitWigners.h: Warning: Invalid number '"<< model <<"' for fit-function. Use 20 parameters to be sure." << std::endl;
	return 20;
};

///////////////////////////////////// NAMES OF THE PARAMETRIZATIONS ///////////////////////////////////////////////////

std::string getModelName(int model){ 	// Currently has to match the function 'bw(...,model)'.
	if (-1==model){
		return "constant";
	};
	if (0==model){
		return "breit-wigner";
	};
	if (1==model){
		return "mass-dependent-BW";
	};
	if (2==model){
		return "mass-dependent-BW-2channels-dima";
	};
	if (3==model){
		return "vandermeulen-phase-space";
	};
	if (4==model){
		return "valera-dorofeev";
	};
	if (5==model){
		return "bowler-parametrization";	
	};
	if (6==model){
		return "flatte";
	};
	if (9==model){
		return "gaus";
	};
	if (10==model){
		return "polynomial";
	};
	if (22==model){
		return "mass-dependent-BW-2channels-dima";
	};
	if (101==model){
		return "t'-dependent";
	};
	if (model>=510 and model <= 512){
		return "pennington-p-vector";
	};
	if (model>=610 and model <= 612){
		return "pennington-p-vector-released";
	};
	if (1000==model or 1100==model){
		return "Flatte_used_by_the_PWA_program";
	};
	if (1001==model or 1101==model){
		return "Epsilon_used_by_PWA";
	};
	if (1002==model or 1102==model){
		return "Rho_used_by_PWA_TYPE_RHO_1"; 
	};
	return "unknown";
};

#endif
