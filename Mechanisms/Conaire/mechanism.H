#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  H
2  N
3  Xe
*/

// Species
#define XE_ID 0
#define H2_ID 1
#define H_ID 2
#define OH_ID 3
#define O_ID 4
#define H2O_ID 5
#define HO2_ID 6
#define H2O2_ID 7
#define N2_ID 8
#define O2_ID 9

#define NUM_ELEMENTS 4
#define NUM_SPECIES 10
#define NUM_IONS 0
#define NUM_REACTIONS 21

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or GPU based. Defined in mechanism.cpp 
void atomicWeight(amrex::Real *  awt);
//  MISC 
void CKAWT(amrex::Real *  awt);
void CKNCF(int * ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int * _rmap);
void CKINU(const int i, int &nspec, int * ki, int * nu);
void CKKFKR(const amrex::Real P, const amrex::Real T,const amrex::Real * x, amrex::Real *  q_f, amrex::Real *  q_r);
void progressRateFR(amrex::Real *  q_f, amrex::Real *  q_r,amrex::Real *  sc, amrex::Real T);
//  SPARSE INFORMATION 
void SPARSITY_INFO(int * nJdata, const int * consP, int NCELLS);
void SPARSITY_INFO_SYST(int * nJdata, const int * consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int * nJdata, const int * consP);
void SPARSITY_PREPROC_CSC(int * rowVals, int * colPtrs, const int * consP, int NCELLS);
void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, const int * consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtrs, const int * consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, const int * consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, const int * consP, int base);


// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
mm = 4;
kk = 10;
ii = 21;
nfit = -1; // Why do you need this anyway ? 
}

//  inverse molecular weights 
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[10]={
0.0076165522914398,// XE
0.4960317460317460,// H2
0.9920634920634921,// H
0.0587993179279120,// OH
0.0625039064941559,// O
0.0555092978073827,// H2O
0.0302975216627280,// HO2
0.0293996589639560,// H2O2
0.0356964374955379,// N2
0.0312519532470779,// O2
};
#endif
const amrex::Real h_global_imw[10]={
0.0076165522914398,// XE
0.4960317460317460,// H2
0.9920634920634921,// H
0.0587993179279120,// OH
0.0625039064941559,// O
0.0555092978073827,// H2O
0.0302975216627280,// HO2
0.0293996589639560,// H2O2
0.0356964374955379,// N2
0.0312519532470779,// O2
};

//  molecular weights 
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[10]={
131.293000,// XE
2.016000,// H2
1.008000,// H
17.007000,// OH
15.999000,// O
18.015000,// H2O
33.006000,// HO2
34.014000,// H2O2
28.014000,// N2
31.998000,// O2
};
#endif
const amrex::Real h_global_mw[10]={
131.293000,// XE
2.016000,// H2
1.008000,// H
17.007000,// OH
15.999000,// O
18.015000,// H2O
33.006000,// HO2
34.014000,// H2O2
28.014000,// N2
31.998000,// O2
};

//  inverse molecular weights 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_imw(amrex::Real *imw_new){
imw_new[0] = 0.0076165522914398;// XE
imw_new[1] = 0.4960317460317460;// H2
imw_new[2] = 0.9920634920634921;// H
imw_new[3] = 0.0587993179279120;// OH
imw_new[4] = 0.0625039064941559;// O
imw_new[5] = 0.0555092978073827;// H2O
imw_new[6] = 0.0302975216627280;// HO2
imw_new[7] = 0.0293996589639560;// H2O2
imw_new[8] = 0.0356964374955379;// N2
imw_new[9] = 0.0312519532470779;// O2
}

//  inverse molecular weight 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real imw(const int n){
#if AMREX_DEVICE_COMPILE
return global_imw[n];
#else
return h_global_imw[n];
#endif
}
//  molecular weights 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_mw(amrex::Real *mw_new){
mw_new[0] = 131.293000;// XE
mw_new[1] = 2.016000;// H2
mw_new[2] = 1.008000;// H
mw_new[3] = 17.007000;// OH
mw_new[4] = 15.999000;// O
mw_new[5] = 18.015000;// H2O
mw_new[6] = 33.006000;// HO2
mw_new[7] = 34.014000;// H2O2
mw_new[8] = 28.014000;// N2
mw_new[9] = 31.998000;// O2
}

//  molecular weight 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mw(const int n){
#if AMREX_DEVICE_COMPILE
return global_mw[n];
#else
return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
 ru  = 8.31446261815324e+07; 
 ruc = 1.98721558317399615845; 
 pa  = 1.01325e+06; 
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void cv_R(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
+1.50000000e+00 
;
// species 1: H2
species[1] =
+4.07832321e+04 * invT2-8.00918604e+02 * invT+7.21470201e+00 -1.26971446e-02 * T+1.75360508e-05  * T2-1.20286027e-08 * T3+3.36809349e-12 * T4
;
// species 2: H
species[2] =
+1.50000000e+00 
;
// species 5: H2O
species[5] =
-3.94796083e+04 * invT2+5.75573102e+02 * invT-6.82173470e-02 +7.22271286e-03 * T-7.34255737e-06  * T2+4.95504349e-09 * T3-1.33693325e-12 * T4
;
// species 6: HO2
species[6] =
-7.59888254e+04 * invT2+1.32938392e+03 * invT-5.67738824e+00 +2.50830820e-02 * T-3.00655159e-05  * T2+1.89560006e-08 * T3-4.82856739e-12 * T4
;
// species 7: H2O2
species[7] =
-9.27953358e+04 * invT2+1.56474839e+03 * invT-6.97646014e+00 +3.27074452e-02 * T-3.93219326e-05  * T2+2.50925524e-08 * T3-6.46504529e-12 * T4
;
// species 8: N2
species[8] =
+2.21037150e+04 * invT2-3.81846182e+02 * invT+5.08273836e+00 -8.53091441e-03 * T+1.38464619e-05  * T2-9.62579362e-09 * T3+2.51970581e-12 * T4
;
}
else {
// species 0: XE
species[0] =
+4.02522668e+03 * invT2-1.20950752e+01 * invT+1.51415335e+00 -8.24810208e-06 * T+2.53023262e-09  * T2-3.89233323e-13 * T3+2.36043914e-17 * T4
;
// species 1: H2
species[1] =
+5.60812801e+05 * invT2-8.37150474e+02 * invT+1.97536453e+00 +1.25224912e-03 * T-3.74071619e-07  * T2+5.93662520e-11 * T3-3.60699410e-15 * T4
;
// species 2: H
species[2] =
+6.07877425e+01 * invT2-1.81935442e-01 * invT+1.50021182e+00 -1.22651286e-07 * T+3.73287633e-11  * T2-5.68774456e-15 * T3+3.41021020e-19 * T4
;
// species 5: H2O
species[5] =
+1.03497210e+06 * invT2-2.41269856e+03 * invT+3.64611078e+00 +2.29199831e-03 * T-6.83683048e-07  * T2+9.42646893e-11 * T3-4.82238053e-15 * T4
;
// species 6: HO2
species[6] =
-1.81066972e+06 * invT2+4.96319203e+03 * invT-2.03949899e+00 +4.56014853e-03 * T-1.06185945e-06  * T2+1.14456788e-10 * T3-4.76306416e-15 * T4
;
// species 7: H2O2
species[7] =
+1.48942803e+06 * invT2-5.17082178e+03 * invT+1.02820497e+01 -8.04239779e-05 * T-1.81838377e-08  * T2+6.94726559e-12 * T3-4.82783190e-16 * T4
;
// species 8: N2
species[8] =
+5.87712406e+05 * invT2-2.23924907e+03 * invT+5.06694922e+00 -6.13968550e-04 * T+1.49180668e-07  * T2-1.92310549e-11 * T3+1.06195439e-15 * T4
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
-1.99885899e+03 * invT2+9.30013616e+01 * invT+2.05085423e+00 +1.52952929e-03 * T-3.15789100e-06  * T2+3.31544618e-09 * T3-1.13876268e-12 * T4
;
// species 4: O
species[4] =
-7.95361130e+03 * invT2+1.60717779e+02 * invT+9.66226438e-01 +1.01367031e-03 * T-1.11041542e-06  * T2+6.51750750e-10 * T3-1.58477925e-13 * T4
;
// species 9: O2
species[9] =
-3.42556342e+04 * invT2+4.84700097e+02 * invT+1.19010961e-01 +4.29388924e-03 * T-6.83630052e-07  * T2-2.02337270e-09 * T3+1.03904002e-12 * T4
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
+1.01739338e+06 * invT2-2.50995728e+03 * invT+4.11654786e+00 +1.30529993e-04 * T-8.28432226e-08  * T2+2.00647594e-11 * T3-1.55699366e-15 * T4
;
// species 4: O
species[4] =
+2.61902026e+05 * invT2-7.29872203e+02 * invT+2.31717727e+00 -4.28133436e-04 * T+1.03610459e-07  * T2-9.43830433e-12 * T3+2.72503830e-16 * T4
;
// species 9: O2
species[9] =
-1.03793902e+06 * invT2+2.34483028e+03 * invT+8.19732036e-01 +1.26784758e-03 * T-2.18806799e-07  * T2+2.05371957e-11 * T3-8.19346705e-16 * T4
;
}
else {
// species 3: OH
species[3] =
+2.84723419e+08 * invT2-1.85953261e+05 * invT+4.90824090e+01 -5.14237498e-03 * T+2.87553659e-07  * T2-8.22881796e-12 * T3+9.56722902e-17 * T4
;
// species 4: O
species[4] =
+1.77900426e+08 * invT2-1.08232826e+05 * invT+2.71077836e+01 -2.97523226e-03 * T+1.85499753e-07  * T2-5.79623154e-12 * T3+7.19172016e-17 * T4
;
// species 9: O2
species[9] =
+4.97529430e+08 * invT2-2.86610687e+05 * invT+6.59035225e+01 -6.16995902e-03 * T+3.01639603e-07  * T2-7.42141660e-12 * T3+7.27817577e-17 * T4
;
}
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void cp_R(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
+2.50000000e+00 
;
// species 1: H2
species[1] =
+4.07832321e+04 * invT2-8.00918604e+02 * invT+8.21470201e+00 -1.26971446e-02 * T+1.75360508e-05  * T2-1.20286027e-08 * T3+3.36809349e-12 * T4
;
// species 2: H
species[2] =
+2.50000000e+00 
;
// species 5: H2O
species[5] =
-3.94796083e+04 * invT2+5.75573102e+02 * invT+9.31782653e-01 +7.22271286e-03 * T-7.34255737e-06  * T2+4.95504349e-09 * T3-1.33693325e-12 * T4
;
// species 6: HO2
species[6] =
-7.59888254e+04 * invT2+1.32938392e+03 * invT-4.67738824e+00 +2.50830820e-02 * T-3.00655159e-05  * T2+1.89560006e-08 * T3-4.82856739e-12 * T4
;
// species 7: H2O2
species[7] =
-9.27953358e+04 * invT2+1.56474839e+03 * invT-5.97646014e+00 +3.27074452e-02 * T-3.93219326e-05  * T2+2.50925524e-08 * T3-6.46504529e-12 * T4
;
// species 8: N2
species[8] =
+2.21037150e+04 * invT2-3.81846182e+02 * invT+6.08273836e+00 -8.53091441e-03 * T+1.38464619e-05  * T2-9.62579362e-09 * T3+2.51970581e-12 * T4
;
}
else {
// species 0: XE
species[0] =
+4.02522668e+03 * invT2-1.20950752e+01 * invT+2.51415335e+00 -8.24810208e-06 * T+2.53023262e-09  * T2-3.89233323e-13 * T3+2.36043914e-17 * T4
;
// species 1: H2
species[1] =
+5.60812801e+05 * invT2-8.37150474e+02 * invT+2.97536453e+00 +1.25224912e-03 * T-3.74071619e-07  * T2+5.93662520e-11 * T3-3.60699410e-15 * T4
;
// species 2: H
species[2] =
+6.07877425e+01 * invT2-1.81935442e-01 * invT+2.50021182e+00 -1.22651286e-07 * T+3.73287633e-11  * T2-5.68774456e-15 * T3+3.41021020e-19 * T4
;
// species 5: H2O
species[5] =
+1.03497210e+06 * invT2-2.41269856e+03 * invT+4.64611078e+00 +2.29199831e-03 * T-6.83683048e-07  * T2+9.42646893e-11 * T3-4.82238053e-15 * T4
;
// species 6: HO2
species[6] =
-1.81066972e+06 * invT2+4.96319203e+03 * invT-1.03949899e+00 +4.56014853e-03 * T-1.06185945e-06  * T2+1.14456788e-10 * T3-4.76306416e-15 * T4
;
// species 7: H2O2
species[7] =
+1.48942803e+06 * invT2-5.17082178e+03 * invT+1.12820497e+01 -8.04239779e-05 * T-1.81838377e-08  * T2+6.94726559e-12 * T3-4.82783190e-16 * T4
;
// species 8: N2
species[8] =
+5.87712406e+05 * invT2-2.23924907e+03 * invT+6.06694922e+00 -6.13968550e-04 * T+1.49180668e-07  * T2-1.92310549e-11 * T3+1.06195439e-15 * T4
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
-1.99885899e+03 * invT2+9.30013616e+01 * invT+3.05085423e+00 +1.52952929e-03 * T-3.15789100e-06  * T2+3.31544618e-09 * T3-1.13876268e-12 * T4
;
// species 4: O
species[4] =
-7.95361130e+03 * invT2+1.60717779e+02 * invT+1.96622644e+00 +1.01367031e-03 * T-1.11041542e-06  * T2+6.51750750e-10 * T3-1.58477925e-13 * T4
;
// species 9: O2
species[9] =
-3.42556342e+04 * invT2+4.84700097e+02 * invT+1.11901096e+00 +4.29388924e-03 * T-6.83630052e-07  * T2-2.02337270e-09 * T3+1.03904002e-12 * T4
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
+1.01739338e+06 * invT2-2.50995728e+03 * invT+5.11654786e+00 +1.30529993e-04 * T-8.28432226e-08  * T2+2.00647594e-11 * T3-1.55699366e-15 * T4
;
// species 4: O
species[4] =
+2.61902026e+05 * invT2-7.29872203e+02 * invT+3.31717727e+00 -4.28133436e-04 * T+1.03610459e-07  * T2-9.43830433e-12 * T3+2.72503830e-16 * T4
;
// species 9: O2
species[9] =
-1.03793902e+06 * invT2+2.34483028e+03 * invT+1.81973204e+00 +1.26784758e-03 * T-2.18806799e-07  * T2+2.05371957e-11 * T3-8.19346705e-16 * T4
;
}
else {
// species 3: OH
species[3] =
+2.84723419e+08 * invT2-1.85953261e+05 * invT+5.00824090e+01 -5.14237498e-03 * T+2.87553659e-07  * T2-8.22881796e-12 * T3+9.56722902e-17 * T4
;
// species 4: O
species[4] =
+1.77900426e+08 * invT2-1.08232826e+05 * invT+2.81077836e+01 -2.97523226e-03 * T+1.85499753e-07  * T2-5.79623154e-12 * T3+7.19172016e-17 * T4
;
// species 9: O2
species[9] =
+4.97529430e+08 * invT2-2.86610687e+05 * invT+6.69035225e+01 -6.16995902e-03 * T+3.01639603e-07  * T2-7.42141660e-12 * T3+7.27817577e-17 * T4
;
}
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void gibbs(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
-7.453750000000000e+02 * invT-2.500000000000000e+00 * logT-3.664454205000000e+00 
;
// species 1: H2
species[1] =
-2.039161605000000e+04 * invT2+1.881566061000000e+03 * invT-8.009186040000000e+02 * logT * invT-8.214702010000000e+00 * logT+3.865259045000000e+01 +6.348572285000000e-03 * T-2.922675126666667e-06 * T2+1.002383558333333e-09 * T3-1.684046745000000e-13 * T4
;
// species 2: H
species[2] =
+2.547370801000000e+04 * invT-2.500000000000000e+00 * logT+2.946682853000000e+00 
;
// species 5: H2O
species[5] =
+1.973980415000000e+04 * invT2-3.246416999800000e+04 * invT+5.755731019999999e+02 * logT * invT-9.317826530000000e-01 * logT-1.631027509700000e+01 -3.611356430000000e-03 * T+1.223759561666667e-06 * T2-4.129202908333333e-10 * T3+6.684666230000000e-14 * T4
;
// species 6: HO2
species[6] =
+3.799441270000000e+04 * invT2-4.543967042000000e+03 * invT+1.329383918000000e+03 * logT * invT+4.677388240000000e+00 * logT-5.661340964000000e+01 -1.254154101000000e-02 * T+5.010919313333334e-06 * T2-1.579666713333333e-09 * T3+2.414283695000000e-13 * T4
;
// species 7: H2O2
species[7] =
+4.639766790000000e+04 * invT2-2.337529889500000e+04 * invT+1.564748385000000e+03 * logT * invT+5.976460140000000e+00 * logT-6.474820194000000e+01 -1.635372260000000e-02 * T+6.553655433333333e-06 * T2-2.091046029166667e-09 * T3+3.232522645000000e-13 * T4
;
// species 8: N2
species[8] =
-1.105185748500000e+04 * invT2+3.289999040000000e+02 * invT-3.818461820000000e+02 * logT * invT-6.082738360000000e+00 * logT+1.684277580000000e+01 +4.265457205000000e-03 * T-2.307743648333333e-06 * T2+8.021494683333333e-10 * T3-1.259852904500000e-13 * T4
;
}
else {
// species 0: XE
species[0] =
-2.012613340000000e+03 * invT2-6.806751482100000e+02 * invT-1.209507521000000e+01 * logT * invT-2.514153347000000e+00 * logT-3.549557368000000e+00 +4.124051040000000e-06 * T-4.217054363333333e-10 * T2+3.243611025000000e-14 * T3-1.180219569000000e-18 * T4
;
// species 1: H2
species[1] =
-2.804064005000000e+05 * invT2+4.502673936000000e+03 * invT-8.371504740000000e+02 * logT * invT-2.975364532000000e+00 * logT+5.178139301000000e+00 -6.261245620000000e-04 * T+6.234526983333333e-08 * T2-4.947187666666667e-12 * T3+1.803497050000000e-16 * T4
;
// species 2: H
species[2] =
-3.039387125000000e+01 * invT2+2.547468204455830e+04 * invT-1.819354417000000e-01 * logT * invT-2.500211817000000e+00 * logT+2.948403594000000e+00 +6.132564319999999e-08 * T-6.221460550000000e-12 * T2+4.739787133333333e-16 * T3-1.705105098500000e-20 * T4
;
// species 5: H2O
species[5] =
-5.174860480000000e+05 * invT2-1.625556365200000e+04 * invT-2.412698562000000e+03 * logT * invT-4.646110780000000e+00 * logT+1.262425929000000e+01 -1.145999153500000e-03 * T+1.139471746666667e-07 * T2-7.855390775000000e-12 * T3+2.411190265000000e-16 * T4
;
// species 6: HO2
species[6] =
+9.053348620000000e+05 * invT2-2.704497987000000e+04 * invT+4.963192030000000e+03 * logT * invT+1.039498992000000e+00 * logT-4.170800819200000e+01 -2.280074265000000e-03 * T+1.769765745000000e-07 * T2-9.538065650000000e-12 * T3+2.381532080000000e-16 * T4
;
// species 7: H2O2
species[7] =
-7.447140135000000e+05 * invT2+9.011688599999999e+03 * invT-5.170821780000000e+03 * logT * invT-1.128204970000000e+01 * logT+5.779060630000000e+01 +4.021198895000000e-05 * T+3.030639615000000e-09 * T2-5.789387991666667e-13 * T3+2.413915950000000e-17 * T4
;
// species 8: N2
species[8] =
-2.938562030000000e+05 * invT2+1.059285507700000e+04 * invT-2.239249073000000e+03 * logT * invT-6.066949220000000e+00 * logT+2.193334949000000e+01 +3.069842750000000e-04 * T-2.486344465000000e-08 * T2+1.602587904166667e-12 * T3-5.309771930000000e-17 * T4
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+9.994294950000000e+02 * invT2+3.084215596600000e+03 * invT+9.300136160000000e+01 * logT * invT-3.050854229000000e+00 * logT-1.623256561000000e+00 -7.647646440000001e-04 * T+5.263151663333333e-07 * T2-2.762871816666667e-10 * T3+5.693813415000000e-14 * T4
;
// species 4: O
species[4] =
+3.976805650000000e+03 * invT2+2.856434214870000e+04 * invT+1.607177787000000e+02 * logT * invT-1.966226438000000e+00 * logT-6.438015382000000e+00 -5.068351550000000e-04 * T+1.850692371666667e-07 * T2-5.431256250000000e-11 * T3+7.923896255000001e-15 * T4
;
// species 9: O2
species[9] =
+1.712781710000000e+04 * invT2-2.906754773000000e+03 * invT+4.847000970000000e+02 * logT * invT-1.119010961000000e+00 * logT-1.737798373900000e+01 -2.146944620000000e-03 * T+1.139383420000000e-07 * T2+1.686143916666667e-10 * T3-5.195200090000000e-14 * T4
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-5.086966895000000e+05 * invT2+1.768644478400000e+04 * invT-2.509957276000000e+03 * logT * invT-5.116547860000000e+00 * logT+1.612937123000000e+01 -6.526499650000001e-05 * T+1.380720376666667e-08 * T2-1.672063284166667e-12 * T3+7.784968280000000e-17 * T4
;
// species 4: O
species[4] =
-1.309510131000000e+05 * invT2+3.319440839700000e+04 * invT-7.298722030000000e+02 * logT * invT-3.317177270000000e+00 * logT+3.985135805000000e+00 +2.140667180000000e-04 * T-1.726840990000000e-08 * T2+7.865253608333333e-13 * T3-1.362519148500000e-17 * T4
;
// species 9: O2
species[9] =
+5.189695110000000e+05 * invT2-1.454527900800000e+04 * invT+2.344830282000000e+03 * logT * invT-1.819732036000000e+00 * logT-1.556743302400000e+01 -6.339237910000000e-04 * T+3.646779980000000e-08 * T2-1.711432976666667e-12 * T3+4.096733525000000e-17 * T4
;
}
else {
// species 3: OH
species[3] =
-1.423617096500000e+08 * invT2+1.282440646800000e+06 * invT-1.859532612000000e+05 * logT * invT-5.008240900000000e+01 * logT+4.524379670000000e+02 +2.571187490000000e-03 * T-4.792560981666667e-08 * T2+6.857348299999999e-13 * T3-4.783614509999999e-18 * T4
;
// species 4: O
species[4] =
-8.895021320000000e+07 * invT2+7.808614373000000e+05 * invT-1.082328257000000e+05 * logT * invT-2.810778365000000e+01 * logT+2.462805987500000e+02 +1.487616131000000e-03 * T-3.091662556666666e-08 * T2+4.830192950000000e-13 * T3-3.595860082000000e-18 * T4
;
// species 9: O2
species[9] =
-2.487647150000000e+08 * invT2+2.006943339600000e+06 * invT-2.866106874000000e+05 * logT * invT-6.690352249999999e+01 * logT+6.199656835000000e+02 +3.084979510000000e-03 * T-5.027326711666667e-08 * T2+6.184513833333334e-13 * T3-3.639087885000000e-18 * T4
;
}
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void helmholtz(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
-7.45375000e+02 * invT-2.50000000e+00 * logT-4.66445421e+00 
;
// species 1: H2
species[1] =
-2.03916161e+04 * invT2+1.88156606e+03 * invT-8.00918604e+02 * logT * invT-8.21470201e+00 * logT+3.76525904e+01 +6.34857229e-03 * T-2.92267513e-06 * T2+1.00238356e-09 * T3-1.68404675e-13 * T4
;
// species 2: H
species[2] =
+2.54737080e+04 * invT-2.50000000e+00 * logT+1.94668285e+00 
;
// species 5: H2O
species[5] =
+1.97398041e+04 * invT2-3.24641700e+04 * invT+5.75573102e+02 * logT * invT-9.31782653e-01 * logT-1.73102751e+01 -3.61135643e-03 * T+1.22375956e-06 * T2-4.12920291e-10 * T3+6.68466623e-14 * T4
;
// species 6: HO2
species[6] =
+3.79944127e+04 * invT2-4.54396704e+03 * invT+1.32938392e+03 * logT * invT+4.67738824e+00 * logT-5.76134096e+01 -1.25415410e-02 * T+5.01091931e-06 * T2-1.57966671e-09 * T3+2.41428370e-13 * T4
;
// species 7: H2O2
species[7] =
+4.63976679e+04 * invT2-2.33752989e+04 * invT+1.56474839e+03 * logT * invT+5.97646014e+00 * logT-6.57482019e+01 -1.63537226e-02 * T+6.55365543e-06 * T2-2.09104603e-09 * T3+3.23252264e-13 * T4
;
// species 8: N2
species[8] =
-1.10518575e+04 * invT2+3.28999904e+02 * invT-3.81846182e+02 * logT * invT-6.08273836e+00 * logT+1.58427758e+01 +4.26545720e-03 * T-2.30774365e-06 * T2+8.02149468e-10 * T3-1.25985290e-13 * T4
;
}
else {
// species 0: XE
species[0] =
-2.01261334e+03 * invT2-6.80675148e+02 * invT-1.20950752e+01 * logT * invT-2.51415335e+00 * logT-4.54955737e+00 +4.12405104e-06 * T-4.21705436e-10 * T2+3.24361103e-14 * T3-1.18021957e-18 * T4
;
// species 1: H2
species[1] =
-2.80406400e+05 * invT2+4.50267394e+03 * invT-8.37150474e+02 * logT * invT-2.97536453e+00 * logT+4.17813930e+00 -6.26124562e-04 * T+6.23452698e-08 * T2-4.94718767e-12 * T3+1.80349705e-16 * T4
;
// species 2: H
species[2] =
-3.03938713e+01 * invT2+2.54746820e+04 * invT-1.81935442e-01 * logT * invT-2.50021182e+00 * logT+1.94840359e+00 +6.13256432e-08 * T-6.22146055e-12 * T2+4.73978713e-16 * T3-1.70510510e-20 * T4
;
// species 5: H2O
species[5] =
-5.17486048e+05 * invT2-1.62555637e+04 * invT-2.41269856e+03 * logT * invT-4.64611078e+00 * logT+1.16242593e+01 -1.14599915e-03 * T+1.13947175e-07 * T2-7.85539078e-12 * T3+2.41119027e-16 * T4
;
// species 6: HO2
species[6] =
+9.05334862e+05 * invT2-2.70449799e+04 * invT+4.96319203e+03 * logT * invT+1.03949899e+00 * logT-4.27080082e+01 -2.28007426e-03 * T+1.76976574e-07 * T2-9.53806565e-12 * T3+2.38153208e-16 * T4
;
// species 7: H2O2
species[7] =
-7.44714014e+05 * invT2+9.01168860e+03 * invT-5.17082178e+03 * logT * invT-1.12820497e+01 * logT+5.67906063e+01 +4.02119890e-05 * T+3.03063962e-09 * T2-5.78938799e-13 * T3+2.41391595e-17 * T4
;
// species 8: N2
species[8] =
-2.93856203e+05 * invT2+1.05928551e+04 * invT-2.23924907e+03 * logT * invT-6.06694922e+00 * logT+2.09333495e+01 +3.06984275e-04 * T-2.48634447e-08 * T2+1.60258790e-12 * T3-5.30977193e-17 * T4
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+9.99429495e+02 * invT2+3.08421560e+03 * invT+9.30013616e+01 * logT * invT-3.05085423e+00 * logT-2.62325656e+00 -7.64764644e-04 * T+5.26315166e-07 * T2-2.76287182e-10 * T3+5.69381341e-14 * T4
;
// species 4: O
species[4] =
+3.97680565e+03 * invT2+2.85643421e+04 * invT+1.60717779e+02 * logT * invT-1.96622644e+00 * logT-7.43801538e+00 -5.06835155e-04 * T+1.85069237e-07 * T2-5.43125625e-11 * T3+7.92389626e-15 * T4
;
// species 9: O2
species[9] =
+1.71278171e+04 * invT2-2.90675477e+03 * invT+4.84700097e+02 * logT * invT-1.11901096e+00 * logT-1.83779837e+01 -2.14694462e-03 * T+1.13938342e-07 * T2+1.68614392e-10 * T3-5.19520009e-14 * T4
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-5.08696689e+05 * invT2+1.76864448e+04 * invT-2.50995728e+03 * logT * invT-5.11654786e+00 * logT+1.51293712e+01 -6.52649965e-05 * T+1.38072038e-08 * T2-1.67206328e-12 * T3+7.78496828e-17 * T4
;
// species 4: O
species[4] =
-1.30951013e+05 * invT2+3.31944084e+04 * invT-7.29872203e+02 * logT * invT-3.31717727e+00 * logT+2.98513581e+00 +2.14066718e-04 * T-1.72684099e-08 * T2+7.86525361e-13 * T3-1.36251915e-17 * T4
;
// species 9: O2
species[9] =
+5.18969511e+05 * invT2-1.45452790e+04 * invT+2.34483028e+03 * logT * invT-1.81973204e+00 * logT-1.65674330e+01 -6.33923791e-04 * T+3.64677998e-08 * T2-1.71143298e-12 * T3+4.09673353e-17 * T4
;
}
else {
// species 3: OH
species[3] =
-1.42361710e+08 * invT2+1.28244065e+06 * invT-1.85953261e+05 * logT * invT-5.00824090e+01 * logT+4.51437967e+02 +2.57118749e-03 * T-4.79256098e-08 * T2+6.85734830e-13 * T3-4.78361451e-18 * T4
;
// species 4: O
species[4] =
-8.89502132e+07 * invT2+7.80861437e+05 * invT-1.08232826e+05 * logT * invT-2.81077836e+01 * logT+2.45280599e+02 +1.48761613e-03 * T-3.09166256e-08 * T2+4.83019295e-13 * T3-3.59586008e-18 * T4
;
// species 9: O2
species[9] =
-2.48764715e+08 * invT2+2.00694334e+06 * invT-2.86610687e+05 * logT * invT-6.69035225e+01 * logT+6.18965683e+02 +3.08497951e-03 * T-5.02732671e-08 * T2+6.18451383e-13 * T3-3.63908789e-18 * T4
;
}
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void speciesInternalEnergy(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
+1.50000000e+00 -7.45375000e+02 * invT
;
// species 1: H2
species[1] =
-4.07832321e+04 * invT2-8.00918604e+02 * logT * invT+7.21470201e+00 -6.34857229e-03 * T+5.84535025e-06 * T2-3.00715068e-09 * T3+6.73618698e-13 * T4+2.68248466e+03 * invT
;
// species 2: H
species[2] =
+1.50000000e+00 +2.54737080e+04 * invT
;
// species 5: H2O
species[5] =
+3.94796083e+04 * invT2+5.75573102e+02 * logT * invT-6.82173470e-02 +3.61135643e-03 * T-2.44751912e-06 * T2+1.23876087e-09 * T3-2.67386649e-13 * T4-3.30397431e+04 * invT
;
// species 6: HO2
species[6] =
+7.59888254e+04 * invT2+1.32938392e+03 * logT * invT-5.67738824e+00 +1.25415410e-02 * T-1.00218386e-05 * T2+4.73900014e-09 * T3-9.65713478e-13 * T4-5.87335096e+03 * invT
;
// species 7: H2O2
species[7] =
+9.27953358e+04 * invT2+1.56474839e+03 * logT * invT-6.97646014e+00 +1.63537226e-02 * T-1.31073109e-05 * T2+6.27313809e-09 * T3-1.29300906e-12 * T4-2.49400473e+04 * invT
;
// species 8: N2
species[8] =
-2.21037150e+04 * invT2-3.81846182e+02 * logT * invT+5.08273836e+00 -4.26545720e-03 * T+4.61548730e-06 * T2-2.40644840e-09 * T3+5.03941162e-13 * T4+7.10846086e+02 * invT
;
}
else {
// species 0: XE
species[0] =
-4.02522668e+03 * invT2-1.20950752e+01 * logT * invT+1.51415335e+00 -4.12405104e-06 * T+8.43410873e-10 * T2-9.73083308e-14 * T3+4.72087828e-18 * T4-6.68580073e+02 * invT
;
// species 1: H2
species[1] =
-5.60812801e+05 * invT2-8.37150474e+02 * logT * invT+1.97536453e+00 +6.26124562e-04 * T-1.24690540e-07 * T2+1.48415630e-11 * T3-7.21398820e-16 * T4+5.33982441e+03 * invT
;
// species 2: H
species[2] =
-6.07877425e+01 * invT2-1.81935442e-01 * logT * invT+1.50021182e+00 -6.13256432e-08 * T+1.24429211e-11 * T2-1.42193614e-15 * T3+6.82042039e-20 * T4+2.54748640e+04 * invT
;
// species 5: H2O
species[5] =
-1.03497210e+06 * invT2-2.41269856e+03 * logT * invT+3.64611078e+00 +1.14599915e-03 * T-2.27894349e-07 * T2+2.35661723e-11 * T3-9.64476106e-16 * T4-1.38428651e+04 * invT
;
// species 6: HO2
species[6] =
+1.81066972e+06 * invT2+4.96319203e+03 * logT * invT-2.03949899e+00 +2.28007426e-03 * T-3.53953149e-07 * T2+2.86141969e-11 * T3-9.52612832e-16 * T4-3.20081719e+04 * invT
;
// species 7: H2O2
species[7] =
-1.48942803e+06 * invT2-5.17082178e+03 * logT * invT+1.02820497e+01 -4.02119890e-05 * T-6.06127923e-09 * T2+1.73681640e-12 * T3-9.65566380e-17 * T4+1.41825104e+04 * invT
;
// species 8: N2
species[8] =
-5.87712406e+05 * invT2-2.23924907e+03 * logT * invT+5.06694922e+00 -3.06984275e-04 * T+4.97268893e-08 * T2-4.80776371e-12 * T3+2.12390877e-16 * T4+1.28321041e+04 * invT
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+1.99885899e+03 * invT2+9.30013616e+01 * logT * invT+2.05085423e+00 +7.64764644e-04 * T-1.05263033e-06 * T2+8.28861545e-10 * T3-2.27752537e-13 * T4+2.99121423e+03 * invT
;
// species 4: O
species[4] =
+7.95361130e+03 * invT2+1.60717779e+02 * logT * invT+9.66226438e-01 +5.06835155e-04 * T-3.70138474e-07 * T2+1.62937687e-10 * T3-3.16955850e-14 * T4+2.84036244e+04 * invT
;
// species 9: O2
species[9] =
+3.42556342e+04 * invT2+4.84700097e+02 * logT * invT+1.19010961e-01 +2.14694462e-03 * T-2.27876684e-07 * T2-5.05843175e-10 * T3+2.07808004e-13 * T4-3.39145487e+03 * invT
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-1.01739338e+06 * invT2-2.50995728e+03 * logT * invT+4.11654786e+00 +6.52649965e-05 * T-2.76144075e-08 * T2+5.01618985e-12 * T3-3.11398731e-16 * T4+2.01964021e+04 * invT
;
// species 4: O
species[4] =
-2.61902026e+05 * invT2-7.29872203e+02 * logT * invT+2.31717727e+00 -2.14066718e-04 * T+3.45368198e-08 * T2-2.35957608e-12 * T3+5.45007659e-17 * T4+3.39242806e+04 * invT
;
// species 9: O2
species[9] =
+1.03793902e+06 * invT2+2.34483028e+03 * logT * invT+8.19732036e-01 +6.33923791e-04 * T-7.29355996e-08 * T2+5.13429893e-12 * T3-1.63869341e-16 * T4-1.68901093e+04 * invT
;
}
else {
// species 3: OH
species[3] =
-2.84723419e+08 * invT2-1.85953261e+05 * logT * invT+4.90824090e+01 -2.57118749e-03 * T+9.58512196e-08 * T2-2.05720449e-12 * T3+1.91344580e-17 * T4+1.46839391e+06 * invT
;
// species 4: O
species[4] =
-1.77900426e+08 * invT2-1.08232826e+05 * logT * invT+2.71077836e+01 -1.48761613e-03 * T+6.18332511e-08 * T2-1.44905788e-12 * T3+1.43834403e-17 * T4+8.89094263e+05 * invT
;
// species 9: O2
species[9] =
-4.97529430e+08 * invT2-2.86610687e+05 * logT * invT+6.59035225e+01 -3.08497951e-03 * T+1.00546534e-07 * T2-1.85535415e-12 * T3+1.45563515e-17 * T4+2.29355403e+06 * invT
;
}
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void speciesEnthalpy(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
+2.50000000e+00 -7.45375000e+02 * invT
;
// species 1: H2
species[1] =
-4.07832321e+04 * invT2-8.00918604e+02 * logT * invT+8.21470201e+00 -6.34857229e-03 * T+5.84535025e-06 * T2-3.00715068e-09 * T3+6.73618698e-13 * T4+2.68248466e+03 * invT
;
// species 2: H
species[2] =
+2.50000000e+00 +2.54737080e+04 * invT
;
// species 5: H2O
species[5] =
+3.94796083e+04 * invT2+5.75573102e+02 * logT * invT+9.31782653e-01 +3.61135643e-03 * T-2.44751912e-06 * T2+1.23876087e-09 * T3-2.67386649e-13 * T4-3.30397431e+04 * invT
;
// species 6: HO2
species[6] =
+7.59888254e+04 * invT2+1.32938392e+03 * logT * invT-4.67738824e+00 +1.25415410e-02 * T-1.00218386e-05 * T2+4.73900014e-09 * T3-9.65713478e-13 * T4-5.87335096e+03 * invT
;
// species 7: H2O2
species[7] =
+9.27953358e+04 * invT2+1.56474839e+03 * logT * invT-5.97646014e+00 +1.63537226e-02 * T-1.31073109e-05 * T2+6.27313809e-09 * T3-1.29300906e-12 * T4-2.49400473e+04 * invT
;
// species 8: N2
species[8] =
-2.21037150e+04 * invT2-3.81846182e+02 * logT * invT+6.08273836e+00 -4.26545720e-03 * T+4.61548730e-06 * T2-2.40644840e-09 * T3+5.03941162e-13 * T4+7.10846086e+02 * invT
;
}
else {
// species 0: XE
species[0] =
-4.02522668e+03 * invT2-1.20950752e+01 * logT * invT+2.51415335e+00 -4.12405104e-06 * T+8.43410873e-10 * T2-9.73083308e-14 * T3+4.72087828e-18 * T4-6.68580073e+02 * invT
;
// species 1: H2
species[1] =
-5.60812801e+05 * invT2-8.37150474e+02 * logT * invT+2.97536453e+00 +6.26124562e-04 * T-1.24690540e-07 * T2+1.48415630e-11 * T3-7.21398820e-16 * T4+5.33982441e+03 * invT
;
// species 2: H
species[2] =
-6.07877425e+01 * invT2-1.81935442e-01 * logT * invT+2.50021182e+00 -6.13256432e-08 * T+1.24429211e-11 * T2-1.42193614e-15 * T3+6.82042039e-20 * T4+2.54748640e+04 * invT
;
// species 5: H2O
species[5] =
-1.03497210e+06 * invT2-2.41269856e+03 * logT * invT+4.64611078e+00 +1.14599915e-03 * T-2.27894349e-07 * T2+2.35661723e-11 * T3-9.64476106e-16 * T4-1.38428651e+04 * invT
;
// species 6: HO2
species[6] =
+1.81066972e+06 * invT2+4.96319203e+03 * logT * invT-1.03949899e+00 +2.28007426e-03 * T-3.53953149e-07 * T2+2.86141969e-11 * T3-9.52612832e-16 * T4-3.20081719e+04 * invT
;
// species 7: H2O2
species[7] =
-1.48942803e+06 * invT2-5.17082178e+03 * logT * invT+1.12820497e+01 -4.02119890e-05 * T-6.06127923e-09 * T2+1.73681640e-12 * T3-9.65566380e-17 * T4+1.41825104e+04 * invT
;
// species 8: N2
species[8] =
-5.87712406e+05 * invT2-2.23924907e+03 * logT * invT+6.06694922e+00 -3.06984275e-04 * T+4.97268893e-08 * T2-4.80776371e-12 * T3+2.12390877e-16 * T4+1.28321041e+04 * invT
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+1.99885899e+03 * invT2+9.30013616e+01 * logT * invT+3.05085423e+00 +7.64764644e-04 * T-1.05263033e-06 * T2+8.28861545e-10 * T3-2.27752537e-13 * T4+2.99121423e+03 * invT
;
// species 4: O
species[4] =
+7.95361130e+03 * invT2+1.60717779e+02 * logT * invT+1.96622644e+00 +5.06835155e-04 * T-3.70138474e-07 * T2+1.62937687e-10 * T3-3.16955850e-14 * T4+2.84036244e+04 * invT
;
// species 9: O2
species[9] =
+3.42556342e+04 * invT2+4.84700097e+02 * logT * invT+1.11901096e+00 +2.14694462e-03 * T-2.27876684e-07 * T2-5.05843175e-10 * T3+2.07808004e-13 * T4-3.39145487e+03 * invT
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-1.01739338e+06 * invT2-2.50995728e+03 * logT * invT+5.11654786e+00 +6.52649965e-05 * T-2.76144075e-08 * T2+5.01618985e-12 * T3-3.11398731e-16 * T4+2.01964021e+04 * invT
;
// species 4: O
species[4] =
-2.61902026e+05 * invT2-7.29872203e+02 * logT * invT+3.31717727e+00 -2.14066718e-04 * T+3.45368198e-08 * T2-2.35957608e-12 * T3+5.45007659e-17 * T4+3.39242806e+04 * invT
;
// species 9: O2
species[9] =
+1.03793902e+06 * invT2+2.34483028e+03 * logT * invT+1.81973204e+00 +6.33923791e-04 * T-7.29355996e-08 * T2+5.13429893e-12 * T3-1.63869341e-16 * T4-1.68901093e+04 * invT
;
}
else {
// species 3: OH
species[3] =
-2.84723419e+08 * invT2-1.85953261e+05 * logT * invT+5.00824090e+01 -2.57118749e-03 * T+9.58512196e-08 * T2-2.05720449e-12 * T3+1.91344580e-17 * T4+1.46839391e+06 * invT
;
// species 4: O
species[4] =
-1.77900426e+08 * invT2-1.08232826e+05 * logT * invT+2.81077836e+01 -1.48761613e-03 * T+6.18332511e-08 * T2-1.44905788e-12 * T3+1.43834403e-17 * T4+8.89094263e+05 * invT
;
// species 9: O2
species[9] =
-4.97529430e+08 * invT2-2.86610687e+05 * logT * invT+6.69035225e+01 -3.08497951e-03 * T+1.00546534e-07 * T2-1.85535415e-12 * T3+1.45563515e-17 * T4+2.29355403e+06 * invT
;
}
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void speciesEntropy(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
+2.50000000e+00 * logT+6.16445421e+00 
;
// species 1: H2
species[1] =
-2.03916161e+04 * invT2+8.00918604e+02 * invT+8.21470201e+00 * logT-1.26971446e-02 * T+8.76802538e-06 * T2-4.00953423e-09 * T3+8.42023372e-13 * T4-3.04378884e+01 
;
// species 2: H
species[2] =
+2.50000000e+00 * logT-4.46682853e-01 
;
// species 5: H2O
species[5] =
+1.97398041e+04 * invT2-5.75573102e+02 * invT+9.31782653e-01 * logT+7.22271286e-03 * T-3.67127868e-06 * T2+1.65168116e-09 * T3-3.34233312e-13 * T4+1.72420578e+01 
;
// species 6: HO2
species[6] =
+3.79944127e+04 * invT2-1.32938392e+03 * invT-4.67738824e+00 * logT+2.50830820e-02 * T-1.50327579e-05 * T2+6.31866685e-09 * T3-1.20714185e-12 * T4+5.19360214e+01 
;
// species 7: H2O2
species[7] =
+4.63976679e+04 * invT2-1.56474839e+03 * invT-5.97646014e+00 * logT+3.27074452e-02 * T-1.96609663e-05 * T2+8.36418412e-09 * T3-1.61626132e-12 * T4+5.87717418e+01 
;
// species 8: N2
species[8] =
-1.10518575e+04 * invT2+3.81846182e+02 * invT+6.08273836e+00 * logT-8.53091441e-03 * T+6.92323094e-06 * T2-3.20859787e-09 * T3+6.29926452e-13 * T4-1.07600374e+01 
;
}
else {
// species 0: XE
species[0] =
-2.01261334e+03 * invT2+1.20950752e+01 * invT+2.51415335e+00 * logT-8.24810208e-06 * T+1.26511631e-09 * T2-1.29744441e-13 * T3+5.90109784e-18 * T4+6.06371072e+00 
;
// species 1: H2
species[1] =
-2.80406400e+05 * invT2+8.37150474e+02 * invT+2.97536453e+00 * logT+1.25224912e-03 * T-1.87035810e-07 * T2+1.97887507e-11 * T3-9.01748525e-16 * T4-2.20277477e+00 
;
// species 2: H
species[2] =
-3.03938713e+01 * invT2+1.81935442e-01 * invT+2.50021182e+00 * logT-1.22651286e-07 * T+1.86643817e-11 * T2-1.89591485e-15 * T3+8.52552549e-20 * T4-4.48191777e-01 
;
// species 5: H2O
species[5] =
-5.17486048e+05 * invT2+2.41269856e+03 * invT+4.64611078e+00 * logT+2.29199831e-03 * T-3.41841524e-07 * T2+3.14215631e-11 * T3-1.20559513e-15 * T4-7.97814851e+00 
;
// species 6: HO2
species[6] =
+9.05334862e+05 * invT2-4.96319203e+03 * invT-1.03949899e+00 * logT+4.56014853e-03 * T-5.30929723e-07 * T2+3.81522626e-11 * T3-1.19076604e-15 * T4+4.06685092e+01 
;
// species 7: H2O2
species[7] =
-7.44714014e+05 * invT2+5.17082178e+03 * invT+1.12820497e+01 * logT-8.04239779e-05 * T-9.09191885e-09 * T2+2.31575520e-12 * T3-1.20695798e-16 * T4-4.65085566e+01 
;
// species 8: N2
species[8] =
-2.93856203e+05 * invT2+2.23924907e+03 * invT+6.06694922e+00 * logT-6.13968550e-04 * T+7.45903340e-08 * T2-6.41035162e-12 * T3+2.65488596e-16 * T4-1.58664003e+01 
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+9.99429495e+02 * invT2-9.30013616e+01 * invT+3.05085423e+00 * logT+1.52952929e-03 * T-1.57894550e-06 * T2+1.10514873e-09 * T3-2.84690671e-13 * T4+4.67411079e+00 
;
// species 4: O
species[4] =
+3.97680565e+03 * invT2-1.60717779e+02 * invT+1.96622644e+00 * logT+1.01367031e-03 * T-5.55207711e-07 * T2+2.17250250e-10 * T3-3.96194813e-14 * T4+8.40424182e+00 
;
// species 9: O2
species[9] =
+1.71278171e+04 * invT2-4.84700097e+02 * invT+1.11901096e+00 * logT+4.29388924e-03 * T-3.41815026e-07 * T2-6.74457567e-10 * T3+2.59760005e-13 * T4+1.84969947e+01 
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-5.08696689e+05 * invT2+2.50995728e+03 * invT+5.11654786e+00 * logT+1.30529993e-04 * T-4.14216113e-08 * T2+6.68825314e-12 * T3-3.89248414e-16 * T4-1.10128234e+01 
;
// species 4: O
species[4] =
-1.30951013e+05 * invT2+7.29872203e+02 * invT+3.31717727e+00 * logT-4.28133436e-04 * T+5.18052297e-08 * T2-3.14610144e-12 * T3+6.81259574e-17 * T4-6.67958535e-01 
;
// species 9: O2
species[9] =
+5.18969511e+05 * invT2-2.34483028e+03 * invT+1.81973204e+00 * logT+1.26784758e-03 * T-1.09403399e-07 * T2+6.84573191e-12 * T3-2.04836676e-16 * T4+1.73871651e+01 
;
}
else {
// species 3: OH
species[3] =
-1.42361710e+08 * invT2+1.85953261e+05 * invT+5.00824090e+01 * logT-5.14237498e-03 * T+1.43776829e-07 * T2-2.74293932e-12 * T3+2.39180725e-17 * T4-4.02355558e+02 
;
// species 4: O
species[4] =
-8.89502132e+07 * invT2+1.08232826e+05 * invT+2.81077836e+01 * logT-2.97523226e-03 * T+9.27498767e-08 * T2-1.93207718e-12 * T3+1.79793004e-17 * T4-2.18172815e+02 
;
// species 9: O2
species[9] =
-2.48764715e+08 * invT2+2.86610687e+05 * invT+6.69035225e+01 * logT-6.16995902e-03 * T+1.50819801e-07 * T2-2.47380553e-12 * T3+1.81954394e-17 * T4-5.53062161e+02 
;
}
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dcvpRdT(amrex::Real * species, const amrex::Real T)
{
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real invT3 = invT*invT*invT;


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
species[0] =
0.0
;
// species 1: H2
species[1] =
-8.15664642e+04 * invT3+8.00918604e+02 * invT2-1.26971446e-02 +3.50721015e-05 * T-3.60858081e-08 * T2+1.34723740e-11 * T3
;
// species 2: H
species[2] =
0.0
;
// species 5: H2O
species[5] =
+7.89592166e+04 * invT3-5.75573102e+02 * invT2+7.22271286e-03 -1.46851147e-05 * T+1.48651305e-08 * T2-5.34773298e-12 * T3
;
// species 6: HO2
species[6] =
+1.51977651e+05 * invT3-1.32938392e+03 * invT2+2.50830820e-02 -6.01310318e-05 * T+5.68680017e-08 * T2-1.93142696e-11 * T3
;
// species 7: H2O2
species[7] =
+1.85590672e+05 * invT3-1.56474839e+03 * invT2+3.27074452e-02 -7.86438652e-05 * T+7.52776571e-08 * T2-2.58601812e-11 * T3
;
// species 8: N2
species[8] =
-4.42074299e+04 * invT3+3.81846182e+02 * invT2-8.53091441e-03 +2.76929238e-05 * T-2.88773809e-08 * T2+1.00788232e-11 * T3
;
}
else {
// species 0: XE
species[0] =
-8.05045336e+03 * invT3+1.20950752e+01 * invT2-8.24810208e-06 +5.06046524e-09 * T-1.16769997e-12 * T2+9.44175655e-17 * T3
;
// species 1: H2
species[1] =
-1.12162560e+06 * invT3+8.37150474e+02 * invT2+1.25224912e-03 -7.48143238e-07 * T+1.78098756e-10 * T2-1.44279764e-14 * T3
;
// species 2: H
species[2] =
-1.21575485e+02 * invT3+1.81935442e-01 * invT2-1.22651286e-07 +7.46575266e-11 * T-1.70632337e-14 * T2+1.36408408e-18 * T3
;
// species 5: H2O
species[5] =
-2.06994419e+06 * invT3+2.41269856e+03 * invT2+2.29199831e-03 -1.36736610e-06 * T+2.82794068e-10 * T2-1.92895221e-14 * T3
;
// species 6: HO2
species[6] =
+3.62133945e+06 * invT3-4.96319203e+03 * invT2+4.56014853e-03 -2.12371889e-06 * T+3.43370363e-10 * T2-1.90522566e-14 * T3
;
// species 7: H2O2
species[7] =
-2.97885605e+06 * invT3+5.17082178e+03 * invT2-8.04239779e-05 -3.63676754e-08 * T+2.08417968e-11 * T2-1.93113276e-15 * T3
;
// species 8: N2
species[8] =
-1.17542481e+06 * invT3+2.23924907e+03 * invT2-6.13968550e-04 +2.98361336e-07 * T-5.76931646e-11 * T2+4.24781754e-15 * T3
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
species[3] =
+3.99771798e+03 * invT3-9.30013616e+01 * invT2+1.52952929e-03 -6.31578200e-06 * T+9.94633854e-09 * T2-4.55505073e-12 * T3
;
// species 4: O
species[4] =
+1.59072226e+04 * invT3-1.60717779e+02 * invT2+1.01367031e-03 -2.22083085e-06 * T+1.95525225e-09 * T2-6.33911700e-13 * T3
;
// species 9: O2
species[9] =
+6.85112684e+04 * invT3-4.84700097e+02 * invT2+4.29388924e-03 -1.36726010e-06 * T-6.07011810e-09 * T2+4.15616007e-12 * T3
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
species[3] =
-2.03478676e+06 * invT3+2.50995728e+03 * invT2+1.30529993e-04 -1.65686445e-07 * T+6.01942782e-11 * T2-6.22797462e-15 * T3
;
// species 4: O
species[4] =
-5.23804052e+05 * invT3+7.29872203e+02 * invT2-4.28133436e-04 +2.07220919e-07 * T-2.83149130e-11 * T2+1.09001532e-15 * T3
;
// species 9: O2
species[9] =
+2.07587804e+06 * invT3-2.34483028e+03 * invT2+1.26784758e-03 -4.37613598e-07 * T+6.16115872e-11 * T2-3.27738682e-15 * T3
;
}
else {
// species 3: OH
species[3] =
-5.69446839e+08 * invT3+1.85953261e+05 * invT2-5.14237498e-03 +5.75107318e-07 * T-2.46864539e-11 * T2+3.82689161e-16 * T3
;
// species 4: O
species[4] =
-3.55800853e+08 * invT3+1.08232826e+05 * invT2-2.97523226e-03 +3.70999507e-07 * T-1.73886946e-11 * T2+2.87668807e-16 * T3
;
// species 9: O2
species[9] =
-9.95058860e+08 * invT3+2.86610687e+05 * invT2-6.16995902e-03 +6.03279205e-07 * T-2.22642498e-11 * T2+2.91127031e-16 * T3
;
}
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
amrex::Real result = 0; 
amrex::Real cpor[10]; //  temporary storage
cp_R(cpor, T);

// perform dot product
for (int id = 0; id < 10; ++id) {
result += x[id]*cpor[id];
}

cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
amrex::Real result = 0.0; 

// compute Cp/R at the given temperature
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
result += y[0] * (
+2.50000000e+00 
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
+4.07832321e+04 * invT2-8.00918604e+02 * invT+8.21470201e+00 -1.26971446e-02 * T+1.75360508e-05  * T2-1.20286027e-08 * T3+3.36809349e-12 * T4
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+2.50000000e+00 
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
-3.94796083e+04 * invT2+5.75573102e+02 * invT+9.31782653e-01 +7.22271286e-03 * T-7.34255737e-06  * T2+4.95504349e-09 * T3-1.33693325e-12 * T4
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
-7.59888254e+04 * invT2+1.32938392e+03 * invT-4.67738824e+00 +2.50830820e-02 * T-3.00655159e-05  * T2+1.89560006e-08 * T3-4.82856739e-12 * T4
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
-9.27953358e+04 * invT2+1.56474839e+03 * invT-5.97646014e+00 +3.27074452e-02 * T-3.93219326e-05  * T2+2.50925524e-08 * T3-6.46504529e-12 * T4
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
+2.21037150e+04 * invT2-3.81846182e+02 * invT+6.08273836e+00 -8.53091441e-03 * T+1.38464619e-05  * T2-9.62579362e-09 * T3+2.51970581e-12 * T4
)* 0.0356964374955379
;
}
else {
// species 0: XE
result += y[0] * (
+4.02522668e+03 * invT2-1.20950752e+01 * invT+2.51415335e+00 -8.24810208e-06 * T+2.53023262e-09  * T2-3.89233323e-13 * T3+2.36043914e-17 * T4
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
+5.60812801e+05 * invT2-8.37150474e+02 * invT+2.97536453e+00 +1.25224912e-03 * T-3.74071619e-07  * T2+5.93662520e-11 * T3-3.60699410e-15 * T4
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+6.07877425e+01 * invT2-1.81935442e-01 * invT+2.50021182e+00 -1.22651286e-07 * T+3.73287633e-11  * T2-5.68774456e-15 * T3+3.41021020e-19 * T4
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
+1.03497210e+06 * invT2-2.41269856e+03 * invT+4.64611078e+00 +2.29199831e-03 * T-6.83683048e-07  * T2+9.42646893e-11 * T3-4.82238053e-15 * T4
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
-1.81066972e+06 * invT2+4.96319203e+03 * invT-1.03949899e+00 +4.56014853e-03 * T-1.06185945e-06  * T2+1.14456788e-10 * T3-4.76306416e-15 * T4
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
+1.48942803e+06 * invT2-5.17082178e+03 * invT+1.12820497e+01 -8.04239779e-05 * T-1.81838377e-08  * T2+6.94726559e-12 * T3-4.82783190e-16 * T4
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
+5.87712406e+05 * invT2-2.23924907e+03 * invT+6.06694922e+00 -6.13968550e-04 * T+1.49180668e-07  * T2-1.92310549e-11 * T3+1.06195439e-15 * T4
)* 0.0356964374955379
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
result += y[3] * (
-1.99885899e+03 * invT2+9.30013616e+01 * invT+3.05085423e+00 +1.52952929e-03 * T-3.15789100e-06  * T2+3.31544618e-09 * T3-1.13876268e-12 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-7.95361130e+03 * invT2+1.60717779e+02 * invT+1.96622644e+00 +1.01367031e-03 * T-1.11041542e-06  * T2+6.51750750e-10 * T3-1.58477925e-13 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-3.42556342e+04 * invT2+4.84700097e+02 * invT+1.11901096e+00 +4.29388924e-03 * T-6.83630052e-07  * T2-2.02337270e-09 * T3+1.03904002e-12 * T4
)* 0.0312519532470779
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
result += y[3] * (
+1.01739338e+06 * invT2-2.50995728e+03 * invT+5.11654786e+00 +1.30529993e-04 * T-8.28432226e-08  * T2+2.00647594e-11 * T3-1.55699366e-15 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+2.61902026e+05 * invT2-7.29872203e+02 * invT+3.31717727e+00 -4.28133436e-04 * T+1.03610459e-07  * T2-9.43830433e-12 * T3+2.72503830e-16 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-1.03793902e+06 * invT2+2.34483028e+03 * invT+1.81973204e+00 +1.26784758e-03 * T-2.18806799e-07  * T2+2.05371957e-11 * T3-8.19346705e-16 * T4
)* 0.0312519532470779
;
}
else {
// species 3: OH
result += y[3] * (
+2.84723419e+08 * invT2-1.85953261e+05 * invT+5.00824090e+01 -5.14237498e-03 * T+2.87553659e-07  * T2-8.22881796e-12 * T3+9.56722902e-17 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+1.77900426e+08 * invT2-1.08232826e+05 * invT+2.81077836e+01 -2.97523226e-03 * T+1.85499753e-07  * T2-5.79623154e-12 * T3+7.19172016e-17 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+4.97529430e+08 * invT2-2.86610687e+05 * invT+6.69035225e+01 -6.16995902e-03 * T+3.01639603e-07  * T2-7.42141660e-12 * T3+7.27817577e-17 * T4
)* 0.0312519532470779
;
}

cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
amrex::Real result = 0; 
amrex::Real cvor[10]; //  temporary storage
cv_R(cvor, T);

// perform dot product
for (int id = 0; id < 10; ++id) {
result += x[id]*cvor[id];
}

cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVBS(const amrex::Real T, const amrex::Real y[],  amrex::Real& cvbs)
{
amrex::Real result = 0.0; 
// compute Cv/R at the given temperature
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
result += y[0] * (
+1.50000000e+00 
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
+4.07832321e+04 * invT2-8.00918604e+02 * invT+7.21470201e+00 -1.26971446e-02 * T+1.75360508e-05  * T2-1.20286027e-08 * T3+3.36809349e-12 * T4
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+1.50000000e+00 
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
-3.94796083e+04 * invT2+5.75573102e+02 * invT-6.82173470e-02 +7.22271286e-03 * T-7.34255737e-06  * T2+4.95504349e-09 * T3-1.33693325e-12 * T4
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
-7.59888254e+04 * invT2+1.32938392e+03 * invT-5.67738824e+00 +2.50830820e-02 * T-3.00655159e-05  * T2+1.89560006e-08 * T3-4.82856739e-12 * T4
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
-9.27953358e+04 * invT2+1.56474839e+03 * invT-6.97646014e+00 +3.27074452e-02 * T-3.93219326e-05  * T2+2.50925524e-08 * T3-6.46504529e-12 * T4
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
+2.21037150e+04 * invT2-3.81846182e+02 * invT+5.08273836e+00 -8.53091441e-03 * T+1.38464619e-05  * T2-9.62579362e-09 * T3+2.51970581e-12 * T4
)* 0.0356964374955379
;
}
else {
// species 0: XE
result += y[0] * (
+4.02522668e+03 * invT2-1.20950752e+01 * invT+1.51415335e+00 -8.24810208e-06 * T+2.53023262e-09  * T2-3.89233323e-13 * T3+2.36043914e-17 * T4
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
+5.60812801e+05 * invT2-8.37150474e+02 * invT+1.97536453e+00 +1.25224912e-03 * T-3.74071619e-07  * T2+5.93662520e-11 * T3-3.60699410e-15 * T4
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+6.07877425e+01 * invT2-1.81935442e-01 * invT+1.50021182e+00 -1.22651286e-07 * T+3.73287633e-11  * T2-5.68774456e-15 * T3+3.41021020e-19 * T4
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
+1.03497210e+06 * invT2-2.41269856e+03 * invT+3.64611078e+00 +2.29199831e-03 * T-6.83683048e-07  * T2+9.42646893e-11 * T3-4.82238053e-15 * T4
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
-1.81066972e+06 * invT2+4.96319203e+03 * invT-2.03949899e+00 +4.56014853e-03 * T-1.06185945e-06  * T2+1.14456788e-10 * T3-4.76306416e-15 * T4
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
+1.48942803e+06 * invT2-5.17082178e+03 * invT+1.02820497e+01 -8.04239779e-05 * T-1.81838377e-08  * T2+6.94726559e-12 * T3-4.82783190e-16 * T4
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
+5.87712406e+05 * invT2-2.23924907e+03 * invT+5.06694922e+00 -6.13968550e-04 * T+1.49180668e-07  * T2-1.92310549e-11 * T3+1.06195439e-15 * T4
)* 0.0356964374955379
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
result += y[3] * (
-1.99885899e+03 * invT2+9.30013616e+01 * invT+2.05085423e+00 +1.52952929e-03 * T-3.15789100e-06  * T2+3.31544618e-09 * T3-1.13876268e-12 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-7.95361130e+03 * invT2+1.60717779e+02 * invT+9.66226438e-01 +1.01367031e-03 * T-1.11041542e-06  * T2+6.51750750e-10 * T3-1.58477925e-13 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-3.42556342e+04 * invT2+4.84700097e+02 * invT+1.19010961e-01 +4.29388924e-03 * T-6.83630052e-07  * T2-2.02337270e-09 * T3+1.03904002e-12 * T4
)* 0.0312519532470779
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
result += y[3] * (
+1.01739338e+06 * invT2-2.50995728e+03 * invT+4.11654786e+00 +1.30529993e-04 * T-8.28432226e-08  * T2+2.00647594e-11 * T3-1.55699366e-15 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+2.61902026e+05 * invT2-7.29872203e+02 * invT+2.31717727e+00 -4.28133436e-04 * T+1.03610459e-07  * T2-9.43830433e-12 * T3+2.72503830e-16 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-1.03793902e+06 * invT2+2.34483028e+03 * invT+8.19732036e-01 +1.26784758e-03 * T-2.18806799e-07  * T2+2.05371957e-11 * T3-8.19346705e-16 * T4
)* 0.0312519532470779
;
}
else {
// species 3: OH
result += y[3] * (
+2.84723419e+08 * invT2-1.85953261e+05 * invT+4.90824090e+01 -5.14237498e-03 * T+2.87553659e-07  * T2-8.22881796e-12 * T3+9.56722902e-17 * T4
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+1.77900426e+08 * invT2-1.08232826e+05 * invT+2.71077836e+01 -2.97523226e-03 * T+1.85499753e-07  * T2-5.79623154e-12 * T3+7.19172016e-17 * T4
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+4.97529430e+08 * invT2-2.86610687e+05 * invT+6.59035225e+01 -6.16995902e-03 * T+3.01639603e-07  * T2-7.42141660e-12 * T3+7.27817577e-17 * T4
)* 0.0312519532470779
;
}

cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
amrex::Real result = 0; 
amrex::Real hml[10]; //  temporary storage
amrex::Real RT = 8.31446261815324e+07*T; // R*T
speciesEnthalpy(hml, T);

// perform dot product
for (int id = 0; id < 10; ++id) {
result += x[id]*hml[id];
}

hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHBMS(const amrex::Real T, const amrex::Real y[],  amrex::Real& hbms)
{
amrex::Real result = 0.0;
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
result += y[0] * (
+2.50000000e+00 -7.45375000e+02 * invT
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
-4.07832321e+04 * invT2-8.00918604e+02 * logT * invT+8.21470201e+00 -6.34857229e-03 * T+5.84535025e-06 * T2-3.00715068e-09 * T3+6.73618698e-13 * T4+2.68248466e+03 * invT
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+2.50000000e+00 +2.54737080e+04 * invT
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
+3.94796083e+04 * invT2+5.75573102e+02 * logT * invT+9.31782653e-01 +3.61135643e-03 * T-2.44751912e-06 * T2+1.23876087e-09 * T3-2.67386649e-13 * T4-3.30397431e+04 * invT
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
+7.59888254e+04 * invT2+1.32938392e+03 * logT * invT-4.67738824e+00 +1.25415410e-02 * T-1.00218386e-05 * T2+4.73900014e-09 * T3-9.65713478e-13 * T4-5.87335096e+03 * invT
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
+9.27953358e+04 * invT2+1.56474839e+03 * logT * invT-5.97646014e+00 +1.63537226e-02 * T-1.31073109e-05 * T2+6.27313809e-09 * T3-1.29300906e-12 * T4-2.49400473e+04 * invT
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
-2.21037150e+04 * invT2-3.81846182e+02 * logT * invT+6.08273836e+00 -4.26545720e-03 * T+4.61548730e-06 * T2-2.40644840e-09 * T3+5.03941162e-13 * T4+7.10846086e+02 * invT
)* 0.0356964374955379
;
}
else {
// species 0: XE
result += y[0] * (
-4.02522668e+03 * invT2-1.20950752e+01 * logT * invT+2.51415335e+00 -4.12405104e-06 * T+8.43410873e-10 * T2-9.73083308e-14 * T3+4.72087828e-18 * T4-6.68580073e+02 * invT
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
-5.60812801e+05 * invT2-8.37150474e+02 * logT * invT+2.97536453e+00 +6.26124562e-04 * T-1.24690540e-07 * T2+1.48415630e-11 * T3-7.21398820e-16 * T4+5.33982441e+03 * invT
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
-6.07877425e+01 * invT2-1.81935442e-01 * logT * invT+2.50021182e+00 -6.13256432e-08 * T+1.24429211e-11 * T2-1.42193614e-15 * T3+6.82042039e-20 * T4+2.54748640e+04 * invT
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
-1.03497210e+06 * invT2-2.41269856e+03 * logT * invT+4.64611078e+00 +1.14599915e-03 * T-2.27894349e-07 * T2+2.35661723e-11 * T3-9.64476106e-16 * T4-1.38428651e+04 * invT
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
+1.81066972e+06 * invT2+4.96319203e+03 * logT * invT-1.03949899e+00 +2.28007426e-03 * T-3.53953149e-07 * T2+2.86141969e-11 * T3-9.52612832e-16 * T4-3.20081719e+04 * invT
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
-1.48942803e+06 * invT2-5.17082178e+03 * logT * invT+1.12820497e+01 -4.02119890e-05 * T-6.06127923e-09 * T2+1.73681640e-12 * T3-9.65566380e-17 * T4+1.41825104e+04 * invT
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
-5.87712406e+05 * invT2-2.23924907e+03 * logT * invT+6.06694922e+00 -3.06984275e-04 * T+4.97268893e-08 * T2-4.80776371e-12 * T3+2.12390877e-16 * T4+1.28321041e+04 * invT
)* 0.0356964374955379
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
result += y[3] * (
+1.99885899e+03 * invT2+9.30013616e+01 * logT * invT+3.05085423e+00 +7.64764644e-04 * T-1.05263033e-06 * T2+8.28861545e-10 * T3-2.27752537e-13 * T4+2.99121423e+03 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+7.95361130e+03 * invT2+1.60717779e+02 * logT * invT+1.96622644e+00 +5.06835155e-04 * T-3.70138474e-07 * T2+1.62937687e-10 * T3-3.16955850e-14 * T4+2.84036244e+04 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+3.42556342e+04 * invT2+4.84700097e+02 * logT * invT+1.11901096e+00 +2.14694462e-03 * T-2.27876684e-07 * T2-5.05843175e-10 * T3+2.07808004e-13 * T4-3.39145487e+03 * invT
)* 0.0312519532470779
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
result += y[3] * (
-1.01739338e+06 * invT2-2.50995728e+03 * logT * invT+5.11654786e+00 +6.52649965e-05 * T-2.76144075e-08 * T2+5.01618985e-12 * T3-3.11398731e-16 * T4+2.01964021e+04 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-2.61902026e+05 * invT2-7.29872203e+02 * logT * invT+3.31717727e+00 -2.14066718e-04 * T+3.45368198e-08 * T2-2.35957608e-12 * T3+5.45007659e-17 * T4+3.39242806e+04 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+1.03793902e+06 * invT2+2.34483028e+03 * logT * invT+1.81973204e+00 +6.33923791e-04 * T-7.29355996e-08 * T2+5.13429893e-12 * T3-1.63869341e-16 * T4-1.68901093e+04 * invT
)* 0.0312519532470779
;
}
else {
// species 3: OH
result += y[3] * (
-2.84723419e+08 * invT2-1.85953261e+05 * logT * invT+5.00824090e+01 -2.57118749e-03 * T+9.58512196e-08 * T2-2.05720449e-12 * T3+1.91344580e-17 * T4+1.46839391e+06 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-1.77900426e+08 * invT2-1.08232826e+05 * logT * invT+2.81077836e+01 -1.48761613e-03 * T+6.18332511e-08 * T2-1.44905788e-12 * T3+1.43834403e-17 * T4+8.89094263e+05 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-4.97529430e+08 * invT2-2.86610687e+05 * logT * invT+6.69035225e+01 -3.08497951e-03 * T+1.00546534e-07 * T2-1.85535415e-12 * T3+1.45563515e-17 * T4+2.29355403e+06 * invT
)* 0.0312519532470779
;
}

const amrex::Real RT = 8.31446261815324e+07*T; // R*T

hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
amrex::Real result = 0; 
amrex::Real uml[10]; //  temporary energy array
amrex::Real RT = 8.31446261815324e+07*T; // R*T
speciesInternalEnergy(uml, T);

// perform dot product
for (int id = 0; id < 10; ++id) {
result += x[id]*uml[id];
}

ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
amrex::Real result = 0.0;
const amrex::Real T2 = T*T;
const amrex::Real T3 = T*T*T;
const amrex::Real T4 = T*T*T*T;
const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT*invT;
const amrex::Real logT = log(T);


// species with midpoint at T=1000 kelvin
if (T < 1000) {
// species 0: XE
result += y[0] * (
+1.50000000e+00 -7.45375000e+02 * invT
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
-4.07832321e+04 * invT2-8.00918604e+02 * logT * invT+7.21470201e+00 -6.34857229e-03 * T+5.84535025e-06 * T2-3.00715068e-09 * T3+6.73618698e-13 * T4+2.68248466e+03 * invT
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
+1.50000000e+00 +2.54737080e+04 * invT
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
+3.94796083e+04 * invT2+5.75573102e+02 * logT * invT-6.82173470e-02 +3.61135643e-03 * T-2.44751912e-06 * T2+1.23876087e-09 * T3-2.67386649e-13 * T4-3.30397431e+04 * invT
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
+7.59888254e+04 * invT2+1.32938392e+03 * logT * invT-5.67738824e+00 +1.25415410e-02 * T-1.00218386e-05 * T2+4.73900014e-09 * T3-9.65713478e-13 * T4-5.87335096e+03 * invT
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
+9.27953358e+04 * invT2+1.56474839e+03 * logT * invT-6.97646014e+00 +1.63537226e-02 * T-1.31073109e-05 * T2+6.27313809e-09 * T3-1.29300906e-12 * T4-2.49400473e+04 * invT
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
-2.21037150e+04 * invT2-3.81846182e+02 * logT * invT+5.08273836e+00 -4.26545720e-03 * T+4.61548730e-06 * T2-2.40644840e-09 * T3+5.03941162e-13 * T4+7.10846086e+02 * invT
)* 0.0356964374955379
;
}
else {
// species 0: XE
result += y[0] * (
-4.02522668e+03 * invT2-1.20950752e+01 * logT * invT+1.51415335e+00 -4.12405104e-06 * T+8.43410873e-10 * T2-9.73083308e-14 * T3+4.72087828e-18 * T4-6.68580073e+02 * invT
)* 0.0076165522914398
;
// species 1: H2
result += y[1] * (
-5.60812801e+05 * invT2-8.37150474e+02 * logT * invT+1.97536453e+00 +6.26124562e-04 * T-1.24690540e-07 * T2+1.48415630e-11 * T3-7.21398820e-16 * T4+5.33982441e+03 * invT
)* 0.4960317460317460
;
// species 2: H
result += y[2] * (
-6.07877425e+01 * invT2-1.81935442e-01 * logT * invT+1.50021182e+00 -6.13256432e-08 * T+1.24429211e-11 * T2-1.42193614e-15 * T3+6.82042039e-20 * T4+2.54748640e+04 * invT
)* 0.9920634920634921
;
// species 5: H2O
result += y[5] * (
-1.03497210e+06 * invT2-2.41269856e+03 * logT * invT+3.64611078e+00 +1.14599915e-03 * T-2.27894349e-07 * T2+2.35661723e-11 * T3-9.64476106e-16 * T4-1.38428651e+04 * invT
)* 0.0555092978073827
;
// species 6: HO2
result += y[6] * (
+1.81066972e+06 * invT2+4.96319203e+03 * logT * invT-2.03949899e+00 +2.28007426e-03 * T-3.53953149e-07 * T2+2.86141969e-11 * T3-9.52612832e-16 * T4-3.20081719e+04 * invT
)* 0.0302975216627280
;
// species 7: H2O2
result += y[7] * (
-1.48942803e+06 * invT2-5.17082178e+03 * logT * invT+1.02820497e+01 -4.02119890e-05 * T-6.06127923e-09 * T2+1.73681640e-12 * T3-9.65566380e-17 * T4+1.41825104e+04 * invT
)* 0.0293996589639560
;
// species 8: N2
result += y[8] * (
-5.87712406e+05 * invT2-2.23924907e+03 * logT * invT+5.06694922e+00 -3.06984275e-04 * T+4.97268893e-08 * T2-4.80776371e-12 * T3+2.12390877e-16 * T4+1.28321041e+04 * invT
)* 0.0356964374955379
;
}

// species with inflection points at T = (1000.0, 6000.0) kelvin
if (T < 1000) {
// species 3: OH
result += y[3] * (
+1.99885899e+03 * invT2+9.30013616e+01 * logT * invT+2.05085423e+00 +7.64764644e-04 * T-1.05263033e-06 * T2+8.28861545e-10 * T3-2.27752537e-13 * T4+2.99121423e+03 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
+7.95361130e+03 * invT2+1.60717779e+02 * logT * invT+9.66226438e-01 +5.06835155e-04 * T-3.70138474e-07 * T2+1.62937687e-10 * T3-3.16955850e-14 * T4+2.84036244e+04 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+3.42556342e+04 * invT2+4.84700097e+02 * logT * invT+1.19010961e-01 +2.14694462e-03 * T-2.27876684e-07 * T2-5.05843175e-10 * T3+2.07808004e-13 * T4-3.39145487e+03 * invT
)* 0.0312519532470779
;
}
else if ( (1000 <= T) && (T < 6000)) {
// species 3: OH
result += y[3] * (
-1.01739338e+06 * invT2-2.50995728e+03 * logT * invT+4.11654786e+00 +6.52649965e-05 * T-2.76144075e-08 * T2+5.01618985e-12 * T3-3.11398731e-16 * T4+2.01964021e+04 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-2.61902026e+05 * invT2-7.29872203e+02 * logT * invT+2.31717727e+00 -2.14066718e-04 * T+3.45368198e-08 * T2-2.35957608e-12 * T3+5.45007659e-17 * T4+3.39242806e+04 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
+1.03793902e+06 * invT2+2.34483028e+03 * logT * invT+8.19732036e-01 +6.33923791e-04 * T-7.29355996e-08 * T2+5.13429893e-12 * T3-1.63869341e-16 * T4-1.68901093e+04 * invT
)* 0.0312519532470779
;
}
else {
// species 3: OH
result += y[3] * (
-2.84723419e+08 * invT2-1.85953261e+05 * logT * invT+4.90824090e+01 -2.57118749e-03 * T+9.58512196e-08 * T2-2.05720449e-12 * T3+1.91344580e-17 * T4+1.46839391e+06 * invT
)* 0.0587993179279120
;
// species 4: O
result += y[4] * (
-1.77900426e+08 * invT2-1.08232826e+05 * logT * invT+2.71077836e+01 -1.48761613e-03 * T+6.18332511e-08 * T2-1.44905788e-12 * T3+1.43834403e-17 * T4+8.89094263e+05 * invT
)* 0.0625039064941559
;
// species 9: O2
result += y[9] * (
-4.97529430e+08 * invT2-2.86610687e+05 * logT * invT+6.59035225e+01 -3.08497951e-03 * T+1.00546534e-07 * T2-1.85535415e-12 * T3+1.45563515e-17 * T4+2.29355403e+06 * invT
)* 0.0312519532470779
;
}

const amrex::Real RT = 8.31446261815324e+07*T; // R*T

ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSBML(const amrex::Real P, const amrex::Real T,const amrex::Real x[], amrex::Real& sbml)
{
amrex::Real result = 0; 
// Log of normalized pressure in cgs units dynes/cm^2 by Patm
amrex::Real logPratio = log ( P / 1013250.0 ); 
amrex::Real sor[10]; //  temporary storage
speciesEntropy(sor, T);

// Compute Eq 42
for (int id = 0; id < 10; ++id) {
result += x[id]*(sor[id]-log((x[id]+1e-100))-logPratio);
}

sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSBMS(const amrex::Real P, const amrex::Real T,const amrex::Real y[], amrex::Real& sbms)
{
amrex::Real result = 0; 
// Log of normalized pressure in cgs units dynes/cm^2 by Patm
amrex::Real logPratio = log ( P / 1013250.0 ); 
amrex::Real sor[10]; //  temporary storage
amrex::Real x[10]; //  need a ytx conversion
amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

// Compute inverse of mean molecular wt first
for (int i = 0; i < 10; i++)
{
YOW += y[i]* imw(i);
}
// Now compute y to x conversion
x[0] = y[0]/(131.293000*YOW); 
x[1] = y[1]/(2.016000*YOW); 
x[2] = y[2]/(1.008000*YOW); 
x[3] = y[3]/(17.007000*YOW); 
x[4] = y[4]/(15.999000*YOW); 
x[5] = y[5]/(18.015000*YOW); 
x[6] = y[6]/(33.006000*YOW); 
x[7] = y[7]/(34.014000*YOW); 
x[8] = y[8]/(28.014000*YOW); 
x[9] = y[9]/(31.998000*YOW); 
speciesEntropy(sor, T);
// Perform computation in Eq 42 and 43
for (int i = 0; i < 10; i++)
{
result += x[i]*(sor[i]-log((x[i]+1e-100))-logPratio);
}
// Scale by R/W
sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void GET_T_GIVEN_EY(const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
const int maxiter = 5000;
const amrex::Real tol = 1.e-12;
#else
const int maxiter = 200;
const amrex::Real tol = 1.e-6;
#endif
amrex::Real tmin = 90;// max lower bound for thermo def
amrex::Real tmax = 4000;// min upper bound for thermo def
amrex::Real e1,emin,emax,cv,t1,dt;
CKUBMS(tmin, y, emin);
CKUBMS(tmax, y, emax);
if (e < emin) {
// Linear Extrapolation below tmin
CKCVBS(tmin, y, cv);
t = tmin - (emin-e)/cv;
ierr = 1;
return;
}
if (e > emax) {
// Linear Extrapolation above tmax
CKCVBS(tmax, y, cv);
t = tmax - (emax-e)/cv;
ierr = 1;
return;
}
t1 = t;
if (t1 < tmin || t1 > tmax) {
t1 = tmin + (tmax-tmin)/(emax-emin)*(e-emin);
}
for (int i = 0; i < maxiter; ++i) {
CKUBMS(t1,y,e1);
CKCVBS(t1,y,cv);
dt = (e - e1) / cv;
if (dt > 100.) { dt = 100.; }
else if (dt < -100.) { dt = -100.; }
else if (fabs(dt) < tol) {break;}
t1 += dt;
}
t = t1;
ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void GET_T_GIVEN_HY(const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
const int maxiter = 5000;
const amrex::Real tol = 1.e-12;
#else
const int maxiter = 200;
const amrex::Real tol = 1.e-6;
#endif
amrex::Real tmin = 90;// max lower bound for thermo def
amrex::Real tmax = 4000;// min upper bound for thermo def
amrex::Real h1,hmin,hmax,cp,t1,dt;
CKHBMS(tmin, y, hmin);
CKHBMS(tmax, y, hmax);
if (h < hmin) {
// Linear Extrapolation below tmin
CKCPBS(tmin, y, cp);
t = tmin - (hmin-h)/cp;
ierr = 1;
return;
}
if (h > hmax) {
// Linear Extrapolation above tmax
CKCPBS(tmax, y, cp);
t = tmax - (hmax-h)/cp;
ierr = 1;
return;
}
t1 = t;
if (t1 < tmin || t1 > tmax) {
t1 = tmin + (tmax-tmin)/(hmax-hmin)*(h-hmin);
}
for (int i = 0; i < maxiter; ++i) {
CKHBMS(t1,y,h1);
CKCPBS(t1,y,cp);
dt = (h - h1) / cp;
if (dt > 100.) { dt = 100.; }
else if (dt < -100.) { dt = -100.; }
else if (fabs(dt) < tol) {break;}
t1 += dt;
}
t = t1;
ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPX(const amrex::Real rho, const amrex::Real T, const amrex::Real x[], amrex::Real& P)
{
amrex::Real XW = 0;//  To hold mean molecular wt
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W

}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPY(const amrex::Real rho, const amrex::Real T, const amrex::Real y[],  amrex::Real& P)
{
amrex::Real YOW = 0;//  for computing mean MW

for (int i = 0; i < 10; i++)
{
YOW += y[i]*imw(i);
}
P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W

}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPC(const amrex::Real rho, const amrex::Real T, const amrex::Real c[],  amrex::Real& P)
{
// See Eq 5 in CK Manual
amrex::Real W = 0;
amrex::Real sumC = 0;
W += c[0]*131.293000; // XE
W += c[1]*2.016000; // H2
W += c[2]*1.008000; // H
W += c[3]*17.007000; // OH
W += c[4]*15.999000; // O
W += c[5]*18.015000; // H2O
W += c[6]*33.006000; // HO2
W += c[7]*34.014000; // H2O2
W += c[8]*28.014000; // N2
W += c[9]*31.998000; // O2

for (int id = 0; id < 10; ++id) {
sumC += c[id];
}
P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W

}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOX(const amrex::Real P, const amrex::Real T,const amrex::Real x[], amrex::Real& rho)
{
amrex::Real XW = 0;//  To hold mean molecular wt
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)

}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOY(const amrex::Real P, const amrex::Real T,const amrex::Real y[], amrex::Real& rho)
{
amrex::Real YOW = 0;

for (int i = 0; i < 10; i++)
{
YOW += y[i]*imw(i);
}

rho = P / (8.31446261815324e+07 * T * YOW);// rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOC(const amrex::Real P, const amrex::Real T,const amrex::Real c[], amrex::Real& rho)
{
// See Eq 5 in CK Manual
amrex::Real W = 0;
amrex::Real sumC = 0;
W += c[0]*131.293000; // XE
W += c[1]*2.016000; // H2
W += c[2]*1.008000; // H
W += c[3]*17.007000; // OH
W += c[4]*15.999000; // O
W += c[5]*18.015000; // H2O
W += c[6]*33.006000; // HO2
W += c[7]*34.014000; // H2O2
W += c[8]*28.014000; // N2
W += c[9]*31.998000; // O2

for (int id = 0; id < 10; ++id) {
sumC += c[id];
}
rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)

}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWT( amrex::Real wt[])
{
get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
amrex::Real YOW = 0;

for (int i = 0; i < 10; i++)
{
YOW += y[i]*imw(i);
}

wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWX(const amrex::Real x[],  amrex::Real& wtm)
{
amrex::Real XW = 0;//  see Eq 4 in CK Manual
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
wtm = XW;

}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWC(const amrex::Real c[],  amrex::Real& wtm)
{
// See Eq 5 in CK Manual
amrex::Real W = 0;
amrex::Real sumC = 0;
W += c[0]*131.293000; // XE
W += c[1]*2.016000; // H2
W += c[2]*1.008000; // H
W += c[3]*17.007000; // OH
W += c[4]*15.999000; // O
W += c[5]*18.015000; // H2O
W += c[6]*33.006000; // HO2
W += c[7]*34.014000; // H2O2
W += c[8]*28.014000; // N2
W += c[9]*31.998000; // O2

for (int id = 0; id < 10; ++id) {
sumC += c[id];
}
//  CK provides no guard against division by zero
wtm = W/sumC;

}

// get Cp/R as a function of T 
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
cp_R(cpor, T);
}

// get H/RT as a function of T 
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHORT(const amrex::Real T, amrex::Real hort[])
{
speciesEnthalpy(hort, T);
}

// get S/R as a function of T 
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSOR(const amrex::Real T, amrex::Real sor[])
{
speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTX(const amrex::Real y[], amrex::Real x[])
{
amrex::Real YOW = 0;

for (int i = 0; i < 10; i++)
{
YOW += y[i]*imw(i);
}

amrex::Real YOWINV = 1.0/YOW;

for (int i = 0; i < 10; i++)
{
x[i] = y[i]*imw(i)*YOWINV;
}
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTCP(const amrex::Real P, const amrex::Real T,const amrex::Real y[], amrex::Real c[])
{
amrex::Real YOW = 0;
amrex::Real PWORT;

// Compute inverse of mean molecular wt first
for (int i = 0; i < 10; i++)
{
c[i] = y[i]*imw(i);
}
for (int i = 0; i < 10; i++)
{
YOW += c[i];
}

// PW/RT (see Eq. 7)
PWORT = P/(YOW * 8.31446261815324e+07 * T); 
// Now compute conversion

for (int i = 0; i < 10; i++)
{
c[i] = PWORT * y[i] * imw(i);
}
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTCR(const amrex::Real rho, amrex::Real /*T*/, const amrex::Real y[],  amrex::Real c[])
{

for (int i = 0; i < 10; i++)
{
c[i] = rho  * y[i] * imw(i);
}
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTY(const amrex::Real x[], amrex::Real y[])
{
amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
// Compute mean molecular wt first
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
// Now compute conversion
amrex::Real XWinv = 1.0/XW;
y[0] = x[0]*131.293000*XWinv; 
y[1] = x[1]*2.016000*XWinv; 
y[2] = x[2]*1.008000*XWinv; 
y[3] = x[3]*17.007000*XWinv; 
y[4] = x[4]*15.999000*XWinv; 
y[5] = x[5]*18.015000*XWinv; 
y[6] = x[6]*33.006000*XWinv; 
y[7] = x[7]*34.014000*XWinv; 
y[8] = x[8]*28.014000*XWinv; 
y[9] = x[9]*31.998000*XWinv; 

}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTCP(const amrex::Real P, const amrex::Real T,const amrex::Real x[], amrex::Real c[])
{
amrex::Real PORT = P/(8.31446261815324e+07 * T); // P/RT

// Compute conversion, see Eq 10
for (int id = 0; id < 10; ++id) {
c[id] = x[id]*PORT;
}

}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTCR(const amrex::Real rho, const amrex::Real /*T*/, const amrex::Real x[], amrex::Real c[])
{
amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
amrex::Real ROW; 
// Compute mean molecular wt first
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
ROW = rho / XW;

// Compute conversion, see Eq 11
for (int id = 0; id < 10; ++id) {
c[id] = x[id]*ROW;
}

}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCTX(const amrex::Real c[], amrex::Real x[])
{
amrex::Real sumC = 0; 

// compute sum of c 
for (int id = 0; id < 10; ++id) {
sumC += c[id];
}

//  See Eq 13 
amrex::Real sumCinv = 1.0/sumC;
for (int id = 0; id < 10; ++id) {
x[id] = c[id]*sumCinv;
}

}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCTY(const amrex::Real c[], amrex::Real y[])
{
amrex::Real CW = 0; // See Eq 12 in CK Manual
// compute denominator in eq 12 first
CW += c[0]*131.293000; // XE
CW += c[1]*2.016000; // H2
CW += c[2]*1.008000; // H
CW += c[3]*17.007000; // OH
CW += c[4]*15.999000; // O
CW += c[5]*18.015000; // H2O
CW += c[6]*33.006000; // HO2
CW += c[7]*34.014000; // H2O2
CW += c[8]*28.014000; // N2
CW += c[9]*31.998000; // O2
// Now compute conversion
amrex::Real CWinv = 1.0/CW;
y[0] = c[0]*131.293000*CWinv; 
y[1] = c[1]*2.016000*CWinv; 
y[2] = c[2]*1.008000*CWinv; 
y[3] = c[3]*17.007000*CWinv; 
y[4] = c[4]*15.999000*CWinv; 
y[5] = c[5]*18.015000*CWinv; 
y[6] = c[6]*33.006000*CWinv; 
y[7] = c[7]*34.014000*CWinv; 
y[8] = c[8]*28.014000*CWinv; 
y[9] = c[9]*31.998000*CWinv; 

}

// get specific heat at constant volume as a function 
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVML(const amrex::Real T, amrex::Real cvml[])
{
cv_R(cvml, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
cvml[id] *= 8.31446261815324e+07;
}
}

// get specific heat at constant pressure as a 
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPML(const amrex::Real T, amrex::Real cpml[])
{
cp_R(cpml, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
cpml[id] *= 8.31446261815324e+07;
}
}

// get internal energy as a function 
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUML(const amrex::Real T, amrex::Real uml[])
{
amrex::Real RT = 8.31446261815324e+07*T; // R*T
speciesInternalEnergy(uml, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
uml[id] *= RT;
}
}

// get enthalpy as a function 
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHML(const amrex::Real T, amrex::Real hml[])
{
amrex::Real RT = 8.31446261815324e+07*T; // R*T
speciesEnthalpy(hml, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
hml[id] *= RT;
}
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSML(const amrex::Real T, amrex::Real sml[])
{
speciesEntropy(sml, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
sml[id] *= 8.31446261815324e+07;
}
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
cv_R(cvms, T);
// multiply by R/molecularweight
cvms[0] *= 6.332753930638526e+05; // XE
cvms[1] *= 4.124237409798234e+07; // H2
cvms[2] *= 8.248474819596468e+07; // H
cvms[3] *= 4.888847308845322e+06; // OH
cvms[4] *= 5.196863940342046e+06; // O
cvms[5] *= 4.615299815794193e+06; // H2O
cvms[6] *= 2.519076112874398e+06; // HO2
cvms[7] *= 2.444423654422661e+06; // H2O2
cvms[8] *= 2.967966951578939e+06; // N2
cvms[9] *= 2.598431970171023e+06; // O2
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
cp_R(cpms, T);
// multiply by R/molecularweight
cpms[0] *= 6.332753930638526e+05; // XE
cpms[1] *= 4.124237409798234e+07; // H2
cpms[2] *= 8.248474819596468e+07; // H
cpms[3] *= 4.888847308845322e+06; // OH
cpms[4] *= 5.196863940342046e+06; // O
cpms[5] *= 4.615299815794193e+06; // H2O
cpms[6] *= 2.519076112874398e+06; // HO2
cpms[7] *= 2.444423654422661e+06; // H2O2
cpms[8] *= 2.967966951578939e+06; // N2
cpms[9] *= 2.598431970171023e+06; // O2
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUMS(const amrex::Real T, amrex::Real ums[])
{
amrex::Real RT = 8.31446261815324e+07*T; // R*T

speciesInternalEnergy(ums, T);

for (int i = 0; i < 10; i++)
{
ums[i] *= RT*imw(i);
}
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHMS(const amrex::Real T, amrex::Real hms[])
{
amrex::Real RT = 8.31446261815324e+07*T; // R*T

speciesEnthalpy(hms, T);

for (int i = 0; i < 10; i++)
{
hms[i] *= RT*imw(i);
}
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSMS(const amrex::Real T, amrex::Real sms[])
{
speciesEntropy(sms, T);
// multiply by R/molecularweight
sms[0] *= 6.332753930638526e+05; // XE
sms[1] *= 4.124237409798234e+07; // H2
sms[2] *= 8.248474819596468e+07; // H
sms[3] *= 4.888847308845322e+06; // OH
sms[4] *= 5.196863940342046e+06; // O
sms[5] *= 4.615299815794193e+06; // H2O
sms[6] *= 2.519076112874398e+06; // HO2
sms[7] *= 2.444423654422661e+06; // H2O2
sms[8] *= 2.967966951578939e+06; // N2
sms[9] *= 2.598431970171023e+06; // O2
}

// GPU version of productionRate: no more use of thermo namespace vectors 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_qfqr(amrex::Real *  qf, amrex::Real * qr, const amrex::Real * sc, const amrex::Real * /*sc_qss*/,const amrex::Real T, const amrex::Real invT, const amrex::Real logT)
{

// reaction 8: H + O2 (+M) <=> HO2 (+M)
qf[0] = sc[2]*sc[9];
qr[0] = sc[6];

// reaction 15: H2O2 (+M) <=> 2 OH (+M)
qf[1] = sc[7];
qr[1] = (sc[3] * sc[3]);

// reaction 4: H2 + M <=> 2 H + M
qf[2] = sc[1];
qr[2] = (sc[2] * sc[2]);

// reaction 5: 2 O + M <=> O2 + M
qf[3] = (sc[4] * sc[4]);
qr[3] = sc[9];

// reaction 6: H + O + M <=> OH + M
qf[4] = sc[2]*sc[4];
qr[4] = sc[3];

// reaction 7: H + OH + M <=> H2O + M
qf[5] = sc[2]*sc[3];
qr[5] = sc[5];

// reaction 0: H + O2 <=> O + OH
qf[6] = sc[2]*sc[9];
qr[6] = sc[3]*sc[4];

// reaction 1: H2 + O <=> H + OH
qf[7] = sc[1]*sc[4];
qr[7] = sc[2]*sc[3];

// reaction 2: H2 + OH <=> H + H2O
qf[8] = sc[1]*sc[3];
qr[8] = sc[2]*sc[5];

// reaction 3: H2O + O <=> 2 OH
qf[9] = sc[4]*sc[5];
qr[9] = (sc[3] * sc[3]);

// reaction 9: H + HO2 <=> H2 + O2
qf[10] = sc[2]*sc[6];
qr[10] = sc[1]*sc[9];

// reaction 10: H + HO2 <=> 2 OH
qf[11] = sc[2]*sc[6];
qr[11] = (sc[3] * sc[3]);

// reaction 11: HO2 + O <=> O2 + OH
qf[12] = sc[4]*sc[6];
qr[12] = sc[3]*sc[9];

// reaction 12: HO2 + OH <=> H2O + O2
qf[13] = sc[3]*sc[6];
qr[13] = sc[5]*sc[9];

// reaction 13: 2 HO2 <=> H2O2 + O2
qf[14] = (sc[6] * sc[6]);
qr[14] = sc[7]*sc[9];

// reaction 14: 2 HO2 <=> H2O2 + O2
qf[15] = (sc[6] * sc[6]);
qr[15] = sc[7]*sc[9];

// reaction 16: H + H2O2 <=> H2O + OH
qf[16] = sc[2]*sc[7];
qr[16] = sc[3]*sc[5];

// reaction 17: H + H2O2 <=> H2 + HO2
qf[17] = sc[2]*sc[7];
qr[17] = sc[1]*sc[6];

// reaction 18: H2O2 + O <=> HO2 + OH
qf[18] = sc[4]*sc[7];
qr[18] = sc[3]*sc[6];

// reaction 19: H2O2 + OH <=> H2O + HO2
qf[19] = sc[3]*sc[7];
qr[19] = sc[5]*sc[6];

// reaction 20: H2O2 + OH <=> H2O + HO2
qf[20] = sc[3]*sc[7];
qr[20] = sc[5]*sc[6];

// compute the mixture concentration
amrex::Real mixture = 0.0;
for (int i = 0; i < 10; ++i) {
mixture += sc[i];
}

// compute the Gibbs free energy
amrex::Real g_RT[10];
gibbs(g_RT, T);

// reference concentration: P_atm / (RT) in inverse mol/m^3
amrex::Real refC = 101325 / 8.31446 * invT;
amrex::Real refCinv = 1 / refC;

// Evaluate the kfs
amrex::Real k_f, Corr;
amrex::Real redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;

// reaction 0:  H + O2 <=> O + OH
k_f = 191000000
           * exp(-(8272.88198582959) * invT);
qf[6] *= k_f;
qr[6] *= k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[9]));
// reaction 1:  H2 + O <=> H + OH
k_f = 0.0508
           * exp((2.67) * logT - (3166.2392612433) * invT);
qf[7] *= k_f;
qr[7] *= k_f * exp(-(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]));
// reaction 2:  H2 + OH <=> H + H2O
k_f = 216
           * exp((1.51) * logT - (1726.03316371019) * invT);
qf[8] *= k_f;
qr[8] *= k_f * exp(-(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[5]));
// reaction 3:  H2O + O <=> 2 OH
k_f = 2.97
           * exp((2.02) * logT - (6743.10332178324) * invT);
qf[9] *= k_f;
qr[9] *= k_f * exp(-(-2.000000*g_RT[3] + g_RT[4] + g_RT[5]));
// reaction 4:  H2 + M <=> 2 H + M
k_f = 45700000000000
           * exp((-1.4) * logT - (52888.071576076) * invT);
Corr  = mixture + (1.5)*sc[1] + (11)*sc[5];
qf[2] *= Corr * k_f;
qr[2] *= Corr * k_f * exp(-(g_RT[1] - 2.000000*g_RT[2])) * (refCinv);
// reaction 5:  2 O + M <=> O2 + M
k_f = 6170
           * exp((-0.5) * logT);
Corr  = mixture + (-0.17)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
qf[3] *= Corr * k_f;
qr[3] *= Corr * k_f * exp(-(2.000000*g_RT[4] - g_RT[9])) * (refC);
// reaction 6:  H + O + M <=> OH + M
k_f = 4720000
           * exp((-1) * logT);
Corr  = mixture + (-0.25)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
qf[4] *= Corr * k_f;
qr[4] *= Corr * k_f * exp(-(g_RT[2] - g_RT[3] + g_RT[4])) * (refC);
// reaction 7:  H + OH + M <=> H2O + M
k_f = 45000000000
           * exp((-2) * logT);
Corr  = mixture + (-0.62)*sc[0] + (-0.27)*sc[1] + (11)*sc[5];
qf[5] *= Corr * k_f;
qr[5] *= Corr * k_f * exp(-(g_RT[2] + g_RT[3] - g_RT[5])) * (refC);
// reaction 8:  H + O2 (+M) <=> HO2 (+M)
k_f = 1480000
           * exp((0.6) * logT);
Corr  = mixture + (-0.33)*sc[0] + (0.3)*sc[1] + (13)*sc[5];
redP = Corr / k_f * 34800 
           * exp(-0.41  * logT - (-563.602665701286) *invT);
F = redP / (1.0 + redP);
logPred = log10(redP);
logFcent = log10(
    (0.5)*exp(-T * 1e+30)
    + 0.5 * exp(-T * 1e-30)
    + exp(-1e+100 * invT));
troe_c = -0.4 - 0.67 * logFcent;
troe_n = 0.75 - 1.27 * logFcent;
troe = (troe_c + logPred) / (troe_n - 0.14*(troe_c + logPred));
F_troe = exp(M_LN10 * logFcent / (1.0 + troe*troe));
Corr = F * F_troe;
qf[0] *= Corr * k_f;
qr[0] *= Corr * k_f * exp(-(g_RT[2] - g_RT[6] + g_RT[9])) * (refC);
// reaction 9:  H + HO2 <=> H2 + O2
k_f = 16600000
           * exp(-(412.63766595987) * invT);
qf[10] *= k_f;
qr[10] *= k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[6] - g_RT[9]));
// reaction 10:  H + HO2 <=> 2 OH
k_f = 70800000
           * exp(-(150.964999741416) * invT);
qf[11] *= k_f;
qr[11] *= k_f * exp(-(g_RT[2] - 2.000000*g_RT[3] + g_RT[6]));
// reaction 11:  HO2 + O <=> O2 + OH
k_f = 32500000
           ;
qf[12] *= k_f;
qr[12] *= k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[6] - g_RT[9]));
// reaction 12:  HO2 + OH <=> H2O + O2
k_f = 28900000
           * exp(-(-251.60833290236) * invT);
qf[13] *= k_f;
qr[13] *= k_f * exp(-(g_RT[3] - g_RT[5] + g_RT[6] - g_RT[9]));
// reaction 13:  2 HO2 <=> H2O2 + O2
k_f = 420000000
           * exp(-(6028.53565634054) * invT);
qf[14] *= k_f;
qr[14] *= k_f * exp(-(2.000000*g_RT[6] - g_RT[7] - g_RT[9]));
// reaction 14:  2 HO2 <=> H2O2 + O2
k_f = 130000
           * exp(-(-819.739948595888) * invT);
qf[15] *= k_f;
qr[15] *= k_f * exp(-(2.000000*g_RT[6] - g_RT[7] - g_RT[9]));
// reaction 15:  H2O2 (+M) <=> 2 OH (+M)
k_f = 295000000000000
           * exp(-(24355.6866249484) * invT);
Corr  = mixture + (-0.55)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
redP = Corr / k_f * 127000000000 
           * exp(- (22896.3582941147) *invT);
F = redP / (1.0 + redP);
logPred = log10(redP);
logFcent = log10(
    (0.5)*exp(-T * 1e+30)
    + 0.5 * exp(-T * 1e-30)
    + 0.0);
troe_c = -0.4 - 0.67 * logFcent;
troe_n = 0.75 - 1.27 * logFcent;
troe = (troe_c + logPred) / (troe_n - 0.14*(troe_c + logPred));
F_troe = exp(M_LN10 * logFcent / (1.0 + troe*troe));
Corr = F * F_troe;
qf[1] *= Corr * k_f;
qr[1] *= Corr * k_f * exp(-(-2.000000*g_RT[3] + g_RT[7])) * (refCinv);
// reaction 16:  H + H2O2 <=> H2O + OH
k_f = 24100000
           * exp(-(1997.77016324474) * invT);
qf[16] *= k_f;
qr[16] *= k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[5] + g_RT[7]));
// reaction 17:  H + H2O2 <=> H2 + HO2
k_f = 60300000
           * exp(-(4000.57249314752) * invT);
qf[17] *= k_f;
qr[17] *= k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[6] + g_RT[7]));
// reaction 18:  H2O2 + O <=> HO2 + OH
k_f = 9.55
           * exp((2) * logT - (1997.77016324474) * invT);
qf[18] *= k_f;
qr[18] *= k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[7]));
// reaction 19:  H2O2 + OH <=> H2O + HO2
k_f = 1000000
           ;
qf[19] *= k_f;
qr[19] *= k_f * exp(-(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]));
// reaction 20:  H2O2 + OH <=> H2O + HO2
k_f = 580000000
           * exp(-(4810.75132509312) * invT);
qf[20] *= k_f;
qr[20] *= k_f * exp(-(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]));


}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void  productionRate(amrex::Real * wdot, const amrex::Real * sc, const amrex::Real T)
{
const amrex::Real invT = 1.0 / T;
const amrex::Real logT = log(T);

// reference concentration: P_atm / (RT) in inverse mol/m^3
const amrex::Real refC = 101325 / 8.31446 * invT;
const amrex::Real refCinv = 1 / refC;

for (int i = 0; i < 10; ++i) {
wdot[i] = 0.0;
}

// compute the mixture concentration
amrex::Real mixture = 0.0;
for (int i = 0; i < 10; ++i) {
mixture += sc[i];
}

// compute the Gibbs free energy
amrex::Real g_RT[10];
gibbs(g_RT, T);

{
// reaction 8:  H + O2 (+M) <=> HO2 (+M)
const amrex::Real k_f = 1480000
           * exp((0.6) * logT);
amrex::Real Corr = mixture + (-0.33)*sc[0] + (0.3)*sc[1] + (13)*sc[5];
const amrex::Real redP = Corr / k_f * 34800 
           * exp(-0.41 * logT - -563.602665701286 * invT);
const amrex::Real F = redP / (1.0 + redP);
const amrex::Real logPred = log10(redP);
const amrex::Real logFcent = log10(
    0.5 * exp(-T * 1e+30)
    + 0.5 * exp(-T * 1e-30)
    + exp(-1e+100 * invT));
const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
const amrex::Real troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
const amrex::Real F_troe = exp(M_LN10 *logFcent / (1.0 + troe * troe));
Corr = F * F_troe;
const amrex::Real qf = Corr * k_f * (sc[2]*sc[9]);
const amrex::Real qr = Corr * k_f * exp(-(g_RT[2] - g_RT[6] + g_RT[9])) * (refC) * (sc[6]);
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[6] += qdot;
wdot[9] -= qdot;
}

{
// reaction 15:  H2O2 (+M) <=> 2 OH (+M)
const amrex::Real k_f = 295000000000000
           * exp(-(24355.6866249484) * invT);
amrex::Real Corr = mixture + (-0.55)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
const amrex::Real redP = Corr / k_f * 127000000000 
           * exp(- 22896.3582941147 * invT);
const amrex::Real F = redP / (1.0 + redP);
const amrex::Real logPred = log10(redP);
const amrex::Real logFcent = log10(
    0.5 * exp(-T * 1e+30)
    + 0.5 * exp(-T * 1e-30)
    + 0.0);
const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
const amrex::Real troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
const amrex::Real F_troe = exp(M_LN10 *logFcent / (1.0 + troe * troe));
Corr = F * F_troe;
const amrex::Real qf = Corr * k_f * (sc[7]);
const amrex::Real qr = Corr * k_f * exp(-(-2.000000*g_RT[3] + g_RT[7])) * (refCinv) * ((sc[3] * sc[3]));
const amrex::Real qdot = qf - qr;
wdot[3] += 2.000000 * qdot;
wdot[7] -= qdot;
}

{
// reaction 4:  H2 + M <=> 2 H + M
const amrex::Real k_f = 45700000000000
           * exp((-1.4) * logT - (52888.071576076) * invT);
const amrex::Real Corr = mixture + (1.5)*sc[1] + (11)*sc[5];
const amrex::Real qf = Corr * k_f * (sc[1]);
const amrex::Real qr = Corr * k_f * exp(-(g_RT[1] - 2.000000*g_RT[2])) * (refCinv) * ((sc[2] * sc[2]));
const amrex::Real qdot = qf - qr;
wdot[1] -= qdot;
wdot[2] += 2.000000 * qdot;
}

{
// reaction 5:  2 O + M <=> O2 + M
const amrex::Real k_f = 6170
           * exp((-0.5) * logT);
const amrex::Real Corr = mixture + (-0.17)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
const amrex::Real qf = Corr * k_f * ((sc[4] * sc[4]));
const amrex::Real qr = Corr * k_f * exp(-(2.000000*g_RT[4] - g_RT[9])) * (refC) * (sc[9]);
const amrex::Real qdot = qf - qr;
wdot[4] -= 2.000000 * qdot;
wdot[9] += qdot;
}

{
// reaction 6:  H + O + M <=> OH + M
const amrex::Real k_f = 4720000
           * exp((-1) * logT);
const amrex::Real Corr = mixture + (-0.25)*sc[0] + (1.5)*sc[1] + (11)*sc[5];
const amrex::Real qf = Corr * k_f * (sc[2]*sc[4]);
const amrex::Real qr = Corr * k_f * exp(-(g_RT[2] - g_RT[3] + g_RT[4])) * (refC) * (sc[3]);
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[3] += qdot;
wdot[4] -= qdot;
}

{
// reaction 7:  H + OH + M <=> H2O + M
const amrex::Real k_f = 45000000000
           * exp((-2) * logT);
const amrex::Real Corr = mixture + (-0.62)*sc[0] + (-0.27)*sc[1] + (11)*sc[5];
const amrex::Real qf = Corr * k_f * (sc[2]*sc[3]);
const amrex::Real qr = Corr * k_f * exp(-(g_RT[2] + g_RT[3] - g_RT[5])) * (refC) * (sc[5]);
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[3] -= qdot;
wdot[5] += qdot;
}

{
// reaction 0:  H + O2 <=> O + OH
const amrex::Real k_f = 191000000
           * exp(-(8272.88198582959) * invT);
const amrex::Real qf = k_f * (sc[2]*sc[9]);
const amrex::Real qr = k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[9])) * (sc[3]*sc[4]);
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[3] += qdot;
wdot[4] += qdot;
wdot[9] -= qdot;
}

{
// reaction 1:  H2 + O <=> H + OH
const amrex::Real k_f = 0.0508
           * exp((2.67) * logT - (3166.2392612433) * invT);
const amrex::Real qf = k_f * (sc[1]*sc[4]);
const amrex::Real qr = k_f * exp(-(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4])) * (sc[2]*sc[3]);
const amrex::Real qdot = qf - qr;
wdot[1] -= qdot;
wdot[2] += qdot;
wdot[3] += qdot;
wdot[4] -= qdot;
}

{
// reaction 2:  H2 + OH <=> H + H2O
const amrex::Real k_f = 216
           * exp((1.51) * logT - (1726.03316371019) * invT);
const amrex::Real qf = k_f * (sc[1]*sc[3]);
const amrex::Real qr = k_f * exp(-(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[5])) * (sc[2]*sc[5]);
const amrex::Real qdot = qf - qr;
wdot[1] -= qdot;
wdot[2] += qdot;
wdot[3] -= qdot;
wdot[5] += qdot;
}

{
// reaction 3:  H2O + O <=> 2 OH
const amrex::Real k_f = 2.97
           * exp((2.02) * logT - (6743.10332178324) * invT);
const amrex::Real qf = k_f * (sc[4]*sc[5]);
const amrex::Real qr = k_f * exp(-(-2.000000*g_RT[3] + g_RT[4] + g_RT[5])) * ((sc[3] * sc[3]));
const amrex::Real qdot = qf - qr;
wdot[3] += 2.000000 * qdot;
wdot[4] -= qdot;
wdot[5] -= qdot;
}

{
// reaction 9:  H + HO2 <=> H2 + O2
const amrex::Real k_f = 16600000
           * exp(-(412.63766595987) * invT);
const amrex::Real qf = k_f * (sc[2]*sc[6]);
const amrex::Real qr = k_f * exp(-(-g_RT[1] + g_RT[2] + g_RT[6] - g_RT[9])) * (sc[1]*sc[9]);
const amrex::Real qdot = qf - qr;
wdot[1] += qdot;
wdot[2] -= qdot;
wdot[6] -= qdot;
wdot[9] += qdot;
}

{
// reaction 10:  H + HO2 <=> 2 OH
const amrex::Real k_f = 70800000
           * exp(-(150.964999741416) * invT);
const amrex::Real qf = k_f * (sc[2]*sc[6]);
const amrex::Real qr = k_f * exp(-(g_RT[2] - 2.000000*g_RT[3] + g_RT[6])) * ((sc[3] * sc[3]));
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[3] += 2.000000 * qdot;
wdot[6] -= qdot;
}

{
// reaction 11:  HO2 + O <=> O2 + OH
const amrex::Real k_f = 32500000
           ;
const amrex::Real qf = k_f * (sc[4]*sc[6]);
const amrex::Real qr = k_f * exp(-(-g_RT[3] + g_RT[4] + g_RT[6] - g_RT[9])) * (sc[3]*sc[9]);
const amrex::Real qdot = qf - qr;
wdot[3] += qdot;
wdot[4] -= qdot;
wdot[6] -= qdot;
wdot[9] += qdot;
}

{
// reaction 12:  HO2 + OH <=> H2O + O2
const amrex::Real k_f = 28900000
           * exp(-(-251.60833290236) * invT);
const amrex::Real qf = k_f * (sc[3]*sc[6]);
const amrex::Real qr = k_f * exp(-(g_RT[3] - g_RT[5] + g_RT[6] - g_RT[9])) * (sc[5]*sc[9]);
const amrex::Real qdot = qf - qr;
wdot[3] -= qdot;
wdot[5] += qdot;
wdot[6] -= qdot;
wdot[9] += qdot;
}

{
// reaction 13:  2 HO2 <=> H2O2 + O2
const amrex::Real k_f = 420000000
           * exp(-(6028.53565634054) * invT);
const amrex::Real qf = k_f * ((sc[6] * sc[6]));
const amrex::Real qr = k_f * exp(-(2.000000*g_RT[6] - g_RT[7] - g_RT[9])) * (sc[7]*sc[9]);
const amrex::Real qdot = qf - qr;
wdot[6] -= 2.000000 * qdot;
wdot[7] += qdot;
wdot[9] += qdot;
}

{
// reaction 14:  2 HO2 <=> H2O2 + O2
const amrex::Real k_f = 130000
           * exp(-(-819.739948595888) * invT);
const amrex::Real qf = k_f * ((sc[6] * sc[6]));
const amrex::Real qr = k_f * exp(-(2.000000*g_RT[6] - g_RT[7] - g_RT[9])) * (sc[7]*sc[9]);
const amrex::Real qdot = qf - qr;
wdot[6] -= 2.000000 * qdot;
wdot[7] += qdot;
wdot[9] += qdot;
}

{
// reaction 16:  H + H2O2 <=> H2O + OH
const amrex::Real k_f = 24100000
           * exp(-(1997.77016324474) * invT);
const amrex::Real qf = k_f * (sc[2]*sc[7]);
const amrex::Real qr = k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[5] + g_RT[7])) * (sc[3]*sc[5]);
const amrex::Real qdot = qf - qr;
wdot[2] -= qdot;
wdot[3] += qdot;
wdot[5] += qdot;
wdot[7] -= qdot;
}

{
// reaction 17:  H + H2O2 <=> H2 + HO2
const amrex::Real k_f = 60300000
           * exp(-(4000.57249314752) * invT);
const amrex::Real qf = k_f * (sc[2]*sc[7]);
const amrex::Real qr = k_f * exp(-(-g_RT[1] + g_RT[2] - g_RT[6] + g_RT[7])) * (sc[1]*sc[6]);
const amrex::Real qdot = qf - qr;
wdot[1] += qdot;
wdot[2] -= qdot;
wdot[6] += qdot;
wdot[7] -= qdot;
}

{
// reaction 18:  H2O2 + O <=> HO2 + OH
const amrex::Real k_f = 9.55
           * exp((2) * logT - (1997.77016324474) * invT);
const amrex::Real qf = k_f * (sc[4]*sc[7]);
const amrex::Real qr = k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[7])) * (sc[3]*sc[6]);
const amrex::Real qdot = qf - qr;
wdot[3] += qdot;
wdot[4] -= qdot;
wdot[6] += qdot;
wdot[7] -= qdot;
}

{
// reaction 19:  H2O2 + OH <=> H2O + HO2
const amrex::Real k_f = 1000000
           ;
const amrex::Real qf = k_f * (sc[3]*sc[7]);
const amrex::Real qr = k_f * exp(-(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7])) * (sc[5]*sc[6]);
const amrex::Real qdot = qf - qr;
wdot[3] -= qdot;
wdot[5] += qdot;
wdot[6] += qdot;
wdot[7] -= qdot;
}

{
// reaction 20:  H2O2 + OH <=> H2O + HO2
const amrex::Real k_f = 580000000
           * exp(-(4810.75132509312) * invT);
const amrex::Real qf = k_f * (sc[3]*sc[7]);
const amrex::Real qr = k_f * exp(-(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7])) * (sc[5]*sc[6]);
const amrex::Real qdot = qf - qr;
wdot[3] -= qdot;
wdot[5] += qdot;
wdot[6] += qdot;
wdot[7] -= qdot;
}


}


// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

// convert to SI
for (int id = 0; id < 10; ++id) {
C[id] *= 1.0e6;
}

// convert to chemkin units
productionRate(wdot, C, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
C[id] *= 1.0e-6;
wdot[id] *= 1.0e-6;
}
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWYP(const amrex::Real P, const amrex::Real T,const amrex::Real y[], amrex::Real wdot[])
{
amrex::Real c[10]; // temporary storage
amrex::Real YOW = 0; 
amrex::Real PWORT; 

// Compute inverse of mean molecular wt first
for (int i = 0; i < 10; i++)
{
YOW += y[i]*imw(i);
}
// PW/RT (see Eq. 7)
PWORT = P/(YOW * 8.31446261815324e+07 * T); 
// multiply by 1e6 so c goes to SI
PWORT *= 1e6; 
// Now compute conversion (and go to SI)
for (int i = 0; i < 10; i++)
{
c[i] = PWORT * y[i]*imw(i);
}

// convert to chemkin units
productionRate(wdot, c, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
wdot[id] *= 1.0e-6;
}
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWXP(const amrex::Real P, const amrex::Real T,const amrex::Real x[], amrex::Real wdot[])
{
amrex::Real c[10]; // temporary storage
amrex::Real PORT = 1e6 * P/(8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

// Compute conversion, see Eq 10
for (int id = 0; id < 10; ++id) {
c[id] = x[id]*PORT;
}

// convert to chemkin units
productionRate(wdot, c, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
wdot[id] *= 1.0e-6;
}
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWYR(const amrex::Real rho, const amrex::Real T, const amrex::Real y[],  amrex::Real wdot[])
{
amrex::Real c[10]; // temporary storage

// See Eq 8 with an extra 1e6 so c goes to SI
for (int i = 0; i < 10; i++)
{
c[i] = 1e6 * rho * y[i]*imw(i);
}

// call productionRate
productionRate(wdot, c, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
wdot[id] *= 1.0e-6;
}
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWXR(const amrex::Real rho, const amrex::Real T, const amrex::Real x[],  amrex::Real wdot[])
{
amrex::Real c[10]; // temporary storage
amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
amrex::Real ROW; 
// Compute mean molecular wt first
XW += x[0]*131.293000; // XE
XW += x[1]*2.016000; // H2
XW += x[2]*1.008000; // H
XW += x[3]*17.007000; // OH
XW += x[4]*15.999000; // O
XW += x[5]*18.015000; // H2O
XW += x[6]*33.006000; // HO2
XW += x[7]*34.014000; // H2O2
XW += x[8]*28.014000; // N2
XW += x[9]*31.998000; // O2
// Extra 1e6 factor to take c to SI
ROW = 1e6*rho / XW;

// Compute conversion, see Eq 11
for (int id = 0; id < 10; ++id) {
c[id] = x[id]*ROW;
}

// convert to chemkin units
productionRate(wdot, c, T);

// convert to chemkin units
for (int id = 0; id < 10; ++id) {
wdot[id] *= 1.0e-6;
}
}

//  species unit charge number 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
kcharge[0] = 0;// XE
kcharge[1] = 0;// H2
kcharge[2] = 0;// H
kcharge[3] = 0;// OH
kcharge[4] = 0;// O
kcharge[5] = 0;// H2O
kcharge[6] = 0;// HO2
kcharge[7] = 0;// H2O2
kcharge[8] = 0;// N2
kcharge[9] = 0;// O2
}

//  species charge per unit mass 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

int kchrg[10];
CKCHRG(kchrg);

for (int id = 0; id < 10; ++id) {
zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
}
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void aJacobian_precond(amrex::Real *  J, const amrex::Real *  sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
for (int i=0; i<121; i++) {
J[i] = 0.0;
}

amrex::Real wdot[10];
for (auto& val : wdot) {
val = 0.0;
}

const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT * invT;
const amrex::Real logT = log(T);

// reference concentration: P_atm / (RT) in inverse mol/m^3
amrex::Real refC = 101325 / 8.31446 / T;
amrex::Real refCinv = 1.0 / refC;

// compute the mixture concentration
amrex::Real mixture = 0.0;
for (int k = 0; k < 10; ++k) {
mixture += sc[k];
}

// compute the Gibbs free energy
amrex::Real g_RT[10];
gibbs(g_RT, T);

// compute the species enthalpy
amrex::Real h_RT[10];
speciesEnthalpy(h_RT, T);

amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
amrex::Real dqdci, dcdc_fac, dqdc[10];
amrex::Real Pr, fPr, F, k_0, logPr;
amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
const amrex::Real ln10 = log(10.0);
const amrex::Real log10e = 1.0/log(10.0);
// reaction 8: H + O2 (+M) <=> HO2 (+M)
// a third-body and pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.67 - 1)*sc[0] + ( 1.3 - 1)*sc[1] + ( 14 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[9];
k_f = 1480000
            * exp(0.6 * logT);
dlnkfdT = 0.6 * invT;
// pressure-fall-off
k_0 = 3.48e+16 * exp(-0.41 * logT - (-563.602665701286) * invT);
Pr = 1e-12 * alpha / k_f * k_0;
fPr = Pr / (1.0+Pr);
dlnk0dT = -0.41 * invT + (-563.602665701286) * invT2;
dlogPrdT = log10e*(dlnk0dT - dlnkfdT);
dlogfPrdT = dlogPrdT / (1.0+Pr);
// Troe form
logPr = log10(Pr);
Fcent1 = (1.-0.5)*exp(-T/1e-30);
Fcent2 = 0.5 * exp(-T/1e+30);
Fcent3 = exp(-1e+100 * invT);
Fcent = Fcent1 + Fcent2 + Fcent3;
logFcent = log10(Fcent);
troe_c = -.4 - .67 * logFcent;
troe_n = .75 - 1.27 * logFcent;
troePr_den = 1.0 / (troe_n - .14*(troe_c + logPr));
troePr = (troe_c + logPr) * troePr_den;
troe = 1.0 / (1.0 + troePr*troePr);
F = exp(M_LN10 * logFcent * troe);
dlogFcentdT = log10e/Fcent*( 
    -Fcent1/1e-30
    -Fcent2/1e+30
    + Fcent3*1e+100*invT2
);
dlogFdcn_fac = 2.0 * logFcent * troe*troe * troePr * troePr_den;
dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
dlogFdn = dlogFdcn_fac * troePr;
dlogFdlogPr = dlogFdc;
dlogFdT = dlogFcentdT*(troe - 0.67*dlogFdc - 1.27*dlogFdn) + dlogFdlogPr * dlogPrdT;
// reverse
phi_r = sc[6];
Kc = refCinv * exp(g_RT[2] - g_RT[6] + g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[9]) + (h_RT[6]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
Corr = fPr * F;
q = Corr * q_nocor;
dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);
dqdT = Corr *(dlnkfdT*k_f*phi_f - dkrdT*phi_r) + dlnCorrdT*q;
// update wdot
wdot[2] -= q;// H
wdot[6] += q;// HO2
wdot[9] -= q;// O2
// for convenience
k_f *= Corr;
k_r *= Corr;
dcdc_fac = 0.0;
dqdc[0] = 0.67*dcdc_fac;
dqdc[1] = 1.3*dcdc_fac;
dqdc[2] = dcdc_fac + k_f*sc[9];
dqdc[3] = dcdc_fac;
dqdc[4] = dcdc_fac;
dqdc[5] = 14*dcdc_fac;
dqdc[6] = dcdc_fac - k_r;
dqdc[7] = dcdc_fac;
dqdc[8] = dcdc_fac;
dqdc[9] = dcdc_fac + k_f*sc[2];
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+6] += dqdc[k];
J[11*k+9] -= dqdc[k];
}
J[112] -= dqdT;// dwdot[H]/dT
J[116] += dqdT;// dwdot[HO2]/dT
J[119] -= dqdT;// dwdot[O2]/dT

// reaction 15: H2O2 (+M) <=> 2 OH (+M)
// a third-body and pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.45 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[7];
k_f = 295000000000000
            * exp(- (24355.6866249484) * invT);
dlnkfdT = (24355.6866249484) * invT2;
// pressure-fall-off
k_0 = 1.27e+17 * exp(-(22896.3582941147) * invT);
Pr = 1e-6 * alpha / k_f * k_0;
fPr = Pr / (1.0+Pr);
dlnk0dT = (22896.3582941147) * invT2;
dlogPrdT = log10e*(dlnk0dT - dlnkfdT);
dlogfPrdT = dlogPrdT / (1.0+Pr);
// Troe form
logPr = log10(Pr);
Fcent1 = (1.-0.5)*exp(-T/1e-30);
Fcent2 = 0.5 * exp(-T/1e+30);
Fcent3 = 0.;
Fcent = Fcent1 + Fcent2 + Fcent3;
logFcent = log10(Fcent);
troe_c = -.4 - .67 * logFcent;
troe_n = .75 - 1.27 * logFcent;
troePr_den = 1.0 / (troe_n - .14*(troe_c + logPr));
troePr = (troe_c + logPr) * troePr_den;
troe = 1.0 / (1.0 + troePr*troePr);
F = exp(M_LN10 * logFcent * troe);
dlogFcentdT = log10e/Fcent*( 
    -Fcent1/1e-30
    -Fcent2/1e+30
);
dlogFdcn_fac = 2.0 * logFcent * troe*troe * troePr * troePr_den;
dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
dlogFdn = dlogFdcn_fac * troePr;
dlogFdlogPr = dlogFdc;
dlogFdT = dlogFcentdT*(troe - 0.67*dlogFdc - 1.27*dlogFdn) + dlogFdlogPr * dlogPrdT;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = refC * exp(-2.000000*g_RT[3] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[7]) + (2.000000*h_RT[3]) - 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
Corr = fPr * F;
q = Corr * q_nocor;
dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);
dqdT = Corr *(dlnkfdT*k_f*phi_f - dkrdT*phi_r) + dlnCorrdT*q;
// update wdot
wdot[3] += 2 * q;// OH
wdot[7] -= q;// H2O2
// for convenience
k_f *= Corr;
k_r *= Corr;
dcdc_fac = 0.0;
dqdc[0] = 0.45*dcdc_fac;
dqdc[1] = 2.5*dcdc_fac;
dqdc[2] = dcdc_fac;
dqdc[3] = dcdc_fac - k_r*2.000000*sc[3];
dqdc[4] = dcdc_fac;
dqdc[5] = 12*dcdc_fac;
dqdc[6] = dcdc_fac;
dqdc[7] = dcdc_fac + k_f;
dqdc[8] = dcdc_fac;
dqdc[9] = dcdc_fac;
for (int k=0; k<10; k++) {
J[11*k+3] += 2 * dqdc[k];
J[11*k+7] -= dqdc[k];
}
J[113] += 2 * dqdT;// dwdot[OH]/dT
J[117] -= dqdT;// dwdot[H2O2]/dT

// reaction 4: H2 + M <=> 2 H + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[1];
k_f = 45700000000000
            * exp(-1.4 * logT - (52888.071576076) * invT);
dlnkfdT = -1.4 * invT + (52888.071576076) * invT2;
// reverse
phi_r = (sc[2] * sc[2]);
Kc = refC * exp(g_RT[1] - 2.000000*g_RT[2]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1]) + (2.000000*h_RT[2]) - 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += 2 * q;// H
// for convenience
k_f *= alpha;
k_r *= alpha;
dqdc[0] = q_nocor;
dqdc[1] = 2.5*q_nocor + k_f;
dqdc[2] = q_nocor - k_r*2.000000*sc[2];
dqdc[3] = q_nocor;
dqdc[4] = q_nocor;
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+1] -= dqdc[k];
J[11*k+2] += 2 * dqdc[k];
}
J[111] -= dqdT;// dwdot[H2]/dT
J[112] += 2 * dqdT;// dwdot[H]/dT

// reaction 5: 2 O + M <=> O2 + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.83 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = (sc[4] * sc[4]);
k_f = 6170
            * exp(-0.5 * logT);
dlnkfdT = -0.5 * invT;
// reverse
phi_r = sc[9];
Kc = refCinv * exp(2.000000*g_RT[4] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[4]) + (h_RT[9]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[4] -= 2 * q;// O
wdot[9] += q;// O2
// for convenience
k_f *= alpha;
k_r *= alpha;
dqdc[0] = 0.83*q_nocor;
dqdc[1] = 2.5*q_nocor;
dqdc[2] = q_nocor;
dqdc[3] = q_nocor;
dqdc[4] = q_nocor + k_f*2.000000*sc[4];
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor - k_r;
for (int k=0; k<10; k++) {
J[11*k+4] += -2 * dqdc[k];
J[11*k+9] += dqdc[k];
}
J[114] += -2 * dqdT;// dwdot[O]/dT
J[119] += dqdT;// dwdot[O2]/dT

// reaction 6: H + O + M <=> OH + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.75 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[4];
k_f = 4720000
            * exp(-1 * logT);
dlnkfdT = -1 * invT;
// reverse
phi_r = sc[3];
Kc = refCinv * exp(g_RT[2] - g_RT[3] + g_RT[4]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[3]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[4] -= q;// O
// for convenience
k_f *= alpha;
k_r *= alpha;
dqdc[0] = 0.75*q_nocor;
dqdc[1] = 2.5*q_nocor;
dqdc[2] = q_nocor + k_f*sc[4];
dqdc[3] = q_nocor - k_r;
dqdc[4] = q_nocor + k_f*sc[2];
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+3] += dqdc[k];
J[11*k+4] -= dqdc[k];
}
J[112] -= dqdT;// dwdot[H]/dT
J[113] += dqdT;// dwdot[OH]/dT
J[114] -= dqdT;// dwdot[O]/dT

// reaction 7: H + OH + M <=> H2O + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.38 - 1)*sc[0] + ( 0.73 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[3];
k_f = 45000000000
            * exp(-2 * logT);
dlnkfdT = -2 * invT;
// reverse
phi_r = sc[5];
Kc = refCinv * exp(g_RT[2] + g_RT[3] - g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[5]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] -= q;// OH
wdot[5] += q;// H2O
// for convenience
k_f *= alpha;
k_r *= alpha;
dqdc[0] = 0.38*q_nocor;
dqdc[1] = 0.73*q_nocor;
dqdc[2] = q_nocor + k_f*sc[3];
dqdc[3] = q_nocor + k_f*sc[2];
dqdc[4] = q_nocor;
dqdc[5] = 12*q_nocor - k_r;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+3] -= dqdc[k];
J[11*k+5] += dqdc[k];
}
J[112] -= dqdT;// dwdot[H]/dT
J[113] -= dqdT;// dwdot[OH]/dT
J[115] += dqdT;// dwdot[H2O]/dT

// reaction 0: H + O2 <=> O + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[9];
k_f = 191000000
            * exp(- (8272.88198582959) * invT);
dlnkfdT = (8272.88198582959) * invT2;
// reverse
phi_r = sc[3]*sc[4];
Kc = exp(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[9]) + (h_RT[3] + h_RT[4]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[4] += q;// O
wdot[9] -= q;// O2
// d()/d[H]
dqdci =  + k_f*sc[9];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[26] += dqdci;               // dwdot[O]/d[H]
J[31] -= dqdci;               // dwdot[O2]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[4];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] += dqdci;               // dwdot[O]/d[OH]
J[42] -= dqdci;               // dwdot[O2]/d[OH]
// d()/d[O]
dqdci =  - k_r*sc[3];
J[46] -= dqdci;               // dwdot[H]/d[O]
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] += dqdci;               // dwdot[O]/d[O]
J[53] -= dqdci;               // dwdot[O2]/d[O]
// d()/d[O2]
dqdci =  + k_f*sc[2];
J[101] -= dqdci;              // dwdot[H]/d[O2]
J[102] += dqdci;              // dwdot[OH]/d[O2]
J[103] += dqdci;              // dwdot[O]/d[O2]
J[108] -= dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] += dqdT;               // dwdot[O]/dT
J[119] -= dqdT;               // dwdot[O2]/dT

// reaction 1: H2 + O <=> H + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[1]*sc[4];
k_f = 0.0508
            * exp(2.67 * logT - (3166.2392612433) * invT);
dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
// reverse
phi_r = sc[2]*sc[3];
Kc = exp(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += q;// H
wdot[3] += q;// OH
wdot[4] -= q;// O
// d()/d[H2]
dqdci =  + k_f*sc[4];
J[12] -= dqdci;               // dwdot[H2]/d[H2]
J[13] += dqdci;               // dwdot[H]/d[H2]
J[14] += dqdci;               // dwdot[OH]/d[H2]
J[15] -= dqdci;               // dwdot[O]/d[H2]
// d()/d[H]
dqdci =  - k_r*sc[3];
J[23] -= dqdci;               // dwdot[H2]/d[H]
J[24] += dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[26] -= dqdci;               // dwdot[O]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[2];
J[34] -= dqdci;               // dwdot[H2]/d[OH]
J[35] += dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[1];
J[45] -= dqdci;               // dwdot[H2]/d[O]
J[46] += dqdci;               // dwdot[H]/d[O]
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
// d()/dT
J[111] -= dqdT;               // dwdot[H2]/dT
J[112] += dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT

// reaction 2: H2 + OH <=> H + H2O
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[1]*sc[3];
k_f = 216
            * exp(1.51 * logT - (1726.03316371019) * invT);
dlnkfdT = 1.51 * invT + (1726.03316371019) * invT2;
// reverse
phi_r = sc[2]*sc[5];
Kc = exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[2] + h_RT[5]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += q;// H
wdot[3] -= q;// OH
wdot[5] += q;// H2O
// d()/d[H2]
dqdci =  + k_f*sc[3];
J[12] -= dqdci;               // dwdot[H2]/d[H2]
J[13] += dqdci;               // dwdot[H]/d[H2]
J[14] -= dqdci;               // dwdot[OH]/d[H2]
J[16] += dqdci;               // dwdot[H2O]/d[H2]
// d()/d[H]
dqdci =  - k_r*sc[5];
J[23] -= dqdci;               // dwdot[H2]/d[H]
J[24] += dqdci;               // dwdot[H]/d[H]
J[25] -= dqdci;               // dwdot[OH]/d[H]
J[27] += dqdci;               // dwdot[H2O]/d[H]
// d()/d[OH]
dqdci =  + k_f*sc[1];
J[34] -= dqdci;               // dwdot[H2]/d[OH]
J[35] += dqdci;               // dwdot[H]/d[OH]
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[2];
J[56] -= dqdci;               // dwdot[H2]/d[H2O]
J[57] += dqdci;               // dwdot[H]/d[H2O]
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
// d()/dT
J[111] -= dqdT;               // dwdot[H2]/dT
J[112] += dqdT;               // dwdot[H]/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT

// reaction 3: H2O + O <=> 2 OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[5];
k_f = 2.97
            * exp(2.02 * logT - (6743.10332178324) * invT);
dlnkfdT = 2.02 * invT + (6743.10332178324) * invT2;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = exp(-2.000000*g_RT[3] + g_RT[4] + g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[5]) + (2.000000*h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += 2 * q;// OH
wdot[4] -= q;// O
wdot[5] -= q;// H2O
// d()/d[OH]
dqdci =  - k_r*2.000000*sc[3];
J[36] += 2 * dqdci;           // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[38] -= dqdci;               // dwdot[H2O]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[5];
J[47] += 2 * dqdci;           // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[49] -= dqdci;               // dwdot[H2O]/d[O]
// d()/d[H2O]
dqdci =  + k_f*sc[4];
J[58] += 2 * dqdci;           // dwdot[OH]/d[H2O]
J[59] -= dqdci;               // dwdot[O]/d[H2O]
J[60] -= dqdci;               // dwdot[H2O]/d[H2O]
// d()/dT
J[113] += 2 * dqdT;           // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[115] -= dqdT;               // dwdot[H2O]/dT

// reaction 9: H + HO2 <=> H2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[6];
k_f = 16600000
            * exp(- (412.63766595987) * invT);
dlnkfdT = (412.63766595987) * invT2;
// reverse
phi_r = sc[1]*sc[9];
Kc = exp(-g_RT[1] + g_RT[2] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[1] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] += q;// H2
wdot[2] -= q;// H
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[H2]
dqdci =  - k_r*sc[9];
J[12] += dqdci;               // dwdot[H2]/d[H2]
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[17] -= dqdci;               // dwdot[HO2]/d[H2]
J[20] += dqdci;               // dwdot[O2]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[6];
J[23] += dqdci;               // dwdot[H2]/d[H]
J[24] -= dqdci;               // dwdot[H]/d[H]
J[28] -= dqdci;               // dwdot[HO2]/d[H]
J[31] += dqdci;               // dwdot[O2]/d[H]
// d()/d[HO2]
dqdci =  + k_f*sc[2];
J[67] += dqdci;               // dwdot[H2]/d[HO2]
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[1];
J[100] += dqdci;              // dwdot[H2]/d[O2]
J[101] -= dqdci;              // dwdot[H]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[111] += dqdT;               // dwdot[H2]/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 10: H + HO2 <=> 2 OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[6];
k_f = 70800000
            * exp(- (150.964999741416) * invT);
dlnkfdT = (150.964999741416) * invT2;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = exp(g_RT[2] - 2.000000*g_RT[3] + g_RT[6]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (2.000000*h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += 2 * q;// OH
wdot[6] -= q;// HO2
// d()/d[H]
dqdci =  + k_f*sc[6];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += 2 * dqdci;           // dwdot[OH]/d[H]
J[28] -= dqdci;               // dwdot[HO2]/d[H]
// d()/d[OH]
dqdci =  - k_r*2.000000*sc[3];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += 2 * dqdci;           // dwdot[OH]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
// d()/d[HO2]
dqdci =  + k_f*sc[2];
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[69] += 2 * dqdci;           // dwdot[OH]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += 2 * dqdT;           // dwdot[OH]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT

// reaction 11: HO2 + O <=> O2 + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[6];
k_f = 32500000
           ;
dlnkfdT = 0.0;
// reverse
phi_r = sc[3]*sc[9];
Kc = exp(-g_RT[3] + g_RT[4] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[6]) + (h_RT[3] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += q;// OH
wdot[4] -= q;// O
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[OH]
dqdci =  - k_r*sc[9];
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
J[42] += dqdci;               // dwdot[O2]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[6];
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[50] -= dqdci;               // dwdot[HO2]/d[O]
J[53] += dqdci;               // dwdot[O2]/d[O]
// d()/d[HO2]
dqdci =  + k_f*sc[4];
J[69] += dqdci;               // dwdot[OH]/d[HO2]
J[70] -= dqdci;               // dwdot[O]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[3];
J[102] += dqdci;              // dwdot[OH]/d[O2]
J[103] -= dqdci;              // dwdot[O]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 12: HO2 + OH <=> H2O + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[6];
k_f = 28900000
            * exp(- (-251.60833290236) * invT);
dlnkfdT = (-251.60833290236) * invT2;
// reverse
phi_r = sc[5]*sc[9];
Kc = exp(g_RT[3] - g_RT[5] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[6]) + (h_RT[5] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[OH]
dqdci =  + k_f*sc[6];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
J[42] += dqdci;               // dwdot[O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[9];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] -= dqdci;               // dwdot[HO2]/d[H2O]
J[64] += dqdci;               // dwdot[O2]/d[H2O]
// d()/d[HO2]
dqdci =  + k_f*sc[3];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[5];
J[102] -= dqdci;              // dwdot[OH]/d[O2]
J[104] += dqdci;              // dwdot[H2O]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 13: 2 HO2 <=> H2O2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = (sc[6] * sc[6]);
k_f = 420000000
            * exp(- (6028.53565634054) * invT);
dlnkfdT = (6028.53565634054) * invT2;
// reverse
phi_r = sc[7]*sc[9];
Kc = exp(2.000000*g_RT[6] - g_RT[7] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[6]) + (h_RT[7] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[6] -= 2 * q;// HO2
wdot[7] += q;// H2O2
wdot[9] += q;// O2
// d()/d[HO2]
dqdci =  + k_f*2.000000*sc[6];
J[72] += -2 * dqdci;          // dwdot[HO2]/d[HO2]
J[73] += dqdci;               // dwdot[H2O2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[H2O2]
dqdci =  - k_r*sc[9];
J[83] += -2 * dqdci;          // dwdot[HO2]/d[H2O2]
J[84] += dqdci;               // dwdot[H2O2]/d[H2O2]
J[86] += dqdci;               // dwdot[O2]/d[H2O2]
// d()/d[O2]
dqdci =  - k_r*sc[7];
J[105] += -2 * dqdci;         // dwdot[HO2]/d[O2]
J[106] += dqdci;              // dwdot[H2O2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[116] += -2 * dqdT;          // dwdot[HO2]/dT
J[117] += dqdT;               // dwdot[H2O2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 14: 2 HO2 <=> H2O2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = (sc[6] * sc[6]);
k_f = 130000
            * exp(- (-819.739948595888) * invT);
dlnkfdT = (-819.739948595888) * invT2;
// reverse
phi_r = sc[7]*sc[9];
Kc = exp(2.000000*g_RT[6] - g_RT[7] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[6]) + (h_RT[7] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[6] -= 2 * q;// HO2
wdot[7] += q;// H2O2
wdot[9] += q;// O2
// d()/d[HO2]
dqdci =  + k_f*2.000000*sc[6];
J[72] += -2 * dqdci;          // dwdot[HO2]/d[HO2]
J[73] += dqdci;               // dwdot[H2O2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[H2O2]
dqdci =  - k_r*sc[9];
J[83] += -2 * dqdci;          // dwdot[HO2]/d[H2O2]
J[84] += dqdci;               // dwdot[H2O2]/d[H2O2]
J[86] += dqdci;               // dwdot[O2]/d[H2O2]
// d()/d[O2]
dqdci =  - k_r*sc[7];
J[105] += -2 * dqdci;         // dwdot[HO2]/d[O2]
J[106] += dqdci;              // dwdot[H2O2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[116] += -2 * dqdT;          // dwdot[HO2]/dT
J[117] += dqdT;               // dwdot[H2O2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 16: H + H2O2 <=> H2O + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[7];
k_f = 24100000
            * exp(- (1997.77016324474) * invT);
dlnkfdT = (1997.77016324474) * invT2;
// reverse
phi_r = sc[3]*sc[5];
Kc = exp(g_RT[2] - g_RT[3] - g_RT[5] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[3] + h_RT[5]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[5] += q;// H2O
wdot[7] -= q;// H2O2
// d()/d[H]
dqdci =  + k_f*sc[7];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[27] += dqdci;               // dwdot[H2O]/d[H]
J[29] -= dqdci;               // dwdot[H2O2]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[5];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[3];
J[57] -= dqdci;               // dwdot[H]/d[H2O]
J[58] += dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[H2O2]
dqdci =  + k_f*sc[2];
J[79] -= dqdci;               // dwdot[H]/d[H2O2]
J[80] += dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 17: H + H2O2 <=> H2 + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[7];
k_f = 60300000
            * exp(- (4000.57249314752) * invT);
dlnkfdT = (4000.57249314752) * invT2;
// reverse
phi_r = sc[1]*sc[6];
Kc = exp(-g_RT[1] + g_RT[2] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[1] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] += q;// H2
wdot[2] -= q;// H
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[H2]
dqdci =  - k_r*sc[6];
J[12] += dqdci;               // dwdot[H2]/d[H2]
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[17] += dqdci;               // dwdot[HO2]/d[H2]
J[18] -= dqdci;               // dwdot[H2O2]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[7];
J[23] += dqdci;               // dwdot[H2]/d[H]
J[24] -= dqdci;               // dwdot[H]/d[H]
J[28] += dqdci;               // dwdot[HO2]/d[H]
J[29] -= dqdci;               // dwdot[H2O2]/d[H]
// d()/d[HO2]
dqdci =  - k_r*sc[1];
J[67] += dqdci;               // dwdot[H2]/d[HO2]
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[2];
J[78] += dqdci;               // dwdot[H2]/d[H2O2]
J[79] -= dqdci;               // dwdot[H]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[111] += dqdT;               // dwdot[H2]/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 18: H2O2 + O <=> HO2 + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[7];
k_f = 9.55
            * exp(2 * logT - (1997.77016324474) * invT);
dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
// reverse
phi_r = sc[3]*sc[6];
Kc = exp(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[3] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += q;// OH
wdot[4] -= q;// O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  - k_r*sc[6];
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[7];
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[50] += dqdci;               // dwdot[HO2]/d[O]
J[51] -= dqdci;               // dwdot[H2O2]/d[O]
// d()/d[HO2]
dqdci =  - k_r*sc[3];
J[69] += dqdci;               // dwdot[OH]/d[HO2]
J[70] -= dqdci;               // dwdot[O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[4];
J[80] += dqdci;               // dwdot[OH]/d[H2O2]
J[81] -= dqdci;               // dwdot[O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 19: H2O2 + OH <=> H2O + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[7];
k_f = 1000000
           ;
dlnkfdT = 0.0;
// reverse
phi_r = sc[5]*sc[6];
Kc = exp(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[7]) + (h_RT[5] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  + k_f*sc[7];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[6];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] += dqdci;               // dwdot[HO2]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[HO2]
dqdci =  - k_r*sc[5];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[3];
J[80] -= dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 20: H2O2 + OH <=> H2O + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[7];
k_f = 580000000
            * exp(- (4810.75132509312) * invT);
dlnkfdT = (4810.75132509312) * invT2;
// reverse
phi_r = sc[5]*sc[6];
Kc = exp(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[7]) + (h_RT[5] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  + k_f*sc[7];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[6];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] += dqdci;               // dwdot[HO2]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[HO2]
dqdci =  - k_r*sc[5];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[3];
J[80] -= dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

amrex::Real c_R[10], dcRdT[10], e_RT[10];
amrex::Real * eh_RT;
if (HP == 1) {
cp_R(c_R, T);
dcvpRdT(dcRdT, T);
eh_RT = &h_RT[0];
}
else {
cv_R(c_R, T);
dcvpRdT(dcRdT, T);
speciesInternalEnergy(e_RT, T);
eh_RT = &e_RT[0];
}

amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
for (int k = 0; k < 10; ++k) {
cmix += c_R[k]*sc[k];
dcmixdT += dcRdT[k]*sc[k];
ehmix += eh_RT[k]*wdot[k];
dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[110+k];
}

amrex::Real cmixinv = 1.0/cmix;
amrex::Real tmp1 = ehmix*cmixinv;
amrex::Real tmp3 = cmixinv*T;
amrex::Real tmp2 = tmp1*tmp3;
amrex::Real dehmixdc;
// dTdot/d[X]
for (int k = 0; k < 10; ++k) {
dehmixdc = 0.0;
for (int m = 0; m < 10; ++m) {
dehmixdc += eh_RT[m]*J[k*11+m];
}
J[k*11+10] = tmp2*c_R[k] - tmp3*dehmixdc;
}
// dTdot/dT
J[120] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
#else
amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void DWDOT_SIMPLIFIED(amrex::Real *  J, const amrex::Real *  sc, const amrex::Real *  Tp, const int * HP)
{
amrex::Real c[10];

for (int k=0; k<10; k++) {
c[k] = 1.e6 * sc[k];
}

aJacobian_precond(J, c, *Tp, *HP);

// dwdot[k]/dT
// dTdot/d[X]
for (int k=0; k<10; k++) {
J[110+k] *= 1.e-6;
J[k*11+10] *= 1.e6;
}

}


// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void aJacobian(amrex::Real * J, const amrex::Real * sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
for (int i=0; i<121; i++) {
J[i] = 0.0;
}

amrex::Real wdot[10];
for (auto& val : wdot) {
val = 0.0;
}

const amrex::Real invT = 1.0 / T;
const amrex::Real invT2 = invT * invT;
const amrex::Real logT = log(T);

// reference concentration: P_atm / (RT) in inverse mol/m^3
amrex::Real refC = 101325 / 8.31446 / T;
amrex::Real refCinv = 1.0 / refC;

// compute the mixture concentration
amrex::Real mixture = 0.0;
for (int k = 0; k < 10; ++k) {
mixture += sc[k];
}

// compute the Gibbs free energy
amrex::Real g_RT[10];
gibbs(g_RT, T);

// compute the species enthalpy
amrex::Real h_RT[10];
speciesEnthalpy(h_RT, T);

amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
amrex::Real dqdci, dcdc_fac, dqdc[10];
amrex::Real Pr, fPr, F, k_0, logPr;
amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
const amrex::Real ln10 = log(10.0);
const amrex::Real log10e = 1.0/log(10.0);
// reaction 8: H + O2 (+M) <=> HO2 (+M)
// a third-body and pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.67 - 1)*sc[0] + ( 1.3 - 1)*sc[1] + ( 14 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[9];
k_f = 1480000
            * exp(0.6 * logT);
dlnkfdT = 0.6 * invT;
// pressure-fall-off
k_0 = 3.48e+16 * exp(-0.41 * logT - (-563.602665701286) * invT);
Pr = 1e-12 * alpha / k_f * k_0;
fPr = Pr / (1.0+Pr);
dlnk0dT = -0.41 * invT + (-563.602665701286) * invT2;
dlogPrdT = log10e*(dlnk0dT - dlnkfdT);
dlogfPrdT = dlogPrdT / (1.0+Pr);
// Troe form
logPr = log10(Pr);
Fcent1 = (1.-0.5)*exp(-T/1e-30);
Fcent2 = 0.5 * exp(-T/1e+30);
Fcent3 = exp(-1e+100 * invT);
Fcent = Fcent1 + Fcent2 + Fcent3;
logFcent = log10(Fcent);
troe_c = -.4 - .67 * logFcent;
troe_n = .75 - 1.27 * logFcent;
troePr_den = 1.0 / (troe_n - .14*(troe_c + logPr));
troePr = (troe_c + logPr) * troePr_den;
troe = 1.0 / (1.0 + troePr*troePr);
F = exp(M_LN10 * logFcent * troe);
dlogFcentdT = log10e/Fcent*( 
    -Fcent1/1e-30
    -Fcent2/1e+30
    + Fcent3*1e+100*invT2
);
dlogFdcn_fac = 2.0 * logFcent * troe*troe * troePr * troePr_den;
dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
dlogFdn = dlogFdcn_fac * troePr;
dlogFdlogPr = dlogFdc;
dlogFdT = dlogFcentdT*(troe - 0.67*dlogFdc - 1.27*dlogFdn) + dlogFdlogPr * dlogPrdT;
// reverse
phi_r = sc[6];
Kc = refCinv * exp(g_RT[2] - g_RT[6] + g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[9]) + (h_RT[6]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
Corr = fPr * F;
q = Corr * q_nocor;
dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);
dqdT = Corr *(dlnkfdT*k_f*phi_f - dkrdT*phi_r) + dlnCorrdT*q;
// update wdot
wdot[2] -= q;// H
wdot[6] += q;// HO2
wdot[9] -= q;// O2
// for convenience
k_f *= Corr;
k_r *= Corr;
dcdc_fac = q/alpha*(1.0/(Pr+1.0) + dlogFdlogPr);
if (consP == 1) {
// d()/d[XE]
dqdci = (0.67 - 1)*dcdc_fac;
J[2] -= dqdci;                // dwdot[H]/d[XE]
J[6] += dqdci;                // dwdot[HO2]/d[XE]
J[9] -= dqdci;                // dwdot[O2]/d[XE]
// d()/d[H2]
dqdci = (1.3 - 1)*dcdc_fac;
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[17] += dqdci;               // dwdot[HO2]/d[H2]
J[20] -= dqdci;               // dwdot[O2]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[9];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[28] += dqdci;               // dwdot[HO2]/d[H]
J[31] -= dqdci;               // dwdot[O2]/d[H]
// d()/d[H2O]
dqdci = (14 - 1)*dcdc_fac;
J[57] -= dqdci;               // dwdot[H]/d[H2O]
J[61] += dqdci;               // dwdot[HO2]/d[H2O]
J[64] -= dqdci;               // dwdot[O2]/d[H2O]
// d()/d[HO2]
dqdci =  - k_r;
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[75] -= dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  + k_f*sc[2];
J[101] -= dqdci;              // dwdot[H]/d[O2]
J[105] += dqdci;              // dwdot[HO2]/d[O2]
J[108] -= dqdci;              // dwdot[O2]/d[O2]
}
else {
dqdc[0] = 0.67*dcdc_fac;
dqdc[1] = 1.3*dcdc_fac;
dqdc[2] = dcdc_fac + k_f*sc[9];
dqdc[3] = dcdc_fac;
dqdc[4] = dcdc_fac;
dqdc[5] = 14*dcdc_fac;
dqdc[6] = dcdc_fac - k_r;
dqdc[7] = dcdc_fac;
dqdc[8] = dcdc_fac;
dqdc[9] = dcdc_fac + k_f*sc[2];
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+6] += dqdc[k];
J[11*k+9] -= dqdc[k];
}
}
J[112] -= dqdT;// dwdot[H]/dT
J[116] += dqdT;// dwdot[HO2]/dT
J[119] -= dqdT;// dwdot[O2]/dT

// reaction 15: H2O2 (+M) <=> 2 OH (+M)
// a third-body and pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.45 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[7];
k_f = 295000000000000
            * exp(- (24355.6866249484) * invT);
dlnkfdT = (24355.6866249484) * invT2;
// pressure-fall-off
k_0 = 1.27e+17 * exp(-(22896.3582941147) * invT);
Pr = 1e-6 * alpha / k_f * k_0;
fPr = Pr / (1.0+Pr);
dlnk0dT = (22896.3582941147) * invT2;
dlogPrdT = log10e*(dlnk0dT - dlnkfdT);
dlogfPrdT = dlogPrdT / (1.0+Pr);
// Troe form
logPr = log10(Pr);
Fcent1 = (1.-0.5)*exp(-T/1e-30);
Fcent2 = 0.5 * exp(-T/1e+30);
Fcent3 = 0.;
Fcent = Fcent1 + Fcent2 + Fcent3;
logFcent = log10(Fcent);
troe_c = -.4 - .67 * logFcent;
troe_n = .75 - 1.27 * logFcent;
troePr_den = 1.0 / (troe_n - .14*(troe_c + logPr));
troePr = (troe_c + logPr) * troePr_den;
troe = 1.0 / (1.0 + troePr*troePr);
F = exp(M_LN10 * logFcent * troe);
dlogFcentdT = log10e/Fcent*( 
    -Fcent1/1e-30
    -Fcent2/1e+30
);
dlogFdcn_fac = 2.0 * logFcent * troe*troe * troePr * troePr_den;
dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
dlogFdn = dlogFdcn_fac * troePr;
dlogFdlogPr = dlogFdc;
dlogFdT = dlogFcentdT*(troe - 0.67*dlogFdc - 1.27*dlogFdn) + dlogFdlogPr * dlogPrdT;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = refC * exp(-2.000000*g_RT[3] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[7]) + (2.000000*h_RT[3]) - 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
Corr = fPr * F;
q = Corr * q_nocor;
dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);
dqdT = Corr *(dlnkfdT*k_f*phi_f - dkrdT*phi_r) + dlnCorrdT*q;
// update wdot
wdot[3] += 2 * q;// OH
wdot[7] -= q;// H2O2
// for convenience
k_f *= Corr;
k_r *= Corr;
dcdc_fac = q/alpha*(1.0/(Pr+1.0) + dlogFdlogPr);
if (consP == 1) {
// d()/d[XE]
dqdci = (0.45 - 1)*dcdc_fac;
J[3] += 2 * dqdci;            // dwdot[OH]/d[XE]
J[7] -= dqdci;                // dwdot[H2O2]/d[XE]
// d()/d[H2]
dqdci = (2.5 - 1)*dcdc_fac;
J[14] += 2 * dqdci;           // dwdot[OH]/d[H2]
J[18] -= dqdci;               // dwdot[H2O2]/d[H2]
// d()/d[OH]
dqdci =  - k_r*2.000000*sc[3];
J[36] += 2 * dqdci;           // dwdot[OH]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci = (12 - 1)*dcdc_fac;
J[58] += 2 * dqdci;           // dwdot[OH]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[H2O2]
dqdci =  + k_f;
J[80] += 2 * dqdci;           // dwdot[OH]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
}
else {
dqdc[0] = 0.45*dcdc_fac;
dqdc[1] = 2.5*dcdc_fac;
dqdc[2] = dcdc_fac;
dqdc[3] = dcdc_fac - k_r*2.000000*sc[3];
dqdc[4] = dcdc_fac;
dqdc[5] = 12*dcdc_fac;
dqdc[6] = dcdc_fac;
dqdc[7] = dcdc_fac + k_f;
dqdc[8] = dcdc_fac;
dqdc[9] = dcdc_fac;
for (int k=0; k<10; k++) {
J[11*k+3] += 2 * dqdc[k];
J[11*k+7] -= dqdc[k];
}
}
J[113] += 2 * dqdT;// dwdot[OH]/dT
J[117] -= dqdT;// dwdot[H2O2]/dT

// reaction 4: H2 + M <=> 2 H + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[1];
k_f = 45700000000000
            * exp(-1.4 * logT - (52888.071576076) * invT);
dlnkfdT = -1.4 * invT + (52888.071576076) * invT2;
// reverse
phi_r = (sc[2] * sc[2]);
Kc = refC * exp(g_RT[1] - 2.000000*g_RT[2]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1]) + (2.000000*h_RT[2]) - 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += 2 * q;// H
// for convenience
k_f *= alpha;
k_r *= alpha;
if (consP == 1) {
// d()/d[H2]
dqdci = (2.5 - 1)*q_nocor + k_f;
J[12] -= dqdci;               // dwdot[H2]/d[H2]
J[13] += 2 * dqdci;           // dwdot[H]/d[H2]
// d()/d[H]
dqdci =  - k_r*2.000000*sc[2];
J[23] -= dqdci;               // dwdot[H2]/d[H]
J[24] += 2 * dqdci;           // dwdot[H]/d[H]
// d()/d[H2O]
dqdci = (12 - 1)*q_nocor;
J[56] -= dqdci;               // dwdot[H2]/d[H2O]
J[57] += 2 * dqdci;           // dwdot[H]/d[H2O]
}
else {
dqdc[0] = q_nocor;
dqdc[1] = 2.5*q_nocor + k_f;
dqdc[2] = q_nocor - k_r*2.000000*sc[2];
dqdc[3] = q_nocor;
dqdc[4] = q_nocor;
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+1] -= dqdc[k];
J[11*k+2] += 2 * dqdc[k];
}
}
J[111] -= dqdT;// dwdot[H2]/dT
J[112] += 2 * dqdT;// dwdot[H]/dT

// reaction 5: 2 O + M <=> O2 + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.83 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = (sc[4] * sc[4]);
k_f = 6170
            * exp(-0.5 * logT);
dlnkfdT = -0.5 * invT;
// reverse
phi_r = sc[9];
Kc = refCinv * exp(2.000000*g_RT[4] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[4]) + (h_RT[9]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[4] -= 2 * q;// O
wdot[9] += q;// O2
// for convenience
k_f *= alpha;
k_r *= alpha;
if (consP == 1) {
// d()/d[XE]
dqdci = (0.83 - 1)*q_nocor;
J[4] += -2 * dqdci;           // dwdot[O]/d[XE]
J[9] += dqdci;                // dwdot[O2]/d[XE]
// d()/d[H2]
dqdci = (2.5 - 1)*q_nocor;
J[15] += -2 * dqdci;          // dwdot[O]/d[H2]
J[20] += dqdci;               // dwdot[O2]/d[H2]
// d()/d[O]
dqdci =  + k_f*2.000000*sc[4];
J[48] += -2 * dqdci;          // dwdot[O]/d[O]
J[53] += dqdci;               // dwdot[O2]/d[O]
// d()/d[H2O]
dqdci = (12 - 1)*q_nocor;
J[59] += -2 * dqdci;          // dwdot[O]/d[H2O]
J[64] += dqdci;               // dwdot[O2]/d[H2O]
// d()/d[O2]
dqdci =  - k_r;
J[103] += -2 * dqdci;         // dwdot[O]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
}
else {
dqdc[0] = 0.83*q_nocor;
dqdc[1] = 2.5*q_nocor;
dqdc[2] = q_nocor;
dqdc[3] = q_nocor;
dqdc[4] = q_nocor + k_f*2.000000*sc[4];
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor - k_r;
for (int k=0; k<10; k++) {
J[11*k+4] += -2 * dqdc[k];
J[11*k+9] += dqdc[k];
}
}
J[114] += -2 * dqdT;// dwdot[O]/dT
J[119] += dqdT;// dwdot[O2]/dT

// reaction 6: H + O + M <=> OH + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.75 - 1)*sc[0] + ( 2.5 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[4];
k_f = 4720000
            * exp(-1 * logT);
dlnkfdT = -1 * invT;
// reverse
phi_r = sc[3];
Kc = refCinv * exp(g_RT[2] - g_RT[3] + g_RT[4]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[3]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[4] -= q;// O
// for convenience
k_f *= alpha;
k_r *= alpha;
if (consP == 1) {
// d()/d[XE]
dqdci = (0.75 - 1)*q_nocor;
J[2] -= dqdci;                // dwdot[H]/d[XE]
J[3] += dqdci;                // dwdot[OH]/d[XE]
J[4] -= dqdci;                // dwdot[O]/d[XE]
// d()/d[H2]
dqdci = (2.5 - 1)*q_nocor;
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[14] += dqdci;               // dwdot[OH]/d[H2]
J[15] -= dqdci;               // dwdot[O]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[4];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[26] -= dqdci;               // dwdot[O]/d[H]
// d()/d[OH]
dqdci =  - k_r;
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[2];
J[46] -= dqdci;               // dwdot[H]/d[O]
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
// d()/d[H2O]
dqdci = (12 - 1)*q_nocor;
J[57] -= dqdci;               // dwdot[H]/d[H2O]
J[58] += dqdci;               // dwdot[OH]/d[H2O]
J[59] -= dqdci;               // dwdot[O]/d[H2O]
}
else {
dqdc[0] = 0.75*q_nocor;
dqdc[1] = 2.5*q_nocor;
dqdc[2] = q_nocor + k_f*sc[4];
dqdc[3] = q_nocor - k_r;
dqdc[4] = q_nocor + k_f*sc[2];
dqdc[5] = 12*q_nocor;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+3] += dqdc[k];
J[11*k+4] -= dqdc[k];
}
}
J[112] -= dqdT;// dwdot[H]/dT
J[113] += dqdT;// dwdot[OH]/dT
J[114] -= dqdT;// dwdot[O]/dT

// reaction 7: H + OH + M <=> H2O + M
// a third-body and non-pressure-fall-off reaction
// 3-body correction factor
alpha = mixture + ( 0.38 - 1)*sc[0] + ( 0.73 - 1)*sc[1] + ( 12 - 1)*sc[5];
// forward
phi_f = sc[2]*sc[3];
k_f = 45000000000
            * exp(-2 * logT);
dlnkfdT = -2 * invT;
// reverse
phi_r = sc[5];
Kc = refCinv * exp(g_RT[2] + g_RT[3] - g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[5]) + 1.000000);
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q_nocor = k_f*phi_f - k_r*phi_r;
q = alpha * q_nocor;
dqdT = alpha *(dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] -= q;// OH
wdot[5] += q;// H2O
// for convenience
k_f *= alpha;
k_r *= alpha;
if (consP == 1) {
// d()/d[XE]
dqdci = (0.38 - 1)*q_nocor;
J[2] -= dqdci;                // dwdot[H]/d[XE]
J[3] -= dqdci;                // dwdot[OH]/d[XE]
J[5] += dqdci;                // dwdot[H2O]/d[XE]
// d()/d[H2]
dqdci = (0.73 - 1)*q_nocor;
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[14] -= dqdci;               // dwdot[OH]/d[H2]
J[16] += dqdci;               // dwdot[H2O]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[3];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] -= dqdci;               // dwdot[OH]/d[H]
J[27] += dqdci;               // dwdot[H2O]/d[H]
// d()/d[OH]
dqdci =  + k_f*sc[2];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
// d()/d[H2O]
dqdci = (12 - 1)*q_nocor - k_r;
J[57] -= dqdci;               // dwdot[H]/d[H2O]
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
}
else {
dqdc[0] = 0.38*q_nocor;
dqdc[1] = 0.73*q_nocor;
dqdc[2] = q_nocor + k_f*sc[3];
dqdc[3] = q_nocor + k_f*sc[2];
dqdc[4] = q_nocor;
dqdc[5] = 12*q_nocor - k_r;
dqdc[6] = q_nocor;
dqdc[7] = q_nocor;
dqdc[8] = q_nocor;
dqdc[9] = q_nocor;
for (int k=0; k<10; k++) {
J[11*k+2] -= dqdc[k];
J[11*k+3] -= dqdc[k];
J[11*k+5] += dqdc[k];
}
}
J[112] -= dqdT;// dwdot[H]/dT
J[113] -= dqdT;// dwdot[OH]/dT
J[115] += dqdT;// dwdot[H2O]/dT

// reaction 0: H + O2 <=> O + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[9];
k_f = 191000000
            * exp(- (8272.88198582959) * invT);
dlnkfdT = (8272.88198582959) * invT2;
// reverse
phi_r = sc[3]*sc[4];
Kc = exp(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[9]) + (h_RT[3] + h_RT[4]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[4] += q;// O
wdot[9] -= q;// O2
// d()/d[H]
dqdci =  + k_f*sc[9];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[26] += dqdci;               // dwdot[O]/d[H]
J[31] -= dqdci;               // dwdot[O2]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[4];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] += dqdci;               // dwdot[O]/d[OH]
J[42] -= dqdci;               // dwdot[O2]/d[OH]
// d()/d[O]
dqdci =  - k_r*sc[3];
J[46] -= dqdci;               // dwdot[H]/d[O]
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] += dqdci;               // dwdot[O]/d[O]
J[53] -= dqdci;               // dwdot[O2]/d[O]
// d()/d[O2]
dqdci =  + k_f*sc[2];
J[101] -= dqdci;              // dwdot[H]/d[O2]
J[102] += dqdci;              // dwdot[OH]/d[O2]
J[103] += dqdci;              // dwdot[O]/d[O2]
J[108] -= dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] += dqdT;               // dwdot[O]/dT
J[119] -= dqdT;               // dwdot[O2]/dT

// reaction 1: H2 + O <=> H + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[1]*sc[4];
k_f = 0.0508
            * exp(2.67 * logT - (3166.2392612433) * invT);
dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
// reverse
phi_r = sc[2]*sc[3];
Kc = exp(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += q;// H
wdot[3] += q;// OH
wdot[4] -= q;// O
// d()/d[H2]
dqdci =  + k_f*sc[4];
J[12] -= dqdci;               // dwdot[H2]/d[H2]
J[13] += dqdci;               // dwdot[H]/d[H2]
J[14] += dqdci;               // dwdot[OH]/d[H2]
J[15] -= dqdci;               // dwdot[O]/d[H2]
// d()/d[H]
dqdci =  - k_r*sc[3];
J[23] -= dqdci;               // dwdot[H2]/d[H]
J[24] += dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[26] -= dqdci;               // dwdot[O]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[2];
J[34] -= dqdci;               // dwdot[H2]/d[OH]
J[35] += dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[1];
J[45] -= dqdci;               // dwdot[H2]/d[O]
J[46] += dqdci;               // dwdot[H]/d[O]
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
// d()/dT
J[111] -= dqdT;               // dwdot[H2]/dT
J[112] += dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT

// reaction 2: H2 + OH <=> H + H2O
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[1]*sc[3];
k_f = 216
            * exp(1.51 * logT - (1726.03316371019) * invT);
dlnkfdT = 1.51 * invT + (1726.03316371019) * invT2;
// reverse
phi_r = sc[2]*sc[5];
Kc = exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[2] + h_RT[5]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] -= q;// H2
wdot[2] += q;// H
wdot[3] -= q;// OH
wdot[5] += q;// H2O
// d()/d[H2]
dqdci =  + k_f*sc[3];
J[12] -= dqdci;               // dwdot[H2]/d[H2]
J[13] += dqdci;               // dwdot[H]/d[H2]
J[14] -= dqdci;               // dwdot[OH]/d[H2]
J[16] += dqdci;               // dwdot[H2O]/d[H2]
// d()/d[H]
dqdci =  - k_r*sc[5];
J[23] -= dqdci;               // dwdot[H2]/d[H]
J[24] += dqdci;               // dwdot[H]/d[H]
J[25] -= dqdci;               // dwdot[OH]/d[H]
J[27] += dqdci;               // dwdot[H2O]/d[H]
// d()/d[OH]
dqdci =  + k_f*sc[1];
J[34] -= dqdci;               // dwdot[H2]/d[OH]
J[35] += dqdci;               // dwdot[H]/d[OH]
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[2];
J[56] -= dqdci;               // dwdot[H2]/d[H2O]
J[57] += dqdci;               // dwdot[H]/d[H2O]
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
// d()/dT
J[111] -= dqdT;               // dwdot[H2]/dT
J[112] += dqdT;               // dwdot[H]/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT

// reaction 3: H2O + O <=> 2 OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[5];
k_f = 2.97
            * exp(2.02 * logT - (6743.10332178324) * invT);
dlnkfdT = 2.02 * invT + (6743.10332178324) * invT2;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = exp(-2.000000*g_RT[3] + g_RT[4] + g_RT[5]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[5]) + (2.000000*h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += 2 * q;// OH
wdot[4] -= q;// O
wdot[5] -= q;// H2O
// d()/d[OH]
dqdci =  - k_r*2.000000*sc[3];
J[36] += 2 * dqdci;           // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[38] -= dqdci;               // dwdot[H2O]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[5];
J[47] += 2 * dqdci;           // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[49] -= dqdci;               // dwdot[H2O]/d[O]
// d()/d[H2O]
dqdci =  + k_f*sc[4];
J[58] += 2 * dqdci;           // dwdot[OH]/d[H2O]
J[59] -= dqdci;               // dwdot[O]/d[H2O]
J[60] -= dqdci;               // dwdot[H2O]/d[H2O]
// d()/dT
J[113] += 2 * dqdT;           // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[115] -= dqdT;               // dwdot[H2O]/dT

// reaction 9: H + HO2 <=> H2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[6];
k_f = 16600000
            * exp(- (412.63766595987) * invT);
dlnkfdT = (412.63766595987) * invT2;
// reverse
phi_r = sc[1]*sc[9];
Kc = exp(-g_RT[1] + g_RT[2] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[1] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] += q;// H2
wdot[2] -= q;// H
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[H2]
dqdci =  - k_r*sc[9];
J[12] += dqdci;               // dwdot[H2]/d[H2]
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[17] -= dqdci;               // dwdot[HO2]/d[H2]
J[20] += dqdci;               // dwdot[O2]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[6];
J[23] += dqdci;               // dwdot[H2]/d[H]
J[24] -= dqdci;               // dwdot[H]/d[H]
J[28] -= dqdci;               // dwdot[HO2]/d[H]
J[31] += dqdci;               // dwdot[O2]/d[H]
// d()/d[HO2]
dqdci =  + k_f*sc[2];
J[67] += dqdci;               // dwdot[H2]/d[HO2]
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[1];
J[100] += dqdci;              // dwdot[H2]/d[O2]
J[101] -= dqdci;              // dwdot[H]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[111] += dqdT;               // dwdot[H2]/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 10: H + HO2 <=> 2 OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[6];
k_f = 70800000
            * exp(- (150.964999741416) * invT);
dlnkfdT = (150.964999741416) * invT2;
// reverse
phi_r = (sc[3] * sc[3]);
Kc = exp(g_RT[2] - 2.000000*g_RT[3] + g_RT[6]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (2.000000*h_RT[3]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += 2 * q;// OH
wdot[6] -= q;// HO2
// d()/d[H]
dqdci =  + k_f*sc[6];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += 2 * dqdci;           // dwdot[OH]/d[H]
J[28] -= dqdci;               // dwdot[HO2]/d[H]
// d()/d[OH]
dqdci =  - k_r*2.000000*sc[3];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += 2 * dqdci;           // dwdot[OH]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
// d()/d[HO2]
dqdci =  + k_f*sc[2];
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[69] += 2 * dqdci;           // dwdot[OH]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += 2 * dqdT;           // dwdot[OH]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT

// reaction 11: HO2 + O <=> O2 + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[6];
k_f = 32500000
           ;
dlnkfdT = 0.0;
// reverse
phi_r = sc[3]*sc[9];
Kc = exp(-g_RT[3] + g_RT[4] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[6]) + (h_RT[3] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += q;// OH
wdot[4] -= q;// O
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[OH]
dqdci =  - k_r*sc[9];
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
J[42] += dqdci;               // dwdot[O2]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[6];
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[50] -= dqdci;               // dwdot[HO2]/d[O]
J[53] += dqdci;               // dwdot[O2]/d[O]
// d()/d[HO2]
dqdci =  + k_f*sc[4];
J[69] += dqdci;               // dwdot[OH]/d[HO2]
J[70] -= dqdci;               // dwdot[O]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[3];
J[102] += dqdci;              // dwdot[OH]/d[O2]
J[103] -= dqdci;              // dwdot[O]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 12: HO2 + OH <=> H2O + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[6];
k_f = 28900000
            * exp(- (-251.60833290236) * invT);
dlnkfdT = (-251.60833290236) * invT2;
// reverse
phi_r = sc[5]*sc[9];
Kc = exp(g_RT[3] - g_RT[5] + g_RT[6] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[6]) + (h_RT[5] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] -= q;// HO2
wdot[9] += q;// O2
// d()/d[OH]
dqdci =  + k_f*sc[6];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] -= dqdci;               // dwdot[HO2]/d[OH]
J[42] += dqdci;               // dwdot[O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[9];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] -= dqdci;               // dwdot[HO2]/d[H2O]
J[64] += dqdci;               // dwdot[O2]/d[H2O]
// d()/d[HO2]
dqdci =  + k_f*sc[3];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] -= dqdci;               // dwdot[HO2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[O2]
dqdci =  - k_r*sc[5];
J[102] -= dqdci;              // dwdot[OH]/d[O2]
J[104] += dqdci;              // dwdot[H2O]/d[O2]
J[105] -= dqdci;              // dwdot[HO2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] -= dqdT;               // dwdot[HO2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 13: 2 HO2 <=> H2O2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = (sc[6] * sc[6]);
k_f = 420000000
            * exp(- (6028.53565634054) * invT);
dlnkfdT = (6028.53565634054) * invT2;
// reverse
phi_r = sc[7]*sc[9];
Kc = exp(2.000000*g_RT[6] - g_RT[7] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[6]) + (h_RT[7] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[6] -= 2 * q;// HO2
wdot[7] += q;// H2O2
wdot[9] += q;// O2
// d()/d[HO2]
dqdci =  + k_f*2.000000*sc[6];
J[72] += -2 * dqdci;          // dwdot[HO2]/d[HO2]
J[73] += dqdci;               // dwdot[H2O2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[H2O2]
dqdci =  - k_r*sc[9];
J[83] += -2 * dqdci;          // dwdot[HO2]/d[H2O2]
J[84] += dqdci;               // dwdot[H2O2]/d[H2O2]
J[86] += dqdci;               // dwdot[O2]/d[H2O2]
// d()/d[O2]
dqdci =  - k_r*sc[7];
J[105] += -2 * dqdci;         // dwdot[HO2]/d[O2]
J[106] += dqdci;              // dwdot[H2O2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[116] += -2 * dqdT;          // dwdot[HO2]/dT
J[117] += dqdT;               // dwdot[H2O2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 14: 2 HO2 <=> H2O2 + O2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = (sc[6] * sc[6]);
k_f = 130000
            * exp(- (-819.739948595888) * invT);
dlnkfdT = (-819.739948595888) * invT2;
// reverse
phi_r = sc[7]*sc[9];
Kc = exp(2.000000*g_RT[6] - g_RT[7] - g_RT[9]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(2.000000*h_RT[6]) + (h_RT[7] + h_RT[9]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[6] -= 2 * q;// HO2
wdot[7] += q;// H2O2
wdot[9] += q;// O2
// d()/d[HO2]
dqdci =  + k_f*2.000000*sc[6];
J[72] += -2 * dqdci;          // dwdot[HO2]/d[HO2]
J[73] += dqdci;               // dwdot[H2O2]/d[HO2]
J[75] += dqdci;               // dwdot[O2]/d[HO2]
// d()/d[H2O2]
dqdci =  - k_r*sc[9];
J[83] += -2 * dqdci;          // dwdot[HO2]/d[H2O2]
J[84] += dqdci;               // dwdot[H2O2]/d[H2O2]
J[86] += dqdci;               // dwdot[O2]/d[H2O2]
// d()/d[O2]
dqdci =  - k_r*sc[7];
J[105] += -2 * dqdci;         // dwdot[HO2]/d[O2]
J[106] += dqdci;              // dwdot[H2O2]/d[O2]
J[108] += dqdci;              // dwdot[O2]/d[O2]
// d()/dT
J[116] += -2 * dqdT;          // dwdot[HO2]/dT
J[117] += dqdT;               // dwdot[H2O2]/dT
J[119] += dqdT;               // dwdot[O2]/dT

// reaction 16: H + H2O2 <=> H2O + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[7];
k_f = 24100000
            * exp(- (1997.77016324474) * invT);
dlnkfdT = (1997.77016324474) * invT2;
// reverse
phi_r = sc[3]*sc[5];
Kc = exp(g_RT[2] - g_RT[3] - g_RT[5] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[3] + h_RT[5]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[2] -= q;// H
wdot[3] += q;// OH
wdot[5] += q;// H2O
wdot[7] -= q;// H2O2
// d()/d[H]
dqdci =  + k_f*sc[7];
J[24] -= dqdci;               // dwdot[H]/d[H]
J[25] += dqdci;               // dwdot[OH]/d[H]
J[27] += dqdci;               // dwdot[H2O]/d[H]
J[29] -= dqdci;               // dwdot[H2O2]/d[H]
// d()/d[OH]
dqdci =  - k_r*sc[5];
J[35] -= dqdci;               // dwdot[H]/d[OH]
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[3];
J[57] -= dqdci;               // dwdot[H]/d[H2O]
J[58] += dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[H2O2]
dqdci =  + k_f*sc[2];
J[79] -= dqdci;               // dwdot[H]/d[H2O2]
J[80] += dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 17: H + H2O2 <=> H2 + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[2]*sc[7];
k_f = 60300000
            * exp(- (4000.57249314752) * invT);
dlnkfdT = (4000.57249314752) * invT2;
// reverse
phi_r = sc[1]*sc[6];
Kc = exp(-g_RT[1] + g_RT[2] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[1] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[1] += q;// H2
wdot[2] -= q;// H
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[H2]
dqdci =  - k_r*sc[6];
J[12] += dqdci;               // dwdot[H2]/d[H2]
J[13] -= dqdci;               // dwdot[H]/d[H2]
J[17] += dqdci;               // dwdot[HO2]/d[H2]
J[18] -= dqdci;               // dwdot[H2O2]/d[H2]
// d()/d[H]
dqdci =  + k_f*sc[7];
J[23] += dqdci;               // dwdot[H2]/d[H]
J[24] -= dqdci;               // dwdot[H]/d[H]
J[28] += dqdci;               // dwdot[HO2]/d[H]
J[29] -= dqdci;               // dwdot[H2O2]/d[H]
// d()/d[HO2]
dqdci =  - k_r*sc[1];
J[67] += dqdci;               // dwdot[H2]/d[HO2]
J[68] -= dqdci;               // dwdot[H]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[2];
J[78] += dqdci;               // dwdot[H2]/d[H2O2]
J[79] -= dqdci;               // dwdot[H]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[111] += dqdT;               // dwdot[H2]/dT
J[112] -= dqdT;               // dwdot[H]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 18: H2O2 + O <=> HO2 + OH
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[4]*sc[7];
k_f = 9.55
            * exp(2 * logT - (1997.77016324474) * invT);
dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
// reverse
phi_r = sc[3]*sc[6];
Kc = exp(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[3] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] += q;// OH
wdot[4] -= q;// O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  - k_r*sc[6];
J[36] += dqdci;               // dwdot[OH]/d[OH]
J[37] -= dqdci;               // dwdot[O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[O]
dqdci =  + k_f*sc[7];
J[47] += dqdci;               // dwdot[OH]/d[O]
J[48] -= dqdci;               // dwdot[O]/d[O]
J[50] += dqdci;               // dwdot[HO2]/d[O]
J[51] -= dqdci;               // dwdot[H2O2]/d[O]
// d()/d[HO2]
dqdci =  - k_r*sc[3];
J[69] += dqdci;               // dwdot[OH]/d[HO2]
J[70] -= dqdci;               // dwdot[O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[4];
J[80] += dqdci;               // dwdot[OH]/d[H2O2]
J[81] -= dqdci;               // dwdot[O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] += dqdT;               // dwdot[OH]/dT
J[114] -= dqdT;               // dwdot[O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 19: H2O2 + OH <=> H2O + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[7];
k_f = 1000000
           ;
dlnkfdT = 0.0;
// reverse
phi_r = sc[5]*sc[6];
Kc = exp(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[7]) + (h_RT[5] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  + k_f*sc[7];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[6];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] += dqdci;               // dwdot[HO2]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[HO2]
dqdci =  - k_r*sc[5];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[3];
J[80] -= dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

// reaction 20: H2O2 + OH <=> H2O + HO2
// a non-third-body and non-pressure-fall-off reaction
// forward
phi_f = sc[3]*sc[7];
k_f = 580000000
            * exp(- (4810.75132509312) * invT);
dlnkfdT = (4810.75132509312) * invT2;
// reverse
phi_r = sc[5]*sc[6];
Kc = exp(g_RT[3] - g_RT[5] - g_RT[6] + g_RT[7]);
k_r = k_f / Kc;
dlnKcdT = invT * (-(h_RT[3] + h_RT[7]) + (h_RT[5] + h_RT[6]));
dkrdT = (dlnkfdT - dlnKcdT)*k_r;
// rate of progress
q = k_f*phi_f - k_r*phi_r;
dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
// update wdot
wdot[3] -= q;// OH
wdot[5] += q;// H2O
wdot[6] += q;// HO2
wdot[7] -= q;// H2O2
// d()/d[OH]
dqdci =  + k_f*sc[7];
J[36] -= dqdci;               // dwdot[OH]/d[OH]
J[38] += dqdci;               // dwdot[H2O]/d[OH]
J[39] += dqdci;               // dwdot[HO2]/d[OH]
J[40] -= dqdci;               // dwdot[H2O2]/d[OH]
// d()/d[H2O]
dqdci =  - k_r*sc[6];
J[58] -= dqdci;               // dwdot[OH]/d[H2O]
J[60] += dqdci;               // dwdot[H2O]/d[H2O]
J[61] += dqdci;               // dwdot[HO2]/d[H2O]
J[62] -= dqdci;               // dwdot[H2O2]/d[H2O]
// d()/d[HO2]
dqdci =  - k_r*sc[5];
J[69] -= dqdci;               // dwdot[OH]/d[HO2]
J[71] += dqdci;               // dwdot[H2O]/d[HO2]
J[72] += dqdci;               // dwdot[HO2]/d[HO2]
J[73] -= dqdci;               // dwdot[H2O2]/d[HO2]
// d()/d[H2O2]
dqdci =  + k_f*sc[3];
J[80] -= dqdci;               // dwdot[OH]/d[H2O2]
J[82] += dqdci;               // dwdot[H2O]/d[H2O2]
J[83] += dqdci;               // dwdot[HO2]/d[H2O2]
J[84] -= dqdci;               // dwdot[H2O2]/d[H2O2]
// d()/dT
J[113] -= dqdT;               // dwdot[OH]/dT
J[115] += dqdT;               // dwdot[H2O]/dT
J[116] += dqdT;               // dwdot[HO2]/dT
J[117] -= dqdT;               // dwdot[H2O2]/dT

amrex::Real c_R[10], dcRdT[10], e_RT[10];
amrex::Real * eh_RT;
if (consP == 1) {
cp_R(c_R, T);
dcvpRdT(dcRdT, T);
eh_RT = &h_RT[0];
}
else {
cv_R(c_R, T);
dcvpRdT(dcRdT, T);
speciesInternalEnergy(e_RT, T);
eh_RT = &e_RT[0];
}

amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
for (int k = 0; k < 10; ++k) {
cmix += c_R[k]*sc[k];
dcmixdT += dcRdT[k]*sc[k];
ehmix += eh_RT[k]*wdot[k];
dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[110+k];
}

amrex::Real cmixinv = 1.0/cmix;
amrex::Real tmp1 = ehmix*cmixinv;
amrex::Real tmp3 = cmixinv*T;
amrex::Real tmp2 = tmp1*tmp3;
amrex::Real dehmixdc;
// dTdot/d[X]
for (int k = 0; k < 10; ++k) {
dehmixdc = 0.0;
for (int m = 0; m < 10; ++m) {
dehmixdc += eh_RT[m]*J[k*11+m];
}
J[k*11+10] = tmp2*c_R[k] - tmp3*dehmixdc;
}
// dTdot/dT
J[120] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
#else
amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void DWDOT(amrex::Real *  J, const amrex::Real *  sc, const amrex::Real *  Tp, const int * consP)
{
amrex::Real c[10];

for (int k=0; k<10; k++) {
c[k] = 1.e6 * sc[k];
}

aJacobian(J, c, *Tp, *consP);

// dwdot[k]/dT
// dTdot/d[X]
for (int k=0; k<10; k++) {
J[110+k] *= 1.e-6;
J[k*11+10] *= 1.e6;
}

}


// Transport function declarations 

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetLENIMC(int* LENIMC ) {
*LENIMC = 42;}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetLENRMC(int* LENRMC ) {
*LENRMC = 2250;}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetNO(int* NO ) {
*NO = 4;}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetKK(int* KK ) {
*KK = 10;}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetNLITE(int* NLITE ) {
*NLITE = 2;}


// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetPATM(amrex::Real* PATM) {
*PATM =   0.1013250000000000E+07;}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetWT(amrex::Real* WT ) {
WT[0] = 1.31293000E+02;
WT[1] = 2.01600000E+00;
WT[2] = 1.00800000E+00;
WT[3] = 1.70070000E+01;
WT[4] = 1.59990000E+01;
WT[5] = 1.80150000E+01;
WT[6] = 3.30060000E+01;
WT[7] = 3.40140000E+01;
WT[8] = 2.80140000E+01;
WT[9] = 3.19980000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetEPS(amrex::Real* EPS ) {
EPS[0] = 2.82290000E+02;
EPS[1] = 3.80000000E+01;
EPS[2] = 1.45000000E+02;
EPS[3] = 8.00000000E+01;
EPS[4] = 8.00000000E+01;
EPS[5] = 5.72400000E+02;
EPS[6] = 1.07400000E+02;
EPS[7] = 1.07400000E+02;
EPS[8] = 9.75300000E+01;
EPS[9] = 1.07400000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetSIG(amrex::Real* SIG ) {
SIG[0] = 3.89240000E+00;
SIG[1] = 2.92000000E+00;
SIG[2] = 2.05000000E+00;
SIG[3] = 2.75000000E+00;
SIG[4] = 2.75000000E+00;
SIG[5] = 2.60000000E+00;
SIG[6] = 3.46000000E+00;
SIG[7] = 3.46000000E+00;
SIG[8] = 3.62000000E+00;
SIG[9] = 3.46000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetDIP(amrex::Real* DIP ) {
DIP[0] = 0.00000000E+00;
DIP[1] = 0.00000000E+00;
DIP[2] = 0.00000000E+00;
DIP[3] = 0.00000000E+00;
DIP[4] = 0.00000000E+00;
DIP[5] = 1.84000000E+00;
DIP[6] = 0.00000000E+00;
DIP[7] = 0.00000000E+00;
DIP[8] = 0.00000000E+00;
DIP[9] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetPOL(amrex::Real* POL ) {
POL[0] = 0.00000000E+00;
POL[1] = 7.90000000E-01;
POL[2] = 0.00000000E+00;
POL[3] = 0.00000000E+00;
POL[4] = 0.00000000E+00;
POL[5] = 0.00000000E+00;
POL[6] = 0.00000000E+00;
POL[7] = 0.00000000E+00;
POL[8] = 1.76000000E+00;
POL[9] = 1.60000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetZROT(amrex::Real* ZROT ) {
ZROT[0] = 0.00000000E+00;
ZROT[1] = 2.80000000E+02;
ZROT[2] = 0.00000000E+00;
ZROT[3] = 0.00000000E+00;
ZROT[4] = 0.00000000E+00;
ZROT[5] = 4.00000000E+00;
ZROT[6] = 1.00000000E+00;
ZROT[7] = 3.80000000E+00;
ZROT[8] = 4.00000000E+00;
ZROT[9] = 3.80000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetNLIN(int* NLIN) {
NLIN[0] = 0;
NLIN[1] = 1;
NLIN[2] = 0;
NLIN[3] = 1;
NLIN[4] = 0;
NLIN[5] = 2;
NLIN[6] = 2;
NLIN[7] = 2;
NLIN[8] = 1;
NLIN[9] = 1;
}


// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetCOFETA(amrex::Real* COFETA) {
COFETA[0] = -2.28538382E+01;
COFETA[1] = 4.72351261E+00;
COFETA[2] = -5.00398591E-01;
COFETA[3] = 2.04726730E-02;
COFETA[4] = -1.38787908E+01;
COFETA[5] = 1.01622169E+00;
COFETA[6] = -5.13793635E-02;
COFETA[7] = 2.35843970E-03;
COFETA[8] = -1.84708759E+01;
COFETA[9] = 2.79829658E+00;
COFETA[10] = -2.73167137E-01;
COFETA[11] = 1.15310988E-02;
COFETA[12] = -1.38534577E+01;
COFETA[13] = 1.39118532E+00;
COFETA[14] = -9.51599479E-02;
COFETA[15] = 4.02909298E-03;
COFETA[16] = -1.38840071E+01;
COFETA[17] = 1.39118532E+00;
COFETA[18] = -9.51599479E-02;
COFETA[19] = 4.02909298E-03;
COFETA[20] = -1.62641123E+01;
COFETA[21] = 1.14350363E+00;
COFETA[22] = 5.30639143E-02;
COFETA[23] = -6.33591924E-03;
COFETA[24] = -1.55676726E+01;
COFETA[25] = 1.98269739E+00;
COFETA[26] = -1.70640472E-01;
COFETA[27] = 7.23674769E-03;
COFETA[28] = -1.55526311E+01;
COFETA[29] = 1.98269739E+00;
COFETA[30] = -1.70640472E-01;
COFETA[31] = 7.23674769E-03;
COFETA[32] = -1.50490758E+01;
COFETA[33] = 1.71951444E+00;
COFETA[34] = -1.36565789E-01;
COFETA[35] = 5.76765721E-03;
COFETA[36] = -1.55831805E+01;
COFETA[37] = 1.98269739E+00;
COFETA[38] = -1.70640472E-01;
COFETA[39] = 7.23674769E-03;
}


// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetCOFLAM(amrex::Real* COFLAM) {
COFLAM[0] = -8.17342084E+00;
COFLAM[1] = 4.72351261E+00;
COFLAM[2] = -5.00398591E-01;
COFLAM[3] = 2.04726730E-02;
COFLAM[4] = 1.30620062E+01;
COFLAM[5] = -2.20342706E+00;
COFLAM[6] = 3.75973852E-01;
COFLAM[7] = -1.54999822E-02;
COFLAM[8] = 1.07900481E+00;
COFLAM[9] = 2.79829658E+00;
COFLAM[10] = -2.73167137E-01;
COFLAM[11] = 1.15310988E-02;
COFLAM[12] = 1.45176855E+01;
COFLAM[13] = -3.43279165E+00;
COFLAM[14] = 5.65624899E-01;
COFLAM[15] = -2.50398220E-02;
COFLAM[16] = 2.90131550E+00;
COFLAM[17] = 1.39118532E+00;
COFLAM[18] = -9.51599479E-02;
COFLAM[19] = 4.02909298E-03;
COFLAM[20] = 1.30806426E+01;
COFLAM[21] = -4.47738983E+00;
COFLAM[22] = 8.81014357E-01;
COFLAM[23] = -4.49595146E-02;
COFLAM[24] = 6.74947548E+00;
COFLAM[25] = -1.11855728E+00;
COFLAM[26] = 3.39360454E-01;
COFLAM[27] = -1.83842711E-02;
COFLAM[28] = -1.52814329E+00;
COFLAM[29] = 2.38008063E+00;
COFLAM[30] = -1.36271392E-01;
COFLAM[31] = 3.08782609E-03;
COFLAM[32] = 5.67331120E+00;
COFLAM[33] = -3.45183256E-01;
COFLAM[34] = 1.83466157E-01;
COFLAM[35] = -9.73309558E-03;
COFLAM[36] = -1.41355895E+00;
COFLAM[37] = 2.69340935E+00;
COFLAM[38] = -2.47127479E-01;
COFLAM[39] = 1.06111535E-02;
}


// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetCOFD(amrex::Real* COFD) {
COFD[0] = -2.15457505E+01;
COFD[1] = 5.11272080E+00;
COFD[2] = -4.19370338E-01;
COFD[3] = 1.69775218E-02;
COFD[4] = -1.30324172E+01;
COFD[5] = 2.85903920E+00;
COFD[6] = -1.51742677E-01;
COFD[7] = 6.35782269E-03;
COFD[8] = -1.66893843E+01;
COFD[9] = 4.37915633E+00;
COFD[10] = -3.36827940E-01;
COFD[11] = 1.38825024E-02;
COFD[12] = -1.62129756E+01;
COFD[13] = 3.66092001E+00;
COFD[14] = -2.51105729E-01;
COFD[15] = 1.04677344E-02;
COFD[16] = -1.61858364E+01;
COFD[17] = 3.66092001E+00;
COFD[18] = -2.51105729E-01;
COFD[19] = 1.04677344E-02;
COFD[20] = -2.16906441E+01;
COFD[21] = 5.38271594E+00;
COFD[22] = -4.34537288E-01;
COFD[23] = 1.69178174E-02;
COFD[24] = -1.77192800E+01;
COFD[25] = 4.02258943E+00;
COFD[26] = -2.94756302E-01;
COFD[27] = 1.22256407E-02;
COFD[28] = -1.77312632E+01;
COFD[29] = 4.02258943E+00;
COFD[30] = -2.94756302E-01;
COFD[31] = 1.22256407E-02;
COFD[32] = -1.73267098E+01;
COFD[33] = 3.89172363E+00;
COFD[34] = -2.78951049E-01;
COFD[35] = 1.15887531E-02;
COFD[36] = -1.77068490E+01;
COFD[37] = 4.02258943E+00;
COFD[38] = -2.94756302E-01;
COFD[39] = 1.22256407E-02;
COFD[40] = -1.30324172E+01;
COFD[41] = 2.85903920E+00;
COFD[42] = -1.51742677E-01;
COFD[43] = 6.35782269E-03;
COFD[44] = -9.80900467E+00;
COFD[45] = 1.96163221E+00;
COFD[46] = -4.12901694E-02;
COFD[47] = 1.84333974E-03;
COFD[48] = -1.08101189E+01;
COFD[49] = 2.50190243E+00;
COFD[50] = -1.09403379E-01;
COFD[51] = 4.69448836E-03;
COFD[52] = -1.05278454E+01;
COFD[53] = 2.11656257E+00;
COFD[54] = -5.96689375E-02;
COFD[55] = 2.55968757E-03;
COFD[56] = -1.05245180E+01;
COFD[57] = 2.11656257E+00;
COFD[58] = -5.96689375E-02;
COFD[59] = 2.55968757E-03;
COFD[60] = -1.56690835E+01;
COFD[61] = 3.96501125E+00;
COFD[62] = -2.87796763E-01;
COFD[63] = 1.19448912E-02;
COFD[64] = -1.14068845E+01;
COFD[65] = 2.35307161E+00;
COFD[66] = -9.09866633E-02;
COFD[67] = 3.93835178E-03;
COFD[68] = -1.14077381E+01;
COFD[69] = 2.35307161E+00;
COFD[70] = -9.09866633E-02;
COFD[71] = 3.93835178E-03;
COFD[72] = -1.12080264E+01;
COFD[73] = 2.25937836E+00;
COFD[74] = -7.86156855E-02;
COFD[75] = 3.39554799E-03;
COFD[76] = -1.14059786E+01;
COFD[77] = 2.35307161E+00;
COFD[78] = -9.09866633E-02;
COFD[79] = 3.93835178E-03;
COFD[80] = -1.66893843E+01;
COFD[81] = 4.37915633E+00;
COFD[82] = -3.36827940E-01;
COFD[83] = 1.38825024E-02;
COFD[84] = -1.08101189E+01;
COFD[85] = 2.50190243E+00;
COFD[86] = -1.09403379E-01;
COFD[87] = 4.69448836E-03;
COFD[88] = -1.33637042E+01;
COFD[89] = 3.59683604E+00;
COFD[90] = -2.43510741E-01;
COFD[91] = 1.01676642E-02;
COFD[92] = -1.22257788E+01;
COFD[93] = 2.95736606E+00;
COFD[94] = -1.64376124E-01;
COFD[95] = 6.89917971E-03;
COFD[96] = -1.22240192E+01;
COFD[97] = 2.95736606E+00;
COFD[98] = -1.64376124E-01;
COFD[99] = 6.89917971E-03;
COFD[100] = -1.85031870E+01;
COFD[101] = 5.14328386E+00;
COFD[102] = -4.22404408E-01;
COFD[103] = 1.70746798E-02;
COFD[104] = -1.34040286E+01;
COFD[105] = 3.28255086E+00;
COFD[106] = -2.05144984E-01;
COFD[107] = 8.60500584E-03;
COFD[108] = -1.34044679E+01;
COFD[109] = 3.28255086E+00;
COFD[110] = -2.05144984E-01;
COFD[111] = 8.60500584E-03;
COFD[112] = -1.31018058E+01;
COFD[113] = 3.15057727E+00;
COFD[114] = -1.88528038E-01;
COFD[115] = 7.90776051E-03;
COFD[116] = -1.34035620E+01;
COFD[117] = 3.28255086E+00;
COFD[118] = -2.05144984E-01;
COFD[119] = 8.60500584E-03;
COFD[120] = -1.62129756E+01;
COFD[121] = 3.66092001E+00;
COFD[122] = -2.51105729E-01;
COFD[123] = 1.04677344E-02;
COFD[124] = -1.05278454E+01;
COFD[125] = 2.11656257E+00;
COFD[126] = -5.96689375E-02;
COFD[127] = 2.55968757E-03;
COFD[128] = -1.22257788E+01;
COFD[129] = 2.95736606E+00;
COFD[130] = -1.64376124E-01;
COFD[131] = 6.89917971E-03;
COFD[132] = -1.22116801E+01;
COFD[133] = 2.45332065E+00;
COFD[134] = -1.00916129E-01;
COFD[135] = 4.23392204E-03;
COFD[136] = -1.21961721E+01;
COFD[137] = 2.45332065E+00;
COFD[138] = -1.00916129E-01;
COFD[139] = 4.23392204E-03;
COFD[140] = -1.79364167E+01;
COFD[141] = 4.50600723E+00;
COFD[142] = -3.51584883E-01;
COFD[143] = 1.44557185E-02;
COFD[144] = -1.31305689E+01;
COFD[145] = 2.64272371E+00;
COFD[146] = -1.24120540E-01;
COFD[147] = 5.18040209E-03;
COFD[148] = -1.31356332E+01;
COFD[149] = 2.64272371E+00;
COFD[150] = -1.24120540E-01;
COFD[151] = 5.18040209E-03;
COFD[152] = -1.29355792E+01;
COFD[153] = 2.56400897E+00;
COFD[154] = -1.14232442E-01;
COFD[155] = 4.76603158E-03;
COFD[156] = -1.31252412E+01;
COFD[157] = 2.64272371E+00;
COFD[158] = -1.24120540E-01;
COFD[159] = 5.18040209E-03;
COFD[160] = -1.61858364E+01;
COFD[161] = 3.66092001E+00;
COFD[162] = -2.51105729E-01;
COFD[163] = 1.04677344E-02;
COFD[164] = -1.05245180E+01;
COFD[165] = 2.11656257E+00;
COFD[166] = -5.96689375E-02;
COFD[167] = 2.55968757E-03;
COFD[168] = -1.22240192E+01;
COFD[169] = 2.95736606E+00;
COFD[170] = -1.64376124E-01;
COFD[171] = 6.89917971E-03;
COFD[172] = -1.21961721E+01;
COFD[173] = 2.45332065E+00;
COFD[174] = -1.00916129E-01;
COFD[175] = 4.23392204E-03;
COFD[176] = -1.21811307E+01;
COFD[177] = 2.45332065E+00;
COFD[178] = -1.00916129E-01;
COFD[179] = 4.23392204E-03;
COFD[180] = -1.79204695E+01;
COFD[181] = 4.50600723E+00;
COFD[182] = -3.51584883E-01;
COFD[183] = 1.44557185E-02;
COFD[184] = -1.31101998E+01;
COFD[185] = 2.64272371E+00;
COFD[186] = -1.24120540E-01;
COFD[187] = 5.18040209E-03;
COFD[188] = -1.31150609E+01;
COFD[189] = 2.64272371E+00;
COFD[190] = -1.24120540E-01;
COFD[191] = 5.18040209E-03;
COFD[192] = -1.29163518E+01;
COFD[193] = 2.56400897E+00;
COFD[194] = -1.14232442E-01;
COFD[195] = 4.76603158E-03;
COFD[196] = -1.31050838E+01;
COFD[197] = 2.64272371E+00;
COFD[198] = -1.24120540E-01;
COFD[199] = 5.18040209E-03;
COFD[200] = -2.16906441E+01;
COFD[201] = 5.38271594E+00;
COFD[202] = -4.34537288E-01;
COFD[203] = 1.69178174E-02;
COFD[204] = -1.56690835E+01;
COFD[205] = 3.96501125E+00;
COFD[206] = -2.87796763E-01;
COFD[207] = 1.19448912E-02;
COFD[208] = -1.85031870E+01;
COFD[209] = 5.14328386E+00;
COFD[210] = -4.22404408E-01;
COFD[211] = 1.70746798E-02;
COFD[212] = -1.79364167E+01;
COFD[213] = 4.50600723E+00;
COFD[214] = -3.51584883E-01;
COFD[215] = 1.44557185E-02;
COFD[216] = -1.79204695E+01;
COFD[217] = 4.50600723E+00;
COFD[218] = -3.51584883E-01;
COFD[219] = 1.44557185E-02;
COFD[220] = -1.69279399E+01;
COFD[221] = 3.14331335E+00;
COFD[222] = -8.14182387E-02;
COFD[223] = -3.35268067E-04;
COFD[224] = -1.93045117E+01;
COFD[225] = 4.82809810E+00;
COFD[226] = -3.87839926E-01;
COFD[227] = 1.58140074E-02;
COFD[228] = -1.93097711E+01;
COFD[229] = 4.82809810E+00;
COFD[230] = -3.87839926E-01;
COFD[231] = 1.58140074E-02;
COFD[232] = -1.97878851E+01;
COFD[233] = 4.97737361E+00;
COFD[234] = -4.04970580E-01;
COFD[235] = 1.64714079E-02;
COFD[236] = -2.00574073E+01;
COFD[237] = 5.07117863E+00;
COFD[238] = -4.15044627E-01;
COFD[239] = 1.68295105E-02;
COFD[240] = -1.77192800E+01;
COFD[241] = 4.02258943E+00;
COFD[242] = -2.94756302E-01;
COFD[243] = 1.22256407E-02;
COFD[244] = -1.14068845E+01;
COFD[245] = 2.35307161E+00;
COFD[246] = -9.09866633E-02;
COFD[247] = 3.93835178E-03;
COFD[248] = -1.34040286E+01;
COFD[249] = 3.28255086E+00;
COFD[250] = -2.05144984E-01;
COFD[251] = 8.60500584E-03;
COFD[252] = -1.31305689E+01;
COFD[253] = 2.64272371E+00;
COFD[254] = -1.24120540E-01;
COFD[255] = 5.18040209E-03;
COFD[256] = -1.31101998E+01;
COFD[257] = 2.64272371E+00;
COFD[258] = -1.24120540E-01;
COFD[259] = 5.18040209E-03;
COFD[260] = -1.93045117E+01;
COFD[261] = 4.82809810E+00;
COFD[262] = -3.87839926E-01;
COFD[263] = 1.58140074E-02;
COFD[264] = -1.43659711E+01;
COFD[265] = 2.95065736E+00;
COFD[266] = -1.63519956E-01;
COFD[267] = 6.86271545E-03;
COFD[268] = -1.43734352E+01;
COFD[269] = 2.95065736E+00;
COFD[270] = -1.63519956E-01;
COFD[271] = 6.86271545E-03;
COFD[272] = -1.40449548E+01;
COFD[273] = 2.82848714E+00;
COFD[274] = -1.47794217E-01;
COFD[275] = 6.18770175E-03;
COFD[276] = -1.43581569E+01;
COFD[277] = 2.95065736E+00;
COFD[278] = -1.63519956E-01;
COFD[279] = 6.86271545E-03;
COFD[280] = -1.77312632E+01;
COFD[281] = 4.02258943E+00;
COFD[282] = -2.94756302E-01;
COFD[283] = 1.22256407E-02;
COFD[284] = -1.14077381E+01;
COFD[285] = 2.35307161E+00;
COFD[286] = -9.09866633E-02;
COFD[287] = 3.93835178E-03;
COFD[288] = -1.34044679E+01;
COFD[289] = 3.28255086E+00;
COFD[290] = -2.05144984E-01;
COFD[291] = 8.60500584E-03;
COFD[292] = -1.31356332E+01;
COFD[293] = 2.64272371E+00;
COFD[294] = -1.24120540E-01;
COFD[295] = 5.18040209E-03;
COFD[296] = -1.31150609E+01;
COFD[297] = 2.64272371E+00;
COFD[298] = -1.24120540E-01;
COFD[299] = 5.18040209E-03;
COFD[300] = -1.93097711E+01;
COFD[301] = 4.82809810E+00;
COFD[302] = -3.87839926E-01;
COFD[303] = 1.58140074E-02;
COFD[304] = -1.43734352E+01;
COFD[305] = 2.95065736E+00;
COFD[306] = -1.63519956E-01;
COFD[307] = 6.86271545E-03;
COFD[308] = -1.43810125E+01;
COFD[309] = 2.95065736E+00;
COFD[310] = -1.63519956E-01;
COFD[311] = 6.86271545E-03;
COFD[312] = -1.40518042E+01;
COFD[313] = 2.82848714E+00;
COFD[314] = -1.47794217E-01;
COFD[315] = 6.18770175E-03;
COFD[316] = -1.43655045E+01;
COFD[317] = 2.95065736E+00;
COFD[318] = -1.63519956E-01;
COFD[319] = 6.86271545E-03;
COFD[320] = -1.73267098E+01;
COFD[321] = 3.89172363E+00;
COFD[322] = -2.78951049E-01;
COFD[323] = 1.15887531E-02;
COFD[324] = -1.12080264E+01;
COFD[325] = 2.25937836E+00;
COFD[326] = -7.86156855E-02;
COFD[327] = 3.39554799E-03;
COFD[328] = -1.31018058E+01;
COFD[329] = 3.15057727E+00;
COFD[330] = -1.88528038E-01;
COFD[331] = 7.90776051E-03;
COFD[332] = -1.29355792E+01;
COFD[333] = 2.56400897E+00;
COFD[334] = -1.14232442E-01;
COFD[335] = 4.76603158E-03;
COFD[336] = -1.29163518E+01;
COFD[337] = 2.56400897E+00;
COFD[338] = -1.14232442E-01;
COFD[339] = 4.76603158E-03;
COFD[340] = -1.97878851E+01;
COFD[341] = 4.97737361E+00;
COFD[342] = -4.04970580E-01;
COFD[343] = 1.64714079E-02;
COFD[344] = -1.40449548E+01;
COFD[345] = 2.82848714E+00;
COFD[346] = -1.47794217E-01;
COFD[347] = 6.18770175E-03;
COFD[348] = -1.40518042E+01;
COFD[349] = 2.82848714E+00;
COFD[350] = -1.47794217E-01;
COFD[351] = 6.18770175E-03;
COFD[352] = -1.37735495E+01;
COFD[353] = 2.72491264E+00;
COFD[354] = -1.34500038E-01;
COFD[355] = 5.61833600E-03;
COFD[356] = -1.40377754E+01;
COFD[357] = 2.82848714E+00;
COFD[358] = -1.47794217E-01;
COFD[359] = 6.18770175E-03;
COFD[360] = -1.77068490E+01;
COFD[361] = 4.02258943E+00;
COFD[362] = -2.94756302E-01;
COFD[363] = 1.22256407E-02;
COFD[364] = -1.14059786E+01;
COFD[365] = 2.35307161E+00;
COFD[366] = -9.09866633E-02;
COFD[367] = 3.93835178E-03;
COFD[368] = -1.34035620E+01;
COFD[369] = 3.28255086E+00;
COFD[370] = -2.05144984E-01;
COFD[371] = 8.60500584E-03;
COFD[372] = -1.31252412E+01;
COFD[373] = 2.64272371E+00;
COFD[374] = -1.24120540E-01;
COFD[375] = 5.18040209E-03;
COFD[376] = -1.31050838E+01;
COFD[377] = 2.64272371E+00;
COFD[378] = -1.24120540E-01;
COFD[379] = 5.18040209E-03;
COFD[380] = -2.00574073E+01;
COFD[381] = 5.07117863E+00;
COFD[382] = -4.15044627E-01;
COFD[383] = 1.68295105E-02;
COFD[384] = -1.43581569E+01;
COFD[385] = 2.95065736E+00;
COFD[386] = -1.63519956E-01;
COFD[387] = 6.86271545E-03;
COFD[388] = -1.43655045E+01;
COFD[389] = 2.95065736E+00;
COFD[390] = -1.63519956E-01;
COFD[391] = 6.86271545E-03;
COFD[392] = -1.40377754E+01;
COFD[393] = 2.82848714E+00;
COFD[394] = -1.47794217E-01;
COFD[395] = 6.18770175E-03;
COFD[396] = -1.43504631E+01;
COFD[397] = 2.95065736E+00;
COFD[398] = -1.63519956E-01;
COFD[399] = 6.86271545E-03;
}


// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetKTDIF(int* KTDIF) {
KTDIF[0] = 1;
KTDIF[1] = 2;
}


// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void egtransetCOFTD(amrex::Real* COFTD) {
COFTD[0] = 4.14467782E-01;
COFTD[1] = 1.24238458E-04;
COFTD[2] = -3.83580393E-08;
COFTD[3] = 3.70639619E-12;
COFTD[4] = 0.00000000E+00;
COFTD[5] = 0.00000000E+00;
COFTD[6] = 0.00000000E+00;
COFTD[7] = 0.00000000E+00;
COFTD[8] = -1.69911728E-01;
COFTD[9] = -9.81529303E-06;
COFTD[10] = 1.59489253E-09;
COFTD[11] = -1.03913573E-13;
COFTD[12] = 4.26444080E-01;
COFTD[13] = -6.86111968E-06;
COFTD[14] = 9.70585208E-09;
COFTD[15] = -1.54405185E-12;
COFTD[16] = 4.20026397E-01;
COFTD[17] = -6.75786466E-06;
COFTD[18] = 9.55978584E-09;
COFTD[19] = -1.52081496E-12;
COFTD[20] = 1.83353806E-01;
COFTD[21] = 2.64014873E-04;
COFTD[22] = -7.94512706E-08;
COFTD[23] = 7.22879171E-12;
COFTD[24] = 4.69513337E-01;
COFTD[25] = 2.97373976E-06;
COFTD[26] = 5.31922442E-09;
COFTD[27] = -8.69704341E-13;
COFTD[28] = 4.71222342E-01;
COFTD[29] = 2.98456402E-06;
COFTD[30] = 5.33858612E-09;
COFTD[31] = -8.72870020E-13;
COFTD[32] = 4.63271137E-01;
COFTD[33] = -1.84241657E-06;
COFTD[34] = 7.43538276E-09;
COFTD[35] = -1.15907726E-12;
COFTD[36] = 4.67703041E-01;
COFTD[37] = 2.96227395E-06;
COFTD[38] = 5.29871515E-09;
COFTD[39] = -8.66351034E-13;
COFTD[40] = 1.47881891E-01;
COFTD[41] = 3.92425588E-04;
COFTD[42] = -1.14996996E-07;
COFTD[43] = 1.02430591E-11;
COFTD[44] = 1.69911728E-01;
COFTD[45] = 9.81529303E-06;
COFTD[46] = -1.59489253E-09;
COFTD[47] = 1.03913573E-13;
COFTD[48] = 0.00000000E+00;
COFTD[49] = 0.00000000E+00;
COFTD[50] = 0.00000000E+00;
COFTD[51] = 0.00000000E+00;
COFTD[52] = 3.68276396E-01;
COFTD[53] = 1.26592880E-04;
COFTD[54] = -3.92919185E-08;
COFTD[55] = 3.78344330E-12;
COFTD[56] = 3.65525942E-01;
COFTD[57] = 1.25647427E-04;
COFTD[58] = -3.89984687E-08;
COFTD[59] = 3.75518684E-12;
COFTD[60] = -9.00451878E-03;
COFTD[61] = 4.52856794E-04;
COFTD[62] = -1.24283910E-07;
COFTD[63] = 1.05887472E-11;
COFTD[64] = 3.40387913E-01;
COFTD[65] = 1.88498216E-04;
COFTD[66] = -5.86395419E-08;
COFTD[67] = 5.55120679E-12;
COFTD[68] = 3.41005164E-01;
COFTD[69] = 1.88840034E-04;
COFTD[70] = -5.87458773E-08;
COFTD[71] = 5.56127320E-12;
COFTD[72] = 3.53561056E-01;
COFTD[73] = 1.68362559E-04;
COFTD[74] = -5.24529746E-08;
COFTD[75] = 4.99520203E-12;
COFTD[76] = 3.39732961E-01;
COFTD[77] = 1.88135520E-04;
COFTD[78] = -5.85267116E-08;
COFTD[79] = 5.54052552E-12;
}


// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void GET_CRITPARAMS(amrex::Real *  Tci, amrex::Real *  ai, amrex::Real *  bi, amrex::Real *  acentric_i)
{

amrex::Real   EPS[10];
amrex::Real   SIG[10];
amrex::Real    wt[10];
amrex::Real Rcst = 83.144598;// in bar [CGS] !
amrex::Real avogadro = 6.02214199e23;
amrex::Real boltzmann = 1.3806503e-16;// we work in CGS

egtransetEPS(EPS);
egtransetSIG(SIG);
get_mw(wt);

// species 0: XE
Tci[0] = 1.316 * EPS[0] ; 
ai[0] = (5.55 * avogadro * avogadro * EPS[0]*boltzmann * 1e-24 * SIG[0] * SIG[0] * SIG[0] ) / (wt[0] * wt[0]); 
bi[0] = 0.855 * avogadro * 1e-24 * SIG[0] * SIG[0] * SIG[0] / (wt[0]); 
acentric_i[0] = 0.0 ;

// species 1: H2
// Imported from NIST
Tci[1] = 33.145000 ; 
ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] / (2.015880 * 2.015880 * 12.964000); 
bi[1] = 0.08664 * Rcst * Tci[1] / (2.015880 * 12.964000); 
acentric_i[1] = -0.219000 ;

// species 2: H
Tci[2] = 1.316 * EPS[2] ; 
ai[2] = (5.55 * avogadro * avogadro * EPS[2]*boltzmann * 1e-24 * SIG[2] * SIG[2] * SIG[2] ) / (wt[2] * wt[2]); 
bi[2] = 0.855 * avogadro * 1e-24 * SIG[2] * SIG[2] * SIG[2] / (wt[2]); 
acentric_i[2] = 0.0 ;

// species 3: OH
Tci[3] = 1.316 * EPS[3] ; 
ai[3] = (5.55 * avogadro * avogadro * EPS[3]*boltzmann * 1e-24 * SIG[3] * SIG[3] * SIG[3] ) / (wt[3] * wt[3]); 
bi[3] = 0.855 * avogadro * 1e-24 * SIG[3] * SIG[3] * SIG[3] / (wt[3]); 
acentric_i[3] = 0.0 ;

// species 4: O
Tci[4] = 1.316 * EPS[4] ; 
ai[4] = (5.55 * avogadro * avogadro * EPS[4]*boltzmann * 1e-24 * SIG[4] * SIG[4] * SIG[4] ) / (wt[4] * wt[4]); 
bi[4] = 0.855 * avogadro * 1e-24 * SIG[4] * SIG[4] * SIG[4] / (wt[4]); 
acentric_i[4] = 0.0 ;

// species 5: H2O
// Imported from NIST
Tci[5] = 647.096000 ; 
ai[5] = 1e6 * 0.42748 * Rcst * Rcst * Tci[5] * Tci[5] / (18.015340 * 18.015340 * 220.640000); 
bi[5] = 0.08664 * Rcst * Tci[5] / (18.015340 * 220.640000); 
acentric_i[5] = 0.344300 ;

// species 6: HO2
Tci[6] = 1.316 * EPS[6] ; 
ai[6] = (5.55 * avogadro * avogadro * EPS[6]*boltzmann * 1e-24 * SIG[6] * SIG[6] * SIG[6] ) / (wt[6] * wt[6]); 
bi[6] = 0.855 * avogadro * 1e-24 * SIG[6] * SIG[6] * SIG[6] / (wt[6]); 
acentric_i[6] = 0.0 ;

// species 7: H2O2
Tci[7] = 1.316 * EPS[7] ; 
ai[7] = (5.55 * avogadro * avogadro * EPS[7]*boltzmann * 1e-24 * SIG[7] * SIG[7] * SIG[7] ) / (wt[7] * wt[7]); 
bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]); 
acentric_i[7] = 0.0 ;

// species 8: N2
// Imported from NIST
Tci[8] = 126.192000 ; 
ai[8] = 1e6 * 0.42748 * Rcst * Rcst * Tci[8] * Tci[8] / (28.013400 * 28.013400 * 33.958000); 
bi[8] = 0.08664 * Rcst * Tci[8] / (28.013400 * 33.958000); 
acentric_i[8] = 0.037200 ;

// species 9: O2
// Imported from NIST
Tci[9] = 154.581000 ; 
ai[9] = 1e6 * 0.42748 * Rcst * Rcst * Tci[9] * Tci[9] / (31.998800 * 31.998800 * 50.430466); 
bi[9] = 0.08664 * Rcst * Tci[9] / (31.998800 * 50.430466); 
acentric_i[9] = 0.022200 ;

}


// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void GET_CRITPARAMS_SRK(amrex::Real *  sqrtOneOverTc, amrex::Real *  sqrtAsti, amrex::Real *  Bi, amrex::Real *  Fomega)
{

// species 0: XE
// Computed from Lennard-Jones
sqrtOneOverTc[0] = 5.1882907821667e-02;
sqrtAsti[0] = 1.6382180217541e+04;
Bi[0] = 2.3127459310562e-01;
Fomega[0] = 4.8508000000000e-01;

// species 1: H2
// Imported from NIST
sqrtOneOverTc[1] = 1.7369646834812e-01;
sqrtAsti[1] = 2.4824229305911e+05;
Bi[1] = 9.1362254169923e+00;
Fomega[1] = 1.3798618890700e-01;

// species 2: H
// Computed from Lennard-Jones
sqrtOneOverTc[2] = 7.2391579785311e-02;
sqrtAsti[2] = 5.8451221185793e+05;
Bi[2] = 4.4006634563410e+00;
Fomega[2] = 4.8508000000000e-01;

// species 3: OH
// Computed from Lennard-Jones
sqrtOneOverTc[3] = 9.7460146948181e-02;
sqrtAsti[3] = 3.9981183985545e+04;
Bi[3] = 6.2963299076575e-01;
Fomega[3] = 4.8508000000000e-01;

// species 4: O
// Computed from Lennard-Jones
sqrtOneOverTc[4] = 9.7460146948181e-02;
sqrtAsti[4] = 4.2500156012386e+04;
Bi[4] = 6.6930234851885e-01;
Fomega[4] = 4.8508000000000e-01;

// species 5: H2O
// Imported from NIST
sqrtOneOverTc[5] = 3.9311140369715e-02;
sqrtAsti[5] = 1.3145468419592e+05;
Bi[5] = 1.1727204136223e+00;
Fomega[5] = 1.0013577274636e+00;

// species 6: HO2
// Computed from Lennard-Jones
sqrtOneOverTc[6] = 8.4114315417892e-02;
sqrtAsti[6] = 3.3687029897130e+04;
Bi[6] = 6.4617850449304e-01;
Fomega[6] = 4.8508000000000e-01;

// species 7: H2O2
// Computed from Lennard-Jones
sqrtOneOverTc[7] = 8.4114315417892e-02;
sqrtAsti[7] = 3.2688719609122e+04;
Bi[7] = 6.2702909740981e-01;
Fomega[7] = 4.8508000000000e-01;

// species 8: N2
// Imported from NIST
sqrtOneOverTc[8] = 8.9019282240563e-02;
sqrtAsti[8] = 4.2022983929562e+04;
Bi[8] = 9.5560052221563e-01;
Fomega[8] = 5.4259343186608e-01;

// species 9: O2
// Imported from NIST
sqrtOneOverTc[9] = 8.0430717653241e-02;
sqrtAsti[9] = 3.6980081924740e+04;
Bi[9] = 6.9005216444999e-01;
Fomega[9] = 5.1945301904908e-01;

}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/) {
amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/) {
amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
