#include "mechanism.H"
const int rmap[21] = {8,15,4,5,6,7,0,1,2,3,9,10,11,12,13,14,16,17,18,19,20};

// Returns 0-based map of reaction order
void GET_RMAP
(int * _rmap)
{
for (int j=0; j<21; ++j)
{
_rmap[j] = rmap[j];
}
}

// Returns a count of species in a reaction, and their indices
// and stoichiometric coefficients. (Eq 50)
void CKINU(const int i, int& nspec, int ki[], int nu[])
{
const int ns[21] =
     {4,4,4,3,2,2,3,3,3,4,3,4,4,3,3,2,4,4,4,4,4};
const int kiv[84] =
     {2,9,4,3,1,4,2,3,1,3,2,5,5,4,3,0,1,2,0,0,4,9,0,0,2,4,3,0,2,3,5,0,2,9,6,0,2,6,1,9,2,6,3,0,6,4,9,3,6,3,5,9,6,7,9,0,6,7,9,0,7,3,0,0,2,7,5,3,2,7,1,6,7,4,6,3,7,3,5,6,7,3,5,6};
const int nuv[84] =
     {-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,2,0,-1,2,0,0,-2,1,0,0,-1,-1,1,0,-1,-1,1,0,-1,-1,1,0,-1,-1,1,1,-1,-1,2,0,-1,-1,1,1,-1,-1,1,1,-2,1,1,0,-2,1,1,0,-1,2,0,0,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1};
if (i < 1) {
// Return max num species per reaction
nspec = 4;
} else {
if (i > 21) {
nspec = -1;
} else {
nspec = ns[i-1];
for (int j=0; j<nspec; ++j) {
ki[j] = kiv[(i-1)*4 + j] + 1;
nu[j] = nuv[(i-1)*4 + j];
}
}
}
}

// Returns the progress rates of each reactions
// Given P, T, and mole fractions
void CKKFKR(const amrex::Real P, const amrex::Real T, const amrex::Real x[], amrex::Real q_f[], amrex::Real q_r[])
{
amrex::Real c[10]; // temporary storage
amrex::Real PORT = 1e6 * P/(8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

// Compute conversion, see Eq 10
for (int id = 0; id < 10; ++id) {
c[id] = x[id]*PORT;
}

// convert to chemkin units
progressRateFR(q_f, q_r, c, T);

// convert to chemkin units
for (int id = 0; id < 21; ++id) {
q_f[id] *= 1.0e-6;
q_r[id] *= 1.0e-6;
}
}

// compute the progress rate for each reaction
// USES progressRate : todo switch to GPU
void progressRateFR(amrex::Real *  q_f, amrex::Real *  q_r, amrex::Real *  sc, amrex::Real T)
{
const amrex::Real invT = 1.0 / T;
const amrex::Real logT = log(T);
// compute the Gibbs free energy
amrex::Real g_RT[10];
gibbs(g_RT, T);

amrex::Real sc_qss[1];
comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);

}

// save atomic weights into array
void atomicWeight(amrex::Real *  awt)
{
awt[0] = 15.999000; // O
awt[1] = 1.008000; // H
awt[2] = 14.007000; // N
awt[3] = 131.293000; // Xe
}

// get atomic weight for all elements
void CKAWT( amrex::Real *  awt)
{
atomicWeight(awt);
}

// Returns the elemental composition 
// of the speciesi (mdim is num of elements)
void CKNCF(int * ncf)
{
int kd = 4; 
// Zero ncf
for (int id = 0; id < kd * 10; ++ id) {
 ncf[id] = 0; 
}

// XE
ncf[ 0 * kd + 3 ] = 1; // Xe

// H2
ncf[ 1 * kd + 1 ] = 2; // H

// H
ncf[ 2 * kd + 1 ] = 1; // H

// OH
ncf[ 3 * kd + 1 ] = 1; // H
ncf[ 3 * kd + 0 ] = 1; // O

// O
ncf[ 4 * kd + 0 ] = 1; // O

// H2O
ncf[ 5 * kd + 1 ] = 2; // H
ncf[ 5 * kd + 0 ] = 1; // O

// HO2
ncf[ 6 * kd + 1 ] = 1; // H
ncf[ 6 * kd + 0 ] = 2; // O

// H2O2
ncf[ 7 * kd + 1 ] = 2; // H
ncf[ 7 * kd + 0 ] = 2; // O

// N2
ncf[ 8 * kd + 2 ] = 2; // N

// O2
ncf[ 9 * kd + 0 ] = 2; // O

}

// Returns the vector of strings of element names
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
ename.resize(4);
ename[0] = "O";
ename[1] = "H";
ename[2] = "N";
ename[3] = "Xe";
}

// Returns the vector of strings of species names
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
kname.resize(10);
kname[0] = "XE";
kname[1] = "H2";
kname[2] = "H";
kname[3] = "OH";
kname[4] = "O";
kname[5] = "H2O";
kname[6] = "HO2";
kname[7] = "H2O2";
kname[8] = "N2";
kname[9] = "O2";
}

// compute the sparsity pattern of the chemistry Jacobian
void SPARSITY_INFO( int * nJdata, const int * consP, int NCELLS)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

int nJdata_tmp = 0;
for (int k=0; k<11; k++) {
for (int l=0; l<11; l++) {
if(Jac[ 11 * k + l] != 0.0){
nJdata_tmp = nJdata_tmp + 1;
}
}
}

*nJdata = NCELLS * nJdata_tmp;
}



// compute the sparsity pattern of the system Jacobian
void SPARSITY_INFO_SYST( int * nJdata, const int * consP, int NCELLS)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

int nJdata_tmp = 0;
for (int k=0; k<11; k++) {
for (int l=0; l<11; l++) {
if(k == l){
nJdata_tmp = nJdata_tmp + 1;
} else {
if(Jac[ 11 * k + l] != 0.0){
nJdata_tmp = nJdata_tmp + 1;
}
}
}
}

*nJdata = NCELLS * nJdata_tmp;
}



// compute the sparsity pattern of the simplified (for preconditioning) system Jacobian
void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, const int * consP)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

int nJdata_tmp = 0;
for (int k=0; k<11; k++) {
for (int l=0; l<11; l++) {
if(k == l){
nJdata_tmp = nJdata_tmp + 1;
} else {
if(Jac[ 11 * k + l] != 0.0){
nJdata_tmp = nJdata_tmp + 1;
}
}
}
}

nJdata[0] = nJdata_tmp;
}


// compute the sparsity pattern of the chemistry Jacobian in CSC format -- base 0
void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, const int * consP, int NCELLS)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

colPtrs[0] = 0;
int nJdata_tmp = 0;
for (int nc=0; nc<NCELLS; nc++) {
int offset_row = nc * 11;
int offset_col = nc * 11;
for (int k=0; k<11; k++) {
for (int l=0; l<11; l++) {
if(Jac[11*k + l] != 0.0) {
rowVals[nJdata_tmp] = l + offset_row; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
colPtrs[offset_col + (k + 1)] = nJdata_tmp;
}
}
}

// compute the sparsity pattern of the chemistry Jacobian in CSR format -- base 0
void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, const int * consP, int NCELLS, int base)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

if (base == 1) {
rowPtrs[0] = 1;
int nJdata_tmp = 1;
for (int nc=0; nc<NCELLS; nc++) {
int offset = nc * 11;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp-1] = k+1 + offset; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
rowPtrs[offset + (l + 1)] = nJdata_tmp;
}
}
} else {
rowPtrs[0] = 0;
int nJdata_tmp = 0;
for (int nc=0; nc<NCELLS; nc++) {
int offset = nc * 11;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp] = k + offset; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
rowPtrs[offset + (l + 1)] = nJdata_tmp;
}
}
}
}

// compute the sparsity pattern of the system Jacobian
// CSR format BASE is user choice
void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, const int * consP, int NCELLS, int base)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

if (base == 1) {
rowPtr[0] = 1;
int nJdata_tmp = 1;
for (int nc=0; nc<NCELLS; nc++) {
int offset = nc * 11;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if (k == l) {
colVals[nJdata_tmp-1] = l+1 + offset; 
nJdata_tmp = nJdata_tmp + 1; 
} else {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp-1] = k+1 + offset; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
}
rowPtr[offset + (l + 1)] = nJdata_tmp;
}
}
} else {
rowPtr[0] = 0;
int nJdata_tmp = 0;
for (int nc=0; nc<NCELLS; nc++) {
int offset = nc * 11;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if (k == l) {
colVals[nJdata_tmp] = l + offset; 
nJdata_tmp = nJdata_tmp + 1; 
} else {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp] = k + offset; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
}
rowPtr[offset + (l + 1)] = nJdata_tmp;
}
}
}
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian on CPU
// BASE 0
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, const int * consP)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

colPtrs[0] = 0;
int nJdata_tmp = 0;
for (int k=0; k<11; k++) {
for (int l=0; l<11; l++) {
if (k == l) {
rowVals[nJdata_tmp] = l; 
indx[nJdata_tmp] = 11*k + l;
nJdata_tmp = nJdata_tmp + 1; 
} else {
if(Jac[11*k + l] != 0.0) {
rowVals[nJdata_tmp] = l; 
indx[nJdata_tmp] = 11*k + l;
nJdata_tmp = nJdata_tmp + 1; 
}
}
}
colPtrs[k+1] = nJdata_tmp;
}
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian
// CSR format BASE is under choice
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, const int * consP, int base)
{
amrex::GpuArray<amrex::Real,121> Jac = {0.0};
amrex::GpuArray<amrex::Real,10> conc = {0.0};
for (int n=0; n<10; n++) {
    conc[n] = 1.0/ 10.000000 ;
}
aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

if (base == 1) {
rowPtr[0] = 1;
int nJdata_tmp = 1;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if (k == l) {
colVals[nJdata_tmp-1] = l+1; 
nJdata_tmp = nJdata_tmp + 1; 
} else {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp-1] = k+1; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
}
rowPtr[l+1] = nJdata_tmp;
}
} else {
rowPtr[0] = 0;
int nJdata_tmp = 0;
for (int l=0; l<11; l++) {
for (int k=0; k<11; k++) {
if (k == l) {
colVals[nJdata_tmp] = l; 
nJdata_tmp = nJdata_tmp + 1; 
} else {
if(Jac[11*k + l] != 0.0) {
colVals[nJdata_tmp] = k; 
nJdata_tmp = nJdata_tmp + 1; 
}
}
}
rowPtr[l+1] = nJdata_tmp;
}
}
}
