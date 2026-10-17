#include "../../../Sources/VectorCoreC/include/vc_lapack.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
int main(void) {
    if (!vc_lapack_available()) { puts("L01_LINKED_UNAVAILABLE"); return 0; }
    assert(vc_lapack_sgeqrf(-1,0,0,1,0)==-1);
    assert(vc_lapack_sgeqrf(0,-1,0,1,0)==-2);
    assert(vc_lapack_sgeqrf(0,0,0,0,0)==-4);
    assert(vc_lapack_sgeqrf(0,3,0,1,0)==0);
    assert(vc_lapack_sorgqr(-1,0,0,0,1,0)==-1);
    assert(vc_lapack_sorgqr(0,1,0,0,1,0)==-2);
    assert(vc_lapack_sorgqr(1,1,2,0,1,0)==-3);
    assert(vc_lapack_sorgqr(0,0,0,0,0,0)==-5);
    assert(vc_lapack_sorgqr(0,0,0,0,1,0)==0);
    assert(vc_lapack_sgesdd(-1,0,0,1,0,0,1,0,1)==-2);
    assert(vc_lapack_sgesdd(0,-1,0,1,0,0,1,0,1)==-3);
    assert(vc_lapack_sgesdd(0,0,0,0,0,0,1,0,1)==-5);
    assert(vc_lapack_sgesdd(0,0,0,1,0,0,0,0,1)==-8);
    assert(vc_lapack_sgesdd(0,0,0,1,0,0,1,0,0)==-10);
    assert(vc_lapack_sgesdd(0,0,0,1,0,0,1,0,1)==0);
    assert(vc_lapack_ssyevd('X','L',0,0,1,0)==-1);
    assert(vc_lapack_ssyevd('N','X',0,0,1,0)==-2);
    assert(vc_lapack_ssyevd('N','L',-1,0,1,0)==-3);
    assert(vc_lapack_ssyevd('N','L',0,0,0,0)==-5);
    assert(vc_lapack_ssyevd('N','L',0,0,1,0)==0);
    // Rejected locally before vendor query; NULL must never be accessed.
    assert(vc_lapack_sgesdd(30000,30000,0,30000,0,0,30000,0,30000)==VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_lapack_ssyevd('V','L',40000,0,40000,0)==VC_LAPACK_WORKSPACE_OVERFLOW);
    float a[2]={3,4}, tau[1];
    assert(vc_lapack_sgeqrf(2,1,a,2,tau)==0);
    assert(isfinite(a[0]) && fabsf(fabsf(a[0])-5)<1e-5f);
    assert(vc_lapack_sorgqr(2,1,1,a,2,tau)==0);
    assert(isfinite(a[0]) && isfinite(a[1]) && fabsf(a[0]*a[0]+a[1]*a[1]-1)<1e-5f);
    float b[4]={1,0,1,1}, u[4], vt[4], s[2];
    assert(vc_lapack_sgesdd(2,2,b,2,s,u,2,vt,2)==0);
    assert(isfinite(s[0]) && fabsf(s[0]-(1+sqrtf(5))/2)<1e-5f);
    assert(isfinite(s[1]) && fabsf(s[1]-(sqrtf(5)-1)/2)<1e-5f);
    for(int vectors=0;vectors<2;++vectors) {
        float c[4]={2,1,NAN,2}, e[2];
        assert(vc_lapack_ssyevd(vectors?'V':'N','L',2,c,2,e)==0);
        assert(isfinite(e[0]) && isfinite(e[1]) && fabsf(e[0]-1)<1e-5f && fabsf(e[1]-3)<1e-5f);
    }
    puts("L01_LINKED_COMPLETE");
}
