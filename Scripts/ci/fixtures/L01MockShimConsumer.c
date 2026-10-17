// Compile the actual shim with controlled vendor query outputs and real allocator
// calls. This proves local preflight/cleanup behavior, not Accelerate arithmetic.
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
typedef int32_t __LAPACK_int;
static int calls, allocations, frees, fail_at;
static float query = 128;
static int32_t integer_query = 32, last_lwork, last_liwork;
static void *tracked_malloc(size_t n) { ++allocations; return allocations == fail_at ? NULL : malloc(n); }
static void tracked_free(void *p) { if(p) ++frees; free(p); }
static void sgeqrf_(const int32_t *m,const int32_t *n,float *a,const int32_t *lda,float *tau,float *work,const int32_t *lwork,int32_t *info) {
    (void)m;(void)n;(void)a;(void)lda;(void)tau; ++calls; *info=0; if(*lwork==-1) *work=query; else last_lwork=*lwork;
}
static void sorgqr_(const int32_t *m,const int32_t *n,const int32_t *k,float *a,const int32_t *lda,const float *tau,float *work,const int32_t *lwork,int32_t *info) {
    (void)k; sgeqrf_(m,n,a,lda,(float*)tau,work,lwork,info);
}
static void sgesdd_(const char *job,const int32_t *m,const int32_t *n,float *a,const int32_t *lda,float *s,float *u,const int32_t *ldu,float *vt,const int32_t *ldvt,float *work,const int32_t *lwork,int32_t *iw,int32_t *info) {
    (void)job;(void)s;(void)u;(void)ldu;(void)vt;(void)ldvt;(void)iw; sgeqrf_(m,n,a,lda,NULL,work,lwork,info);
}
static void ssyevd_(const char *job,const char *uplo,const int32_t *n,float *a,const int32_t *lda,float *w,float *work,const int32_t *lwork,int32_t *iw,const int32_t *liwork,int32_t *info) {
    (void)job;(void)uplo;(void)w; sgeqrf_(n,n,a,lda,NULL,work,lwork,info); if(*lwork==-1) *iw=integer_query; else last_liwork=*liwork;
}
#undef __APPLE__
#define VC_HAS_LAPACK 1
#define malloc tracked_malloc
#define free tracked_free
#include "../../../Sources/VectorCoreC/common/vc_lapack.c"
#undef malloc
#undef free
static void reset(void) { calls=allocations=frees=fail_at=0;query=128;integer_query=32;last_lwork=last_liwork=0; }
int main(void) {
    reset(); assert(vc_lapack_sgesdd(30000,30000,NULL,30000,NULL,NULL,30000,NULL,30000)==VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(calls==0 && allocations==0);
    reset(); assert(vc_lapack_ssyevd('V','L',40000,NULL,40000,NULL)==VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(calls==0 && allocations==0);
    reset(); assert(vc_lapack_sgeqrf(-1,1,NULL,1,NULL)==-1); assert(calls==0 && allocations==0);
    reset(); assert(vc_lapack_sorgqr(0,0,0,NULL,1,NULL)==0); assert(calls==0 && allocations==0);
    for(int routine=0;routine<4;++routine) {
        for(int bad=0;bad<4;++bad) {
            reset(); query=(float[]){NAN,INFINITY,-1,0}[bad];
            int32_t result;
            if(routine==0) result=vc_lapack_sgeqrf(2,2,NULL,2,NULL);
            else if(routine==1) result=vc_lapack_sorgqr(2,2,2,NULL,2,NULL);
            else if(routine==2) result=vc_lapack_sgesdd(2,2,NULL,2,NULL,NULL,2,NULL,2);
            else result=vc_lapack_ssyevd('V','L',2,NULL,2,NULL);
            assert(result==VC_LAPACK_INVALID_QUERY && calls==1 && frees==allocations);
        }
    }
    // T19: a positive answer below each caller's reference minimum is raised to it and passed as LWORK/LIWORK.
    for(int routine=0;routine<4;++routine) {
        reset(); query=1; integer_query=1;
        int32_t result, expected=routine<2 ? 2 : routine==2 ? 30 : 21;
        if(routine==0) result=vc_lapack_sgeqrf(2,2,NULL,2,NULL);
        else if(routine==1) result=vc_lapack_sorgqr(2,2,2,NULL,2,NULL);
        else if(routine==2) result=vc_lapack_sgesdd(2,2,NULL,2,NULL,NULL,2,NULL,2);
        else result=vc_lapack_ssyevd('V','L',2,NULL,2,NULL);
        assert(result==0 && calls==2 && frees==allocations && last_lwork==expected);
        assert(routine<3 || last_liwork==13);
    }
    reset();integer_query=0;assert(vc_lapack_ssyevd('V','L',2,NULL,2,NULL)==VC_LAPACK_INVALID_QUERY);
    assert(calls==1 && allocations==0);
    reset();fail_at=2;assert(vc_lapack_sgesdd(2,2,NULL,2,NULL,NULL,2,NULL,2)==VC_LAPACK_UNAVAILABLE);
    assert(calls==1 && allocations==2 && frees==1);
    reset();fail_at=2;assert(vc_lapack_ssyevd('V','L',2,NULL,2,NULL)==VC_LAPACK_UNAVAILABLE);
    assert(calls==1 && allocations==2 && frees==1);
    reset();assert(vc_lapack_sgesdd(2,2,NULL,2,NULL,NULL,2,NULL,2)==0);
    assert(calls==2 && allocations==2 && frees==2);
    puts("L01_MOCK_SHIM_COMPLETE");
}
