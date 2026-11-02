#include "../../../Sources/VectorCoreC/common/vc_lapack_workspace.h"
#include <assert.h>
#include <stdio.h>
#include <float.h>

int main(void) {
    size_t value = 0, bytes = 0;
    int32_t f = 0, i = 0;
    assert(vc_la_mul(SIZE_MAX, 0, &value) && value == 0);
    assert(vc_la_mul(SIZE_MAX, 1, &value) && value == SIZE_MAX);
    assert(!vc_la_mul(SIZE_MAX, 2, &value));
    assert(vc_la_add(SIZE_MAX-1, 1, &value) && value == SIZE_MAX);
    assert(!vc_la_add(SIZE_MAX, 1, &value));
    assert(vc_la_bytes(SIZE_MAX/8, 8, &value));
    assert(!vc_la_bytes(SIZE_MAX/8+1, 8, &value));
    assert(!vc_la_bytes(1, 0, &value));
    assert(!vc_la_length(0, &f));
    assert(vc_la_length(INT32_MAX, &f) && f == INT32_MAX);
    assert(!vc_la_length((size_t)INT32_MAX+1, &f));
    assert(vc_la_svd_work(-1,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_svd_work(0,&f,&i) == 0 && f == 1 && i == 1);
    assert(vc_la_svd_work(2,&f,&i) == 0 && f == 30 && i == 16);
    // Exact sufficient-work polynomial cutoffs, independently evaluated in int64.
    int32_t k = 1;
    while (4*(int64_t)(k+1)*(k+1)+7*(int64_t)(k+1) <= INT32_MAX) ++k;
    assert(vc_la_svd_work(k,&f,&i) == 0 && f == 4*(int64_t)k*k+7*(int64_t)k && i == 8*(int64_t)k);
    assert(vc_la_svd_work(k+1,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_svd_work(INT32_MAX,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_eigen_work(-1,1,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_eigen_work(0,1,&f,&i) == 0 && f == 1 && i == 1);
    assert(vc_la_eigen_work(1,1,&f,&i) == 0 && f == 1 && i == 1);
    assert(vc_la_eigen_work(2,1,&f,&i) == 0 && f == 21 && i == 13);
    k = 1;
    while (1+6*(int64_t)(k+1)+2*(int64_t)(k+1)*(k+1) <= INT32_MAX) ++k;
    assert(vc_la_eigen_work(k,1,&f,&i) == 0 && f == 1+6*(int64_t)k+2*(int64_t)k*k && i == 3+5*(int64_t)k);
    assert(vc_la_eigen_work(k+1,1,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_eigen_work((INT32_MAX-1)/2,0,&f,&i) == 0 && f == INT32_MAX && i == 1);
    assert(vc_la_eigen_work((INT32_MAX-1)/2+1,0,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    assert(vc_la_eigen_work(INT32_MAX,1,&f,&i) == VC_LAPACK_WORKSPACE_OVERFLOW);
    float invalid[] = {0,-1,NAN,INFINITY,-INFINITY,FLT_MAX,(float)INT32_MAX};
    for (size_t j=0;j<sizeof(invalid)/sizeof(invalid[0]);++j)
        assert(vc_la_query(invalid[j],1,&f,&bytes) == VC_LAPACK_INVALID_QUERY);
    assert(vc_la_query(1.25f,1,&f,&bytes) == 0 && f == 3 && bytes == 12);
    // T19: `minimum` is an allocation floor, never a validity threshold for a positive finite answer.
    assert(vc_la_query(3,4,&f,&bytes) == 0 && f == 4 && bytes == 16);
    assert(vc_la_query(1,30,&f,&bytes) == 0 && f == 30 && bytes == 120);
    assert(vc_la_query(64,60,&f,&bytes) == 0 && f == 66 && bytes == 264);
    assert(vc_la_query(1,INT32_MAX,&f,&bytes) == 0 && f == INT32_MAX && bytes == (size_t)INT32_MAX*sizeof(float));
    assert(vc_la_query(3,0,&f,&bytes) == VC_LAPACK_INVALID_QUERY);
    float neighboring[] = {nextafterf(16777216.0f,0),16777216.0f,nextafterf(16777216.0f,INFINITY)};
    for (size_t j=0;j<3;++j) {
        int64_t rounded = (int64_t)ceil((double)neighboring[j]);
        assert(vc_la_query(neighboring[j],1,&f,&bytes) == 0);
        assert(f == rounded+rounded/64+1 && bytes == (size_t)f*sizeof(float));
    }
    float edge = (float)((int64_t)INT32_MAX*64/65);
    while ((double)edge + floor((double)edge/64)+1 > INT32_MAX) edge=nextafterf(edge,0);
    assert(vc_la_query(edge,1,&f,&bytes) == 0);
    assert(vc_la_query(nextafterf(edge,INFINITY),1,&f,&bytes) == VC_LAPACK_INVALID_QUERY);
    assert(vc_la_integer_query(-1,1,&bytes) == VC_LAPACK_INVALID_QUERY);
    assert(vc_la_integer_query(0,1,&bytes) == VC_LAPACK_INVALID_QUERY);
    assert(vc_la_integer_query(3,4,&bytes) == 0 && bytes == 16);
    assert(vc_la_integer_query(4,4,&bytes) == 0 && bytes == 16);
    assert(vc_la_integer_query(5,4,&bytes) == 0 && bytes == 20);
    assert(vc_la_integer_query(3,0,&bytes) == VC_LAPACK_INVALID_QUERY);
    printf("L01_WORKSPACE_COMPLETE eigenVectorCutoff=%d queryEdge=%.0f\n", k, (double)edge);
    return 0;
}
