
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

int check_params(int BN, int BM, int BK, int WM, int WN, int NUM_THREADS, int TM, int TN, int WN_ITER, int WN_ITER){
    int WARPSIZE = 32;
    int NUM_WARPS = NUM_THREADS / WARPSIZE;
    if (! (BN % WN == 0 && BM % WM == 0)){
        printf("Error: BN % WN must be 0 and BM % WM must be 0.");
        fflush(stdout);
        return 1;
    }
    
    if (! ((BN / WN) * (BM / WM) == NUM_WARPS)){
        printf("Error: (BN / WN) * (BM / WM) must be equal to NUM_WARPS.");
        fflush(stdout);
        return 1;
    }
    
    if (! ((WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )){
        printf("Error: (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) must be 0.");
        fflush(stdout);
        return 1;
    }
    
    int WM_ITER= (WM * WN) / (WARPSIZE * TM * TN * WN_ITER);
    if (! (WM % WM_ITER == 0 && WN % WN_ITER == 0 )){
        printf ("Error: WM % WM_ITER must be 0 and WN % WN_ITER must be 0.");
        fflush(stdout);
        return 1;
    }

    if (! ((NUM_THREADS * 4) % BK == 0 )) {
        printf("Error: (NUM_THREADS * 4) % BK must be 0.");
        fflush(stdout);
        return 1;
    }
    
    if (! ((NUM_THREADS * 4) % BN == 0) ){
        printf("Error: (NUM_THREADS * 4) % BN must be 0.");
        fflush(stdout);
        return 1;
    }   
    
    if (! ( BN % (16 * TN) == 0 )){
        printf("Error: BN must be a multiple of 16 * TN.");
        fflush(stdout);
        return 1;
    }
    
    if (! ( BM % (16 * TM) == 0 )){
        printf("Error: BM must be a multiple of 16 * TM.");
        fflush(stdout);
        return 1;
    }
    if (! ( (BM * BK) % (4 * NUM_THREADS) == 0 )){
        printf("Error: (BM * BK) % (4 * NUM_THREADS) must be 0.");
        fflush(stdout);
        return 1;
    }
    if (! ((BN * BK) % (4 * NUM_THREADS) == 0 )){
       printf( "Error: (BN * BK) % (4 * NUM_THREADS) must be 0.");
       fflush(stdout);
       return 1;
    }
    return 0;
}
