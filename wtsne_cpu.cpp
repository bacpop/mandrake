/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <chrono>

#include <iostream>
using namespace std;

#ifndef DIM
#define DIM 2
#endif

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

typedef double real;
typedef long long longint;

longint *I, *J, maxIter, workerCount, nn, ne, nRepuSamp;
real *P, *Y, eta0, *weights;
const gsl_rng_type * gsl_T;
gsl_rng *gsl_r_nn, *gsl_r_ne;
int bInit;

void freeMemory()
{
    delete[] I;
    delete[] J;
    delete[] P;
    delete[] Y;
    delete[] weights;
}

void loadP(const char *fnameP, int bBinaryInput)
{
    FILE *fpP = fopen(fnameP, "r");

    if (bBinaryInput)
    {
        if (fread(&nn, sizeof(longint), 1, fpP)!=1) perror("Error in reading nn");
        if (fread(&ne, sizeof(longint), 1, fpP)!=1) perror("Error in reading ne");
        I = new longint[ne];
        J = new longint[ne];
        P = new real[ne];
        if (fread(I, sizeof(longint), ne, fpP)!=ne) perror("Error in reading I");
        if (fread(J, sizeof(longint), ne, fpP)!=ne) perror("Error in reading J");
        if (fread(P, sizeof(real), ne, fpP)!=ne) perror("Error in reading P");
    }
    else
    {
        if (fscanf(fpP, "%lld %lld", &nn, &ne)!=2) perror("Error in reading nn and ne from text file");
        I = new longint[ne];
        J = new longint[ne];
        P = new real[ne];
        for (longint e=0; e<ne; e++)
            if (fscanf(fpP, "%lld %lld %lg", I+e, J+e, P+e)!=3) perror("Error in reading triplets from text file");
    }
    fclose(fpP);
}

void loadWeights(const char *fnameWeights, int bBinaryInput)
{
    weights = new real[nn];
    if (strcmp(fnameWeights, "none")==0)
    {
        for (longint i=0; i<nn; i++)
            weights[i] = 1.0;
    }
    else
    {
        FILE *fpWeights = fopen(fnameWeights, "r");
        if (bBinaryInput)
            if (fread(weights, sizeof(real), nn, fpWeights)!=nn) perror("Error in reading weights from binary file");
        else
            for (longint i=0; i<nn; i++)
                if (fscanf(fpWeights, "%lg", weights+i)!=1) perror("Error in reading weights from text file");
        fclose(fpWeights);
    }
}

void loadY0(const char *fnameY0, int bBinaryInput)
{
    Y = new real[nn*DIM];
    if (strcmp(fnameY0, "none")==0)
    {
        srand(0);
        for (longint i = 0; i < nn; i++)
            for (longint d = 0; d < DIM; d++)
                Y[d + i* DIM] = rand() * 1e-4 / RAND_MAX;
    }
    else
    {
        FILE *fpY0 = fopen(fnameY0, "r");
        if (bBinaryInput)
            if (fread(Y, sizeof(real), nn*2, fpY0)!=nn*2) perror("Error in reading Y0 from binary file");
        else
            for (longint i=0; i<nn; i++)
                if (fscanf(fpY0, "%lg %lg", Y+i*DIM, Y+i*DIM+1)!=2) perror("Error in reading Y0 from text file");
        fclose(fpY0);
    }
}

void
saveY (const char* fnameY)
{
	FILE *fpY = fopen (fnameY, "w+");
	for (longint i = 0; i < nn; i++) {
		for (longint d = 0; d < DIM; d++) {
			fprintf (fpY, "%.6f", Y[d + i * DIM]);
			if (d<DIM-1)
				fprintf(fpY, " ");
		}
		fprintf(fpY, "\n");
	}

	fclose (fpY);
}

void wtsne()
{
    real Psum = 0.0;
    for (longint e=0; e<ne; e++) Psum += P[e];
    for (longint e=0; e<ne; e++) P[e] /= Psum;

    real weights_sum = 0.0;
    for (longint i=0; i<nn; i++) weights_sum += weights[i];
    for (longint i=0; i<nn; i++) weights[i] /= weights_sum;

    gsl_rng_env_setup();
    gsl_T = gsl_rng_default;
    gsl_r_nn = gsl_rng_alloc(gsl_T);
    gsl_r_ne = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r_nn, 314159);
    gsl_rng_set(gsl_r_ne, 271828);

    gsl_ran_discrete_t *gsl_de = gsl_ran_discrete_preproc(ne, P);
    gsl_ran_discrete_t *gsl_dn = gsl_ran_discrete_preproc(nn, weights);

    const real nsq = nn * (nn-1);

    real Eq = 1.0;
    for (longint iter=0; iter<maxIter; iter++)
    {
        real eta = eta0 * (1-(real)iter/maxIter);
        eta = MAX(eta, eta0 * 1e-4);
        real c = 1.0 / (Eq * nsq);

        real qsum = 0;
        longint qcount = 0;

        real attrCoef = (bInit && iter<maxIter/10) ? 8 : 2;
        real repuCoef = 2 * c / nRepuSamp * nsq;
        #pragma omp parallel for reduction(+:qsum,qcount)
        for (longint worker=0; worker<workerCount; worker++)
        {
            real dY[DIM];

            longint e = gsl_ran_discrete(gsl_r_ne, gsl_de) % ne;
            longint i = I[e];
            longint j = J[e];

            for (longint r=0; r<nRepuSamp+1; r++)
            {
                longint k, l;
                if (r==0)
                {
                    k = i;
                    l = j;
                }
                else
                {
                    k = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
                    l = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
                }
                if (k == l) continue;

                longint lk = k * DIM;
                longint ll = l * DIM;
                real dist2 = 0.0;
                for (longint d=0; d<DIM; d++)
                {
                    dY[d] = Y[d+lk]-Y[d+ll];
                    dist2 += dY[d] * dY[d];
                }
                real q = 1.0 / (1 + dist2);

                real g;
                if (r==0)
                    g = -attrCoef * q;
                else
                    g = repuCoef * q * q;

                for (longint d=0; d<DIM; d++)
                {
                    real gain = eta * g * dY[d];
                    Y[d+lk] += gain;
                    Y[d+ll] -= gain;
                }
                qsum += q;
                qcount++;
            }
        }
        Eq = (Eq * nsq + qsum) / (nsq + qcount);

        if (iter % MAX(1,maxIter/1000)==0)
        {
            printf("%cOptimizing\t eta=%f Progress: %.3lf%%, Eq=%.20f", 13, eta, (real)iter / maxIter * 100, 1.0/(c*nsq));
            fflush(stdout);
        }
    }

    gsl_ran_discrete_free(gsl_de);
    gsl_ran_discrete_free(gsl_dn);
}

int main(int argc, char **argv)
{
    printf("Usage: wtsne_stoc bBinaryInput P_file Y_file weights_file Y0_file maxIter eta0 nRepuSamp workerCount bInit\n");
    int bBinaryInput = atoi(argv[1]);
    const char *fnameP = argv[2];
    const char *fnameY = argv[3];
    const char *fnameWeights = argv[4];
    const char *fnameY0 = argv[5];
    maxIter = atoi(argv[6]);
    eta0 = atof(argv[7]);
    nRepuSamp = atoi(argv[8]);
    workerCount = atoi(argv[9]);
    bInit = atoi(argv[10]);

    loadP(fnameP, bBinaryInput);
    loadY0(fnameY0, bBinaryInput);
    loadWeights(fnameWeights, bBinaryInput);

    auto start_time = chrono::steady_clock::now();
	wtsne ();
	auto end_time = chrono::steady_clock::now();
	auto diff = end_time - start_time;
	cout << endl << "wtsne used " << chrono::duration <double, std::ratio<1>> (diff).count() << " seconds." << endl;

    saveY(fnameY);

    freeMemory();

    return 0;
}

