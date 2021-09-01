#include <stdio.h>
#include <zlib.h>
#include <iostream>
#include <string>

#include "kseq.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/dynamic_bitset.hpp>

KSEQ_INIT(gzFile, gzread)

inline std::tuple<std::vector<uint>, std::vector<uint>, std::vector<double>, std::vector<std::string>> pairsnp(const char *fasta, int n_threads, int dist, int knn)
{
    // open filename and initialise kseq
    int l;
    gzFile fp = gzopen(fasta, "r");
    kseq_t *seq = kseq_init(fp);

    size_t n_seqs = 0;
    size_t seq_length;

    // initialise bitmaps
    std::vector<std::string> seq_names;
    std::vector<boost::dynamic_bitset<>> A_snps;
    std::vector<boost::dynamic_bitset<>> C_snps;
    std::vector<boost::dynamic_bitset<>> G_snps;
    std::vector<boost::dynamic_bitset<>> T_snps;
    std::string consensus;

    while (true)
    {
        l = kseq_read(seq);

        if (l == -1) // end of file
            break;
        if (l == -2)
        {
            throw std::runtime_error("Error reading FASTA!");
        }
        if (l == -3)
        {
            throw std::runtime_error("Error reading FASTA!");
        }

        // check sequence length
        if ((n_seqs > 0) && (seq->seq.l != seq_length))
        {
            throw std::runtime_error("Error reading FASTA, variable sequence lengths!");
        }
        seq_length = seq->seq.l;

        seq_names.push_back(seq->name.s);
        boost::dynamic_bitset<> As(seq_length);
        boost::dynamic_bitset<> Cs(seq_length);
        boost::dynamic_bitset<> Gs(seq_length);
        boost::dynamic_bitset<> Ts(seq_length);

        for (size_t j = 0; j < seq_length; j++)
        {

            seq->seq.s[j] = std::toupper(seq->seq.s[j]);

            switch (seq->seq.s[j])
            {
            case 'A':
                As[j] = 1;
                break;
            case 'C':
                Cs[j] = 1;
                break;
            case 'G':
                Gs[j] = 1;
                break;
            case 'T':
                Ts[j] = 1;
                break;

            // M = A or C
            case 'M':
                As[j] = 1;
                Cs[j] = 1;
                break;

            // R = A or G
            case 'R':
                As[j] = 1;
                Gs[j] = 1;
                break;

            // W = A or T
            case 'W':
                As[j] = 1;
                Ts[j] = 1;
                break;

            // S = C or G
            case 'S':
                Cs[j] = 1;
                Gs[j] = 1;
                break;

            // Y = C or T
            case 'Y':
                Cs[j] = 1;
                Ts[j] = 1;
                break;

            // K = G or T
            case 'K':
                Gs[j] = 1;
                Ts[j] = 1;
                break;

            // V = A,C or G
            case 'V':
                As[j] = 1;
                Cs[j] = 1;
                Gs[j] = 1;
                break;

            // H = A,C or T
            case 'H':
                As[j] = 1;
                Cs[j] = 1;
                Ts[j] = 1;
                break;

            // D = A,G or T
            case 'D':
                As[j] = 1;
                Gs[j] = 1;
                Ts[j] = 1;
                break;

            // B = C,G or T
            case 'B':
                Cs[j] = 1;
                Gs[j] = 1;
                Ts[j] = 1;
                break;

            // N = A,C,G or T
            default:
                As[j] = 1;
                Cs[j] = 1;
                Gs[j] = 1;
                Ts[j] = 1;
                break;
            }
        }
        // As.runOptimize();
        A_snps.push_back(As);
        // Cs.runOptimize();
        C_snps.push_back(Cs);
        // Gs.runOptimize();
        G_snps.push_back(Gs);
        // Ts.runOptimize();
        T_snps.push_back(Ts);

        n_seqs++;
    }
    kseq_destroy(seq);
    gzclose(fp);

    std::vector<uint> rows;
    std::vector<uint> cols;
    std::vector<double> distances;

  #pragma omp parallel for ordered shared(A_snps, C_snps \
    , G_snps, T_snps, seq_length \
    , n_seqs, seq_names, dist\
    , rows, cols, distances \
    , knn) default(none) schedule(static,1) num_threads(n_threads)
    for (size_t i = 0; i < n_seqs; i++) {

        std::vector<int> comp_snps(n_seqs);
        boost::dynamic_bitset<> res(seq_length);

        size_t start;
        if (knn < 0)
        {
            start = i + 1;
        }
        else
        {
            start = 0;
        }

        for (size_t j = start; j < n_seqs; j++)
        {

            res = A_snps[i] & A_snps[j];
            res |= C_snps[i] & C_snps[j];
            res |= G_snps[i] & G_snps[j];
            res |= T_snps[i] & T_snps[j];

            comp_snps[j] = seq_length - res.count();
        }

        // if using knn find the distance needed
        if (knn >= 0)
        {
            std::vector<int> s_comp = comp_snps;
            std::sort(s_comp.begin(), s_comp.end());
            dist = s_comp[knn + 1];
            start = 0;
        }
        else
        {
            start = i + 1;
        }

// output distances
#pragma omp critical
        for (size_t j = start; j < n_seqs; j++)
        {
            if ((dist == -1) || (comp_snps[j] <= dist))
            {
                rows.push_back(i);
                cols.push_back(j);
                distances.push_back(comp_snps[j]);
            }
        }
    }

    return std::make_tuple(rows, cols, distances, seq_names);
}
