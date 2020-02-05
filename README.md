# pathoSCE
Fast visualisation of the population structure of pathogens using Stochastic Cluster Embedding

## Notes from Zhirong

Denote nn=number_of_nodes and ne=number_of_edges

The program takes the following arguments:
bBinaryInput: 1 for binary input or 0 for text input
P_file: the path_name of the file containing the similarity matrix
Y_file: the path_name of the output which contains the final embedding coordinates
weights_file: the path_name of the file containing the importance weights of the instances. Specify "none" to use uniform weights.
Y0_file: the path_name of the file containing the initial embedding coordinates. Specify "none" to use uniform random initialization in [0,1] x 1e-4
maxIter: maximum iterations, usually larger is better but takes longer time
eta0: the starting learning rate, usually set to 1
nRepuSamp: number of samples in calculating the repulsive force, usually set to 1 or 5
blockSize: the block size used in CUDA. blockCount: the block size used in CUDA. A larger product blockCount x blockSize will make fuller use of GPU, but with higher risk of conflicting for small data sets. I used Tesla K40c GPU with blockSize=128 and blockCount=128 in the SCE paper.
bInit: 1 for over-exaggeration in early stage. I used 0 for all my experiments.


The program can take two type of inputs, either in text files or in binary files. Binary files are faster for big data but hard to read or edit.

For text type input:
The P_file is given by
- first row (two numbers, space delimited): nn ne - followed by ne rows, each for a nonzero entry (three numbers, space delimited): i j P_ij
The weights_file has nn rows, where each row is a number for the importance of the corresponding instance. If there is no preference of certain instances, you can use uniform weights or pass "none" to weights_file.
The Y0_file has nn rows, where each row has two numbers for the initial embedding coordinates. You can pass "none" to use 
For binary type input, the formats in Matlab code are:
fid = fopen(fnameP, 'w+');
fwrite(fid, [nn ne], 'uint64');
fwrite(fid, I-1, 'uint64');
fwrite(fid, J-1, 'uint64');
fwrite(fid, V, 'double');
fclose(fid);

fid = fopen(fnameWeights, 'w+');
fwrite(fid, weights, 'double');
fclose(fid);

fid = fopen(fnameY0, 'w+');
fwrite(fid, Y0', 'float');
fclose(fid);

The output Y_file is in plain text. It has nn rows, where each row has two numbers for the final embedding coordinates, delimited by space.
