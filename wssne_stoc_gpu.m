function Y = wssne_stoc_gpu(P, maxIter, weights, Y0, eta0, nRepuSamp, blockSize, blockCount, bInit)
nn = size(P,1);
if ~exist('weights', 'var') || isempty(weights)
    weights = ones(nn,1);
end
if ~exist('Y0', 'var') || isempty(Y0)
    rng(0, 'twister');
    Y0 = randn(nn,2)*1e-4;
%     Y0 = randn(nn,10)*1e-4;
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = 1e5;
end
if ~exist('eta0', 'var') || isempty(eta0)
    eta0 = 0.1;
end
if ~exist('nRepuSamp', 'var') || isempty(nRepuSamp)
    nRepuSamp = 5;
end
if ~exist('blockSize', 'var') || isempty(blockSize)
    blockSize = 128;
end
if ~exist('blockCount', 'var') || isempty(blockCount)
    blockCount = 128;
end
if ~exist('bInit', 'var') || isempty(bInit)
    bInit = 0;
end
if bInit~=0
    bInit = 1;
end

[I,J,V] = find(P);
ne = length(I);

fnameP = tempname;
fid = fopen(fnameP, 'w+');
fwrite(fid, [nn ne], 'uint64');
fwrite(fid, I-1, 'uint64');
fwrite(fid, J-1, 'uint64');
fwrite(fid, V, 'double');
fclose(fid);

fnameWeights = tempname;
fid = fopen(fnameWeights, 'w+');
fwrite(fid, weights, 'double');
fclose(fid);

fnameY0 = tempname;
fid = fopen(fnameY0, 'w+');
fwrite(fid, Y0', 'float');
fclose(fid);

fnameY = tempname;

cmd_str = sprintf('./wssne_stoc_gpu 1 %s %s %s %s %d %f %d %d %d %d', ...
    fnameP, fnameY, fnameWeights, fnameY0, maxIter, eta0, nRepuSamp, blockSize, blockCount, bInit);

status = system(cmd_str);

fid = fopen(fnameY);
res = textscan(fid, '%f %f', 'CollectOutput', 1);
% res = textscan(fid, [repmat('%f ', 1,9), '%f'], 'CollectOutput', 1);
fclose(fid);
Y = res{1};

delete(fnameP);
delete(fnameY);
delete(fnameWeights);
delete(fnameY0);
