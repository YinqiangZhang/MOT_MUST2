% Compile libraries and download network

[home_dir, name, ext] = fileparts(mfilename('fullpath'));

warning('ON', 'ECO:install')

% mtimesx
if exist('external_libs/mtimesx', 'dir') == 7
    cd external_libs/mtimesx
    mtimesx_build;
    cd(home_dir)
else
    error('ECO:install', 'Mtimesx not found.')
end

% PDollar toolbox
if exist('external_libs/pdollar_toolbox/external', 'dir') == 7
    cd external_libs/pdollar_toolbox/external
    toolboxCompile;
    cd(home_dir)
else
    warning('ECO:install', 'PDollars toolbox not found. Clone this submodule if you want to use HOG features. Skipping for now.')
end

% matconvnet
%{
if exist('external_libs/matconvnet/matlab', 'dir') == 7
    cd external_libs/matconvnet/matlab
%     vl_compilenn;
    vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-8.0', ...
               'cudaMethod', 'nvcc');
    status = movefile('mex/vl_*.mex*');
    cd(home_dir)
    
    % donwload network
    cd feature_extraction
    mkdir networks
    cd networks
    if ~(exist('imagenet-vgg-m-2048.mat', 'file') == 2)
        disp('Downloading the network "imagenet-vgg-m-2048.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat', 'imagenet-vgg-m-2048.mat')
        disp('Done!')
    end
    cd(home_dir)
else
    warning('ECO:install', 'Matconvnet not found. Clone this submodule if you want to use CNN features. Skipping for now.')
end
%}