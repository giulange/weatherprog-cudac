%% single couple
iFIL1   = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/temp_min_h24-20110101-gprism-80.tif';
iFIL2   = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/temp_min_h24-20110102-gprism-80.tif';
Z1 = geotiffread(iFIL1);
Z2 = geotiffread(iFIL2);

%% create case study with larger number of pixels

copyfile(   '/media/DATI/wg-pedology/db-backup/LIFE+/40_LiDAR-DTM/Lidar_mosaico_dtm_tif/dtm_valle_telesina.tif',...
            '/home/giuliano/git/cuda/weatherprog-cudac/data/')
info = geotiffinfo( '/home/giuliano/git/cuda/weatherprog-cudac/data/dtm_valle_telesina.tif' );
for ii = 1:31
    fprintf('file: %2d\n',ii);
    Zi = ones(info.Height,info.Width)*ii;
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi = ['/home/giuliano/git/cuda/weatherprog-cudac/data/temp_min_h24-201101',day,'-gprism-80.tif'];
    geotiffwrite(iFILi,Zi,info.RefMatrix, ...
        'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
end

%% any number of Nmaps
clear Z
% ------------------
Nmaps = 31;
iDIR 	= '/home/giuliano/git/cuda/weatherprog-cudac/data';
% iDIR 	= '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps';
refFIL  = 'dtm_valle_telesina.tif'; % { 'dem5.tif' , ... }
oFILc 	= '/home/giuliano/git/cuda/weatherprog-cudac/data/sum_C.tif';
oFILcu	= '/home/giuliano/git/cuda/weatherprog-cudac/data/sum_CUDA.tif';
% ------------------

info    = geotiffinfo( fullfile(iDIR,refFIL) );
Om      = zeros( info.Height, info.Width );
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,['temp_min_h24-201101',day,'-gprism-80.tif'] );
    Z = geotiffread(iFILi);
    Om = Om + Z;
end

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))

clear Z
