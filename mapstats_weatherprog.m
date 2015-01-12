%% create case study with a larger number of pixels

% % % % ::LARGE 1m::
% % % refFIL  = '/media/DATI/wg-pedology/db-backup/LIFE+/40_LiDAR-DTM/Lidar_mosaico_dtm_tif/dtm_valle_telesina.tif';
% % % Fname   = 'L1_temp_min_h24-201101';
% % % % ::LARGE 5m::
% % % % refFIL  = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/idata/grid/dem5.tif';
% % % % Fname   = 'L5_temp_min_h24-201101';
% % % 
% % % info    = geotiffinfo( refFIL );
% % % for ii = 1:31
% % %     Zi  = ones(info.Height,info.Width)*ii;
% % %     if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
% % %     iFILi = ['/home/giuliano/git/cuda/weatherprog-cudac/data/',Fname,day,'-',METHOD,'-80.tif'];
% % %     fprintf('file: %2d\t%s\n',ii,iFILi);
% % %     geotiffwrite(iFILi,Zi,info.RefMatrix, ...
% % %         'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
% % % end

%% create ROI

% iFIL = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/rain_cum_h24-20110121-idw2-80.tif';
% info = geotiffinfo( iFIL );
% 
% ROI_VT = ones(info.Height,info.Width,'uint8');
% 
% iFIL_ROI = '/home/giuliano/git/cuda/weatherprog-cudac/data/roi_vt.tif';
% geotiffwrite(iFIL_ROI,ROI_VT,info.RefMatrix, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
% 

%% create random ROI

% iFIL = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/rain_cum_h24-20110121-idw2-80.tif';
% info = geotiffinfo( iFIL );
% 
% ROI_VT = rand(info.Height,info.Width);
% ROI_VT(ROI_VT>0.5)  = 1;
% ROI_VT(ROI_VT<=0.5) = 0;
% ROI_VT = uint8(ROI_VT);
% 
% iFIL_ROI = '/home/giuliano/git/cuda/weatherprog-cudac/data/roi_vt.tif';
% geotiffwrite(iFIL_ROI,ROI_VT,info.RefMatrix, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
% 

%% ::INPUT::

Nmaps = 31;

% ::SMALL::
iDIR        = '/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps';
Fname       = 'rain_cum_h24-201101';
METHOD      = 'idw2'; % 'gprism';
iFIL_ROI    = '/home/giuliano/git/cuda/weatherprog-cudac/data/roi_vt.tif';
ROI         = double(geotiffread(iFIL_ROI));
% ::LARGE::
% iDIR        = '/home/giuliano/git/cuda/weatherprog-cudac/data';
% Fname       = 'L5_temp_min_h24-201101';
% Fname       = 'L1_temp_min_h24-201101';

% output files:
oFILc       = '/home/giuliano/git/cuda/weatherprog-cudac/data/out_C.tif';
oFILcu      = '/home/giuliano/git/cuda/weatherprog-cudac/data/out_CUDA.tif';

% print config:
fpf         = @(s) fprintf('\n\n\t%s\t%s\n\n',['STAT::',s],datestr(now,'yyyy-mmm-dd, HH:MM:SS'));
%% --- ::SUM::

fpf('SUM');

info    = geotiffinfo( fullfile(iDIR,[Fname,num2str(10),'-',METHOD,'-80.tif']) );
Om      = zeros( info.Height, info.Width );
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z = geotiffread(iFILi);
    tic;
    Om = Om + Z;
    myTOC = myTOC + toc;
end
tic;
Om = Om .*ROI;
myTOC = myTOC + toc;

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii

%% --- ::MIN::

fpf('MIN');

info    = geotiffinfo( fullfile(iDIR,[Fname,num2str(10),'-',METHOD,'-80.tif']) );
Om      = ones( info.Height, info.Width )*1000;
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z = geotiffread(iFILi);
    tic;
    Om = min(Om, Z);
    myTOC = myTOC + toc;
end
tic;
Om = Om .*ROI;
myTOC = myTOC + toc;

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear day

%% --- ::MAX::

fpf('MAX');

info    = geotiffinfo( fullfile(iDIR,[Fname,num2str(10),'-',METHOD,'-80.tif']) );
Om      = ones( info.Height, info.Width )*-1000;
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z = geotiffread(iFILi);
    tic;
    Om = max(Om, Z);
    myTOC = myTOC + toc;
end
tic;
Om = Om .*ROI;
myTOC = myTOC + toc;

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii

%% --- ::MEAN::

fpf('MEAN');

info    = geotiffinfo( fullfile(iDIR,[Fname,num2str(10),'-',METHOD,'-80.tif']) );
Om      = zeros( info.Height, info.Width );
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z = geotiffread(iFILi);
    tic;
    Om = Om + Z;
    myTOC = myTOC + toc;
end
Om      = Om .* ROI / Nmaps;

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii
%% --- ::STD::

fpf('STD');

info    = geotiffinfo( fullfile(iDIR,[Fname,num2str(10),'-',METHOD,'-80.tif']) );
Om      = zeros( info.Height, info.Width );
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z(:,:,ii) = geotiffread(iFILi);   
end
tic;
MEAN = mean(Z,3);
for ii = 1:Nmaps
    Z(:,:,ii) = (Z(:,:,ii)-MEAN).^2;
end
Om = sqrt(sum(Z,3) / (Nmaps-1));
myTOC = myTOC + toc;

Oc      = geotiffread(oFILc);
Ocu     = geotiffread(oFILcu);

fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om(:)-Ocu(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii
%% --- ::2d-SUM::

fpf('2d-SUM');

Om_s    = zeros( Nmaps, 1 );
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z   = geotiffread(iFILi);
    tic;
    Om_s(ii)  = sum(Z(:).*ROI(:));
    myTOC = myTOC + toc;
end

% Oc      = [1,1];
Ocu         = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOT');
Oc          = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTc');

fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om_s(:)-Ocu(:,1)))
fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om_s(:)-Oc(:,1)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii

%% --- ::2d-MEAN::

fpf('2d-MEAN');

Om_m    = zeros( Nmaps, 1 );
myTOC   = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z   = geotiffread(iFILi);
    tic;
    Om_m(ii)  = sum(Z(:).*ROI(:)) / (numel(Z)-1);
    myTOC = myTOC + toc;
end

% Oc      = [1,1];
Ocu         = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTcu');
Oc          = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTc');

% fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om(:)-Oc(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om_m(:)-Ocu(:,1)))
fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om_m(:)-Oc(:,1)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear Z iFILi ii

%% --- ::2d-std::
% STD = sqrt( sum( (A-mean(A)).^2 ) / (numel(A)-1) );

fpf('2d-STD');

Om_std      = zeros( Nmaps, 1 );
myTOC       = 0;
for ii = 1:Nmaps
    if ii<10, day = ['0',num2str(ii)]; else day = num2str(ii); end
    iFILi   = fullfile( iDIR,[Fname,day,'-',METHOD,'-80.tif'] );
    Z       = geotiffread(iFILi);
    %fprintf('%s\n',iFILi)
    tic;
    Om_std(ii) = sum( ( (Z(:)-Om_m(ii)) .* ROI(:) ).^2 );
%     Om_std(ii) = sum( (Z(:)-Om_m(ii)).^2 );
    myTOC   = myTOC + toc;
end
% sqrt( XXX / (numel(A)-1) );
Om_std      = sqrt( Om_std / (numel(Z)-1) );
Om_std      = [Om_m,Om_m-Om_std,Om_m+Om_std];

Ocu         = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTcu');
Oc          = load('/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTc');

fprintf('%40s %5.2f\n','Difference between MatLab and CUDA:',sum(Om_std(:)-Ocu(:)))
fprintf('%40s %5.2f\n','Difference between MatLab and C:',sum(Om_std(:)-Oc(:)))
fprintf('%40s %5.2f [ms]\n','MatLab elapsed time:',myTOC*1000)

clear ii
