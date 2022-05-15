%% Create a stable version of gridded data for contour processing
load('../../data/raw/section4_fields.mat')
x_a = xx;
z_a = zz;
lat = 30.;
lon = -140.;

savefile = '../../data/processed/stablized_field.mat';

if isfile(savefile)
    delete savefile
end



% TEOS-10 versions of measured variables
press = gsw_p_from_z(-z_a, lat);
SA = gsw_SA_from_SP(sali.', press, lon, lat);
CT = gsw_CT_from_pt(SA, thetai.');

CT_stable = zeros(size(CT));
SA_stable = zeros(size(SA));
save(savefile, "CT_stable", "SA_stable", "press", "x_a", "z_a", "lat", "lon");

for i=[1:size(SA,1)]
    tic;
    [SA_tmp, CT_tmp] = gsw_stabilise_SA_CT(SA(i, :), CT(i, :), press, lon, lat);
    load(savefile);
    SA_stable(i, :) = SA_tmp;
    CT_stable(i, :) = CT_tmp;
    save(savefile, "CT_stable", "SA_stable", "press", "x_a", "z_a", "lat", "lon");
    toc;
    disp(i)
end
