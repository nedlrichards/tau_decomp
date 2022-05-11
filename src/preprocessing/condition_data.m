%% Create a stable version of gridded data for contour processing
load('../../data/raw/section4_fields.mat')
x_a = xx;
z_a = zz;
lat = 30.;
lon = -140.;

% TEOS-10 versions of measured variables
press = gsw_p_from_z(-z_a, lat);
SA = gsw_SA_from_SP(sali.', press, lon, lat);
CT = gsw_CT_from_pt(SA, thetai.');

[SA_stable, CT_stable] = gsw_stabilise_SA_CT(SA, CT, press, lon, lat);

save('../../data/processed/stablized_field.mat', "CT_stable", "SA_stable", ...
      "press","x_a","z_a","lat","lon")