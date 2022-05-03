function [GSW_var, rho_mean, rho_mat] = Integrating_in_rho_SODA(T, S, depth, LON, LAT)

% Obtaining the Along-density mean of Temperature, Salinity and sound speed.

% % ============================SODA 2018 VALUES for the deep ones
dr = .04;
min_rho = 20+1000;
max_rho = 29+1000;
rho_v = [min_rho:dr:max_rho];

% % ============================NORSE VALUES for the deep ones
%dr = .02;
%min_rho = 26.2+1000;
%max_rho = 28.9+1000;
%rho_v = [min_rho:dr:max_rho];
%
% % ============================SUNRISE VALUES for the deep ones
% dr = .001;
% min_rho = 22.8+1000;
% max_rho = 27.2+1000;
% rho_v = [min_rho:dr:max_rho];
%
%
% created by : Ale Sanchez-Rios 2021,  ¸.·´¯`·...¸><((((º>  Meaw.....
% modified from old code from Jen Mackinnon.

[rd cd] = size(depth);

if rd ==1
    depth= depth';
    %     depth_T= depth_T';
end

if nanmean(depth) <= 0
    depth = -depth;
end
% Obtaining density
SA = gsw_SA_from_SP(S, depth, LON, LAT);
CT = gsw_CT_from_t(SA,T,depth);
rho = gsw_rho(SA,CT, depth);

% estimating the size of the file
r_size = length(rho_v);

[r, c ]= size(rho);
z = depth(:,ones(c,1));
sg0 = rho_v;
disp(['This file has ', num2str(r), ' depth rows, and ', num2str(c), ' columns'])

% Obtaining sound speed
sound_speed = gsw_sound_speed_CT_exact(SA, CT, depth );


% creates matrix to store
s_pd=NaN*ones(length(sg0),c);
t_pd=NaN*ones(length(sg0),c);
c_pd=NaN*ones(length(sg0),c); 
z_pd=NaN*ones(length(sg0),c);


% This is for the case where we want to obtained the mean for a larger data
% set (Working in this section but should work regardless)
for iii=1:c
    ig=find(~isnan(rho(:,(iii))));
    if isempty(ig)==1
    s_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    t_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    c_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    z_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    elseif length(ig)<=5
    s_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    t_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    c_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    z_pd_T(:,iii) = ones(length(sg0), 1)*nan;
    else
    [sgs,xxx]=sort(rho(ig,(iii)));
    s_pd_T(:,iii)=interp1(sgs,SA(ig(xxx),(iii)),sg0(:));
    t_pd_T(:,iii)=interp1(sgs,CT(ig(xxx),(iii)),sg0(:));
    c_pd_T(:,iii)=interp1(sgs,sound_speed(ig(xxx),(iii)),sg0(:));
    z_pd_T(:,iii)=interp1(sgs,z(ig(xxx),(iii)),sg0(:));
    %     sp_pd_T(:,iii)=interp1(sgs,sound_speed(ig(xxx),(iii)),sg0(:));
    end
end

% Mean profiles of S and T along density lines
s_pdm =nanmean(s_pd_T,2);
t_pdm =nanmean(t_pd_T,2);
c_pdm =nanmean(c_pd_T,2);
z_pdm =nanmean(z_pd_T,2);

rho_mean.s_pdm = s_pdm;
rho_mean.t_pdm = t_pdm;
rho_mean.c_pdm = c_pdm;
rho_mean.z_pdm = z_pdm;
rho_mean.rho_v = rho_v;

rho_mat.s_pd_T = s_pd_T;
rho_mat.t_pd_T = t_pd_T;
rho_mat.c_pd_T = c_pd_T;
rho_mat.z_pd_T = z_pd_T;
rho_mat.rho_v = rho_v;

GSW_var.SA = SA;
GSW_var.CT = CT;
GSW_var.rho = rho;
GSW_var.sound_speed = sound_speed;
