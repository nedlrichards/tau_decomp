function [spice, GSW_var, rho_mean, rho_mat, new_mat, dr_TS]=spice_ale_v2_SODA(T, S, depth, LON, LAT, ALLmean)

%  SPICE AND ALONG-ISOPYCNALS VALUES-------
%   check Integratin in rho functin for the correct rho ranges
%   check SMS_TRAIXUS_rho_MEANS code, where we estimated the variable
%   ALLmean

    [rd cd] = size(depth);

if rd ==1
    depth= depth';
    %     depth_T= depth_T';
end

if nanmean(depth) <= 0
    depth = -depth;
end
%
depth_i = nanmean(depth,2);
%
[GSW_var, rho_mean, rho_mat] = Integrating_in_rho_SODA(T, S, depth_i, LON, LAT);

%% Obtaining the mean structure from all the variables

S_PDM = ALLmean.ST_PDM;
T_PDM = ALLmean.TT_PDM;
C_PDM = ALLmean.CS_PDM;

[r, c] = size(GSW_var.rho);

% Removing the mean to the section
 s_pda=rho_mat.s_pd_T - S_PDM(:)*ones(1,c);
 t_pda=rho_mat.t_pd_T - T_PDM(:)*ones(1,c);
 c_pda=rho_mat.c_pd_T - C_PDM(:)*ones(1,c);


% creating the matrix
new_matrixT_j = ones(r,c)*nan;
new_matrixS_j = ones(r,c)*nan;
new_matrixC_j = ones(r,c)*nan;

z_pd = rho_mat.z_pd_T;
for iii=1:c
    ig=find(~isnan(z_pd(:,(iii))));
    %     [sgs,xxx]=sort(rho(ig,(iii)));
    
    if nansum(~isnan(s_pda(:,iii))) <=1
    else
        new_matrixS_j(:,iii)=interp1(z_pd(ig,iii),s_pda(ig,iii),depth_i, 'nearest' ,'extrap');
        new_matrixT_j(:,iii)=interp1(z_pd(ig,iii),t_pda(ig,iii),depth_i, 'nearest' , 'extrap');
        new_matrixC_j(:,iii)=interp1(z_pd(ig,iii),c_pda(ig,iii),depth_i, 'nearest' ,'extrap');
    end
    
    ig=find(isnan(S(:,(iii))));
    
    new_matrixS_j(ig,iii)= nan;
    new_matrixT_j(ig,iii)= nan;
    new_matrixC_j(ig,iii)= nan;
end



new_mat.Salinity = new_matrixS_j;
new_mat.Temperature = new_matrixT_j;
new_mat.Sound_speed = new_matrixC_j;


%% estimated spice This might need correction for deeper depths.
alp_prof  = nanmean(nanmean(gsw_alpha(S,T, depth_i),2));
beta_prof = nanmean(nanmean(gsw_beta(S, T, depth_i),2));

r0 = 1000;
dr_salt = new_matrixS_j*r0*beta_prof;
dr_temp = -new_matrixT_j*r0*alp_prof;

dr_TS.dr_salt = dr_salt;
dr_TS.dr_temp = dr_temp;

spice = sign(new_matrixT_j).*sqrt((dr_salt.^2)+(dr_temp.^2)); hold on
