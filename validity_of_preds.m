%This is valid for datasets with only positive values.  I'll calculate the
%validity of data with both negative and positive values separately.
clear;
%%%%%%%%%avg wage%%%%%%%%%%%%%
[ aw_mse, aw_rmse, aw_r_squared, aw_mae, aw_target_range] = prediction_validity('avg_wage.csv', 16, 32, 0.00325325);

%%%%%%%%%disposable income%%%%%%%%%%%%%
[ di_mse, di_rmse, di_r_squared, di_mae, di_target_range] = prediction_validity('disposable_income.csv', 16, 32, 0.01889175);

%%%%%%%%%years of education%%%%%%%%%%%%%
[ed_mse,ed_rmse, ed_r_squared, ed_mae, ed_target_range] = prediction_validity('education_data.csv', 64, 128, 0.18289);

%%%%%%%%%gdp%%%%%%%%%%%%%
[gdp_mse, gdp_rmse, gdp_r_squared, gdp_mae, gdp_target_range] = prediction_validity('gdp.csv', 128, 256, 0.0097);

%%%%%%%%%gender equality%%%%%%%%%%%%%
[gen_mse, gen_rmse, gen_r_squared, gen_mae, gen_target_range]  = prediction_validity('gender_equality_longitudinal.csv', 64, 128, 0.0998);

%%%%%%%%%hate crimes%%%%%%%%%%%%%
[hate_mse, hate_rmse, hate_r_squared, hate_mae, hate_target_range] = prediction_validity('hate_crimes_longitudinal.csv', 32, 64, 0.120486);

%%%%%%%%%housing cost%%%%%%%%%%%%%
[house_mse, house_rmse, house_r_squared, house_mae, house_target_range] = prediction_validity('housing_cost.csv',256,512,0.000001);

%%%%%%%%%premature mortality %%%%%%%%%%%%%
[premature_mort_mse,premature_mort_rmse, premature_mort_rsquared, premature_mort_mae, premature_mort_target] = prediction_validity('premature_mortality_long.csv',32,64,0.0035);

%%%%%%%%%refugees %%%%%%%%%%%%%
[refugee_mse, refugee_rmse, refugee_r2, refugee_mae, refugee_target_range] = prediction_validity('refugees_clean.csv',16,32,0.0036518);

%%%%%%%%%%unemployment%%%%%%%%%%%%
[unemploy_mse, unemploy_rmse, unemploy_r2, unemploy_mae, unemploy_target_range] = prediction_validity('unemployment.csv',16,32,0.0036518);

%%%%%%%%%%violent crime%%%%%%%%%%%%
[viol_mse, viol_rmse, viol_r2, viol_mae, viol_target_range] = prediction_validity('violent_crime_clean.csv',16,32,0.00135);

%%%%%%%%%%women in parliament%%%%%%%%%%%%
[w_mse, w_rmse, w_r_squared, w_mae, w_target_range] = prediction_validity('women_parliament.csv',128,256,.00006755125);

%%%%%%%%%%world happiness%%%%%%%%%%%%
[happy_rmse, happy_r2,happy_mae, happy_target_range] = prediction_validity('world_happiness_longitudinal_validity.csv',128,256,0.00014838125);

[happy_rmse, happy_r2,happy_mae, happy_target_range] = prediction_validity('world_happ_adjusted.csv',128,256,0.0000001);


%%%%%%%%%%datasets with genuine negatives%%%%%%%%%%%%
%%%%%%%%%%inflation%%%%%%%%%%%%
[inf_mse, inf_rmse, inf_r_squared, inf_mae, inf_target_range] = prediction_validity_negatives('inflation_cleaner.csv',128, 256,0.034426125);

%%%%%%%%%%%%%%%%%%%%%%
[surf_mse, surf_rmse, surf_r_squared, surf_mae, surf_target_range] = prediction_validity_negatives('surface_temp.csv',128, 256,0.01712);


%%%%%%%%%%%%New Model%%%%%%%%%%%%%%%%%%%
[hate_mse, hate_rmse, hate_r_squared, hate_mae, hate_target_range] = prediction_validity_new('hate_crimes_update130317.csv',4,8, .2);

