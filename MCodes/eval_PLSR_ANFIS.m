function [modYP,YP,ErrP]=eval_PLSR_ANFIS(X,PLSR_ANFIS_Result)
%% Checking inputs
validateattributes(X,{'double'},{'2d'});
nIndependentVariables=size(X,2);

if (~isfield(PLSR_ANFIS_Result,'YBETA'))
  error('YBETA cannot be found in PLSR_ANFIS_Result');
else
  validateattributes(PLSR_ANFIS_Result.YBETA,{'double'},{'size',[nIndependentVariables+1, 1]});
end
if (~isfield(PLSR_ANFIS_Result,'ErrBetta'))
  error('ErrBetta cannot be found in PLSR_ANFIS_Result');
else
  validateattributes(PLSR_ANFIS_Result.ErrBetta,{'double'},{'size',[nIndependentVariables+1, 1]});
end
if (~isfield(PLSR_ANFIS_Result,'FISMat'))
  error('FISMat cannot be found in PLSR_ANFIS_Result');
end

%% Phase 1: Part 1: Initial Estimate for Y
YP = [ones(size(X,1),1) X]*PLSR_ANFIS_Result.YBETA;

%% Phase 1: Part 2: Estimating Error
ErrP = [ones(size(X,1),1) X]*PLSR_ANFIS_Result.ErrBetta;

%% Phase 2: Getting final estimate for Y using initial estimate for Y and the estimated Error
modYP=evalfis([YP,ErrP],PLSR_ANFIS_Result.FISMat);

end
