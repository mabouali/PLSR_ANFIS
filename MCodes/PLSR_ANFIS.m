function [result, detail]=PLSR_ANFIS(X,Y,inputMFTypes,mfNum)
%% Initializations
kFold=10;
outputMFTypes = {'linear' 'constant'};
possibleMFTypes = {'trimf', 'trapmf','gbellmf', 'gaussmf', 'gauss2mf'};
nParamNeeded = [3,4,3,2,4];
nIterations = 500;
nParam = 2;

%% checking inputs
nSamples=size(X,1);
validateattributes(X,{'double'},{'2d','size',[nSamples, NaN]});
validateattributes(Y,{'double'},{'column','size',[nSamples, 1]});
validateattributes(mfNum,{'numeric'},{'numel',2});

if (any(isnan(X(:))) || any(isnan(Y)))
  error('X and Y should not have any NaN');
end

if (any(mod(mfNum,1)~=0) || any(mfNum<1))
  error('mfNum must contain only positive non-zero integers.');
end
if (mfNum(1)>mfNum(2))
  error('mfNum(1) must be smaller than or equal to mfNum(2)')
end

if (isempty(inputMFTypes))
  inputMFTypes=possibleMFTypes;
else
  mask=arrayfun(@(c) any(strcmpi(inputMFTypes,possibleMFTypes{c})),1:numel(possibleMFTypes));
  inputMFTypes=possibleMFTypes(mask);
  nParamNeeded=nParamNeeded(mask);
  if (isempty(inputMFTypes))
    error('Non of the requested membership functions are supported.');
  end
end

%% Phase 1: Part 1: Establishing a PLSR model to estimate Y.
disp('Phase 1: Part 1:')
% generating groups for 10-fold cross validation
cv=cvpartition(nSamples,'k',kFold);
% load cv
  
maxNComp=min( min(cv.TrainSize)-1, ...
              size(X,2) );
MSE=zeros(kFold,kFold,maxNComp);
Correlation=zeros(kFold,kFold,maxNComp);
BETA=cell(kFold,maxNComp);

for trainingNo=1:kFold
  fprintf('- - Training #%d ...\n',trainingNo);
  XTraining = X(cv.training(trainingNo),:);
  YTraining = Y(cv.training(trainingNo));
  for nComp=1:maxNComp
    [~,~,~,~, BETA{trainingNo,nComp}]= plsregress(XTraining,YTraining,nComp);
    for testNo=1:kFold
      XTest = X(cv.test(testNo),:);
      YTest = Y(cv.test(testNo));
      YP=[ones(size(XTest,1),1) XTest]*BETA{trainingNo,nComp};
      MSE(testNo,trainingNo,nComp)=mean((YP-YTest).^2);
      Correlation(testNo,trainingNo,nComp)=corr(YTest,YP);
    end
  end
end

% findinf the best model (trainingNo,nComp)
cvMSE=arrayfun(@(nComp) mean(diag(MSE(:,:,nComp))),1:maxNComp);
meanMSE=cell2mat(arrayfun(@(nComp) mean(MSE(:,:,nComp))',1:maxNComp,'UniformOutput',false));
combinedError=  (meanMSE-min(meanMSE(:)))./(max(meanMSE(:))-min(meanMSE(:))) ...
              + repmat((cvMSE-min(cvMSE(:)))./(max(cvMSE(:))-min(cvMSE(:))),kFold,1);
meanCorrelation=cell2mat(arrayfun(@(nComp) mean(Correlation(:,:,nComp))',1:maxNComp,'UniformOutput',false));
trainingNo=repmat((1:kFold)',1,maxNComp);
nComp=repmat(1:maxNComp,kFold,1);
tmpSortedScores=sortrows([combinedError(:) meanMSE(:) meanCorrelation(:) trainingNo(:) nComp(:)],[1,2,-3]);
bestTrainingNo=tmpSortedScores(1,4);
bestNComp=tmpSortedScores(1,5);

% Storing the full detailed Results
detail.X=X;
detail.Y=Y;
detail.cvPartitions=cv;
detail.PLSR_Y.BETA=BETA;
detail.PLSR_Y.bestBETA=BETA{bestTrainingNo,bestNComp};
detail.PLSR_Y.maxNComp=maxNComp;
detail.PLSR_Y.RMSE=sqrt(MSE);
detail.PLSR_Y.correlation=Correlation;
detail.PLSR_Y.cvMSE=cvMSE;
detail.PLSR_Y.bestMSE=meanMSE(bestTrainingNo,bestNComp);
detail.PLSR_Y.bestCorrelation=meanCorrelation(bestTrainingNo,bestNComp);
detail.PLSR_Y.bestTrainingNo=bestTrainingNo;
detail.PLSR_Y.bestNComp=bestNComp;

% storing the results
result.YBETA=detail.PLSR_Y.bestBETA;

%% Phase 1: Part 2: Establishing a PLSR model to estimate the error of the previous model.
disp('Phase 1: Part 2:')
% Forming the error vector
Err=Y-[ones(nSamples,1) X]*result.YBETA;
detail.Err=Err;

% initializing/performing PLSR
MSE=zeros(kFold,kFold,maxNComp);
Correlation=zeros(kFold,kFold,maxNComp);
BETA=cell(kFold,maxNComp);
for trainingNo=1:kFold
  fprintf('- - Training #%d ...\n',trainingNo);
  XTrain = X(cv.training(trainingNo),:);
  ErrTrain = Err(cv.training(trainingNo));
  for nComp=1:maxNComp
    [~,~,~,~, BETA{trainingNo,nComp}]= plsregress(XTrain,ErrTrain,nComp);
    for testNo=1:kFold
      XTest =  X(cv.test(testNo),:); 
      ErrTest = Err(cv.test(testNo));
      ErrP=[ones(size(XTest,1),1) XTest]*BETA{trainingNo,nComp};
      MSE(testNo,trainingNo,nComp)=mean((ErrP-ErrTest).^2);
      Correlation(testNo,trainingNo,nComp)=corr(ErrTest,ErrP);
    end
  end
end

% findinf the best model (trainingNo,nComp)
cvMSE=arrayfun(@(nComp) mean(diag(MSE(:,:,nComp))),1:maxNComp);
meanMSE=cell2mat(arrayfun(@(nComp) mean(MSE(:,:,nComp))',1:maxNComp,'UniformOutput',false));
meanCorrelation=cell2mat(arrayfun(@(nComp) mean(Correlation(:,:,nComp))',1:maxNComp,'UniformOutput',false));
trainingNo=repmat((1:kFold)',1,maxNComp);
nComp=repmat(1:maxNComp,kFold,1);
tmpSortedScores=sortrows([meanMSE(:) meanCorrelation(:) trainingNo(:) nComp(:)],[1,-2]);
bestTrainingNo=tmpSortedScores(1,3);
bestNComp=tmpSortedScores(1,4);

% Writing the results
detail.PLSR_Err.BETA=BETA;
detail.PLSR_Err.bestBETA=BETA{bestTrainingNo,bestNComp};
detail.PLSR_Err.maxNComp=maxNComp;
detail.PLSR_Err.RMSE=sqrt(MSE);
detail.PLSR_Err.correlation=Correlation;
detail.PLSR_Err.cvMSE=cvMSE;
detail.PLSR_Err.bestMSE=meanMSE(bestTrainingNo,bestNComp);
detail.PLSR_Err.bestCorrelation=meanCorrelation(bestTrainingNo,bestNComp);
detail.PLSR_Err.bestTrainingNo=bestTrainingNo;
detail.PLSR_Err.bestNComp=bestNComp;
  
result.ErrBetta=detail.PLSR_Err.bestBETA;

%% Phase 2: Establishing an ANFIS model
disp('Phase 2:')
% Preparing input data
data=[[ones(nSamples,1) X]*result.YBETA, ... %YP
      [ones(size(X,1),1) X]*result.ErrBetta, ... %ErrP
      Y];
% an artifical data with no Error;
noErrY=linspace(min(Y),max(Y),1000)';
noErrData=[ noErrY, zeros(numel(noErrY),1), noErrY ];  

% creating nMFCombs
minMFNum = mfNum(1);
maxMFNum = mfNum(2);
[tmp1, tmp2]=meshgrid(minMFNum:maxMFNum);
MFNumCombs=[tmp1(:),tmp2(:)];
nMFNumCombs=size(MFNumCombs,1);

totalPossibilities=   numel(inputMFTypes) ...
                    * numel(outputMFTypes) ...
                    * nMFNumCombs;
fprintf('- Checking %d ANFIS model Posibilities.\n',totalPossibilities);

minNRecordTraining=min(cv.TrainSize);
counter=0;
for possibilityID=1:totalPossibilities
  [MFNumCombID,outputMFTypesID,inputMFTypesID]=ind2sub([nMFNumCombs,numel(outputMFTypes),numel(inputMFTypes)],possibilityID);

  inputMFType = inputMFTypes{inputMFTypesID};
  outputMFType= outputMFTypes{outputMFTypesID};
  MFNums = MFNumCombs(MFNumCombID,:); 

  % Checking if there are enough input records
  minNDataNeeded=  nParamNeeded(inputMFTypesID)*sum(MFNums) ... % needed by nonlinear part
                 + prod(MFNums)*nParam*strcmpi(outputMFType,'linear') ... % needed by ANFIS if output is linear
                 + prod(MFNums)*strcmpi(outputMFType,'constant');  % needed by ANFIS if output is linear
  if (minNRecordTraining<minNDataNeeded)
    % Not enough inputs skipping.
    fprintf('%d,%s,%s',possibilityID,inputMFType,outputMFType);
    fprintf(',NMF')
    fprintf(',%d',MFNums);
    fprintf(' - NOT ENOUGH INPUTS. SKIPPING.\n');
    continue
  end

  fittedFIS=cell(kFold,1);
  MSE=zeros(kFold,kFold);
  correlation=zeros(kFold,kFold);
  try
    for trainingNo=1:kFold
      initFIS = genfis1([data(cv.training(trainingNo),:);noErrData],MFNums,inputMFType,outputMFType); 
      [~,~,~,fittedFIS{trainingNo},~] = anfis([data(cv.training(trainingNo),:);noErrData],initFIS,nIterations,[0 0 0 0],data(cv.test(trainingNo),:));
      for testNo=1:kFold
        YP = evalfis(data(cv.test(testNo),1:end-1),fittedFIS{trainingNo});
        MSE(testNo,trainingNo)=mean( (YP-data(cv.test(testNo),end)).^2 );
        correlation(testNo,trainingNo)=corr(YP,data(cv.test(testNo),end));
      end
    end
  catch
    % some error has occured. Just skipping.
    fprintf('%d,%s,%s',possibilityID,inputMFType,outputMFType);
    fprintf(',NMF')
    fprintf(',%d',MFNums);
    fprintf(' - SOME ERROR OCCURED. SKIPPING.\n');
    continue
  end

  meanMSE=mean(MSE);
  meanCorrelation=mean(correlation);
  sortedScores=sortrows([meanMSE(:) meanCorrelation(:) (1:kFold)'],[1,-2]);
  theChosenOne= sortedScores(1,3);

  counter=counter+1;
  detail.ANFIS.inputMFType(counter,1)={ inputMFType };
  detail.ANFIS.outputMFType(counter,1)={ outputMFType };
  detail.ANFIS.MFNums(counter,1)={ MFNums };
  detail.ANFIS.RMSE(counter,1)={ sqrt(MSE) };
  detail.ANFIS.correlation(counter,1)={ correlation };
  detail.ANFIS.cvMSE(counter,1)= mean(diag(MSE));
  detail.ANFIS.bestMSE(counter,1)=meanMSE(theChosenOne);
  detail.ANFIS.bestCorrelation(counter,1)=meanCorrelation(theChosenOne);
  detail.ANFIS.fittedFIS(counter,1)=fittedFIS{theChosenOne};

  fprintf('%d,%s,%s',possibilityID,inputMFType,outputMFType);
  fprintf(',NMF')
  fprintf(',%d',MFNums);
  fprintf(' - %f, %f\n',meanMSE(theChosenOne), mean(diag(MSE)));
end

%choosing the best ANFIS
finalChosenOne=find(detail.ANFIS.cvMSE==min(detail.ANFIS.cvMSE));
detail.ANFIS.BestFittedFIS=detail.ANFIS.fittedFIS(finalChosenOne);
result.FISMat=detail.ANFIS.fittedFIS(finalChosenOne);
end
















































































