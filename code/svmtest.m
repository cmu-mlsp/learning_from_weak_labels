function [instancepred,bagpred,instancescr,bagscr,instacc,bagacc,valscr, valacc,valbagacc,traininstancepred,traininstancescr] = svmtest(data,labels,alltest,labeltest,baglabelstest,baginstancecountstest,allval,labelval,baglabelsval,baginstancecountsval,testontrain)

if (nargin < 11)
    testontrain = 0;
end

%svm = fitclinear(data, labels);
svm = fitcsvm(data,labels,'Standardize',true);

% Only if we must test on the train data
if (testontrain)
    [traininstancepred,traininstancescr2] = predict(svm, data);
    traininstancescr = exp(traininstancescr2(:,1))./(exp(traininstancescr2(:,1)) + exp(traininstancescr2(:,2)));
end

% Val
[valpred, valscr2] = predict(svm,allval);
valscr = exp(valscr2(:,1))./(exp(valscr2(:,1))+exp(valscr2(:,2)));
valacc = sum(valpred==labelval)*100/length(allval);
[valbagpred, valbagscr, valbagacc] = computebagacc(valpred,valscr,baglabelsval,baginstancecountsval);

% Classify
[instancepred, instancescr2] = predict(svm, alltest);
instancescr = exp(instancescr2(:,1))./(exp(instancescr2(:,1)) + exp(instancescr2(:,2)));
% Instance-level classification accuracy
instacc = sum(instancepred==labeltest)*100/length(alltest);
% Bag-level accuracy
[bagpred, bagscr, bagacc] = computebagacc(instancepred, instancescr, baglabelstest, baginstancecountstest);