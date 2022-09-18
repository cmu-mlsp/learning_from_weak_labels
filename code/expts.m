
% Load Data
train = load('train.mat');
test = load('test.mat');
val = load('val.mat');

% Flatten all bags and labels
% ========= Train data =============================
pos = []; labelpos = [];
numposbags = size(train.pos_bags,1);
for i = 1:numposbags
    bagpossize(i) = size(train.pos_bags,2);
    for j = 1:size(train.pos_bags,2)
       pos = [pos; squeeze(train.pos_bags(i,j,:))'];
       labelpos = [labelpos; train.pos_bags_labels(i,j)];
    end
end
pos = cast(pos,"double");
poslabel = (labelpos == 2);

neg = []; labelneg = [];
numnegbags = size(train.pos_bags,1);
for i = 1:numnegbags
    bagnegsize(i) = size(train.neg_bags,2);
    for j = 1:size(train.neg_bags,2)
       neg = [neg; squeeze(train.neg_bags(i,j,:))'];
       labelneg = [labelneg; train.neg_bags_labels(i,j)];
    end
end
neg = cast(neg,"double");
neglabel = (labelneg == 2);
truelabel = [poslabel;neglabel];
baglabels = [ones(size(train.pos_bags,1),1); zeros(size(train.neg_bags,1),1)];

baginstancecounts = [bagpossize(:)' bagnegsize(:)'];

% ======== Test data===============
postest = []; labelpostest = [];
numposbagstest = size(test.pos_bags,1);
for i = 1:numposbagstest
    bagpossizetest(i) = size(test.pos_bags,2);
    for j = 1:size(test.pos_bags,2)
       postest = [postest; squeeze(test.pos_bags(i,j,:))'];
       labelpostest = [labelpostest; test.pos_bags_labels(i,j)];
    end
end

postest = cast(postest,"double");
poslabeltest = (labelpostest == 2);

negtest = []; labelnegtest = [];
numnegbagstest = size(test.neg_bags,1);
for i = 1:numnegbagstest
    bagnegsizetest(i) = size(test.neg_bags,2);
    for j = 1:size(test.neg_bags,2)
       negtest = [negtest; squeeze(test.neg_bags(i,j,:))'];
       labelnegtest = [labelnegtest; test.neg_bags_labels(i,j)];
    end
end
negtest = cast(negtest,"double");
neglabeltest = (labelnegtest == 2);

alltest = [postest; negtest];
truelabeltest = [poslabeltest;neglabeltest];
baglabelstest = [ones(size(test.pos_bags,1),1); zeros(size(test.neg_bags,1),1)];

baginstancecountstest = [bagpossizetest(:)' bagnegsizetest(:)'];


% ======== Val data===============

posval = []; labelposval = [];
numposbagsval = size(val.pos_bags,1);
for i = 1:numposbagsval
    bagpossizeval(i) = size(val.pos_bags,2);
    for j = 1:size(val.pos_bags,2)
       posval = [posval; squeeze(val.pos_bags(i,j,:))'];
       labelposval = [labelposval; val.pos_bags_labels(i,j)];
    end
end

posval = cast(posval,"double");
poslabelval = (labelposval == 2);

negval = []; labelnegval = [];
numnegbagsval = size(val.neg_bags,1);
for i = 1:numnegbagsval
    bagnegsizeval(i) = size(val.neg_bags,2);
    for j = 1:size(val.neg_bags,2)
       negval = [negval; squeeze(val.neg_bags(i,j,:))'];
       labelnegval = [labelnegval; val.neg_bags_labels(i,j)];
    end
end
negval = cast(negval,"double");
neglabelval = (labelnegval == 2);

allval = [posval; negval];
truelabelval = [poslabelval;neglabelval];
baglabelsval = [ones(size(val.pos_bags,1),1); zeros(size(val.neg_bags,1),1)];

baginstancecountsval = [bagpossizeval(:)' bagnegsizeval(:)'];


%==========================================================================

% Expt 1

% Train a supervised SVM
disp(['===================== EXPT 1, FULLYSUPERVISED ===================']);
data = [pos;neg];
labels = [poslabel;neglabel];
[inststrongout,bagstrongout,inststrongscr, bagstrongscr, strongacc,strongbagacc,valscr,valacc,valbagacc] = svmtest(data,labels,alltest,truelabeltest,baglabelstest,baginstancecountstest,allval,truelabelval,baglabelsval,baginstancecountsval);

disp(['Strong label training: Instance acc ' num2str(strongacc) '; Bag acc ' num2str(strongbagacc)]);
disp(['                  Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);

%==========================================================================

% Expt 2

% Label all train instances with bag label
disp(['      ']);
disp(['      ']);
disp(['===================== EXPT 2, FLAT LABELLING ===================']);
data = [pos;neg];
labels = [ones(size(poslabel)); zeros(size(neglabel))];
[instflatout,bagflatout,instflatscr, bagflatscr, flatacc,flatbagacc,valscr, valacc, valbagacc] = svmtest(data,labels,alltest,truelabeltest,baglabelstest,baginstancecounts,allval,truelabelval,baglabelsval,baginstancecountsval);
disp(['Flat label training: Instance acc ' num2str(flatacc) '; Bag acc ' num2str(flatbagacc)]);
disp(['                Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);

%==========================================================================


% Expt 3

% Label all train instances with bag label, then iteratively prune out the
% lowest scoring instances
disp(['      ']);
disp(['      ']);
disp(['===================== EXPT 3, TRIM FROM BOTTOM ===================']);

data = [pos;neg];
labelstrim = [ones(size(poslabel)); zeros(size(neglabel))];
testontrain = 1;

bestval = inf;
bestvalid = -1;

for iter = 1:15
    [insttrimout,bagtrimout,insttrimscr, bagtrimscr, trimacc,trimbagacc,valscr,valacc, valbagacc,trainpred,trainscr] = svmtest(data,labelstrim,alltest,truelabeltest,baglabelstest,baginstancecountstest,allval,truelabelval,baglabelsval,baginstancecountsval,testontrain);
    disp(['Label trimming, Iteration ' num2str(iter) ': Instance acc ' num2str(trimacc) '; Bag acc ' num2str(trimbagacc)]);
    disp(['                         Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);

    ne = 0;
    for b = 1:numposbags
        nb = ne+1;
        ne = ne + bagpossize(b);

        baglabel = labelstrim(nb:ne);
        bagscr = trainscr(nb:ne);

        bagscr(find(baglabel==0)) = inf; % Flag out already negated instances in bag
        [bagminscr,bagminidx] = min(bagscr);
        labelstrim(nb+bagminidx-1) = 0;  % Set label of most negative instance to 0
    end
end


%==========================================================================

% Expt 4

% Label all train instances with bag label, then iteratively retain the
% higest scoring instances
disp(['      ']);
disp(['      ']);
disp(['===================== EXPT 4, TAG FROM TOP ===================']);

data = [pos;neg];
labelspeak = [ones(size(poslabel)); zeros(size(neglabel))];
testontrain = 1;

for iter = 1:15
    [instpeakout,bagpeakout,instpeakscr, bagpeakscr, peakacc,peakbagacc,valscr, valacc, valbagacc,trainpred,trainscr] = svmtest(data,labelspeak,alltest,truelabeltest,baglabelstest,baginstancecountstest,allval,truelabelval,baglabelsval,baginstancecountsval,testontrain);
    disp(['Peak tagging, Iteration ' num2str(iter) ': Instance acc ' num2str(peakacc) '; Bag acc ' num2str(peakbagacc)]);
    disp(['                        Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);


    ne = 0;
    for b = 1:numposbags
        nb = ne+1;
        ne = ne + bagpossize(b);

        baglabel = labelspeak(nb:ne);
        bagscr = trainscr(nb:ne);

        [bagmaxscr,bagmaxidx] = max(bagscr);
        labelspeak(nb:ne) = trainpred(nb:ne);  % Use the model prediction
        labelspeak(nb+bagmaxidx-1) = 1;
    end
end

%==========================================================================

% Expt 5

% Label all train instances with bag label, then iteratively retain the
% K higest scoring instances, but this time also tag the rest of the
% instances in the positive bag as negative
disp(['      ']);
disp(['      ']);
disp(['================== EXPT 5, TAG TOP K FROM TOP ==================']);

data = [pos;neg];
labelspeak = [ones(size(poslabel)); zeros(size(neglabel))];
testontrain = 1;

K = 2;

for iter = 1:15
    [instpeakout,bagpeakout,instpeakscr, bagpeakscr, peakacc,peakbagacc,valscr, valacc, valbagacc,trainpred,trainscr] = svmtest(data,labelspeak,alltest,truelabeltest,baglabelstest,baginstancecountstest,allval,truelabelval,baglabelsval,baginstancecountsval,testontrain);
    disp(['Peak tagging, Iteration ' num2str(iter) ': Instance acc ' num2str(peakacc) '; Bag acc ' num2str(peakbagacc)]);
    disp(['                        Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);


    ne = 0;
    for b = 1:numposbags
        nb = ne+1;
        ne = ne + bagpossize(b);

        baglabel = labelspeak(nb:ne);
        bagscr = trainscr(nb:ne);

        [bagmaxscr,bagmaxidx] = maxk(bagscr,K);
        labelspeak(nb:ne) = 0;  % Default 0 class
        labelspeak(nb+bagmaxidx-1) = trainpred(bagmaxidx); % For these retain the classifier's output
        labelspeak(nb+bagmaxidx(1)-1) = 1;  % At least the top 1 is 1
    end
end

%==========================================================================

% Expt 6

% Label all train instances with bag label, then iteratively retain the
% positive-bag instances that have a score higher than the highest -ve bag instance,
% while ensuring at least one positive

disp(['      ']);
disp(['      ']);
disp(['================== EXPT 6, TAG POSITIVE-BAG INSTANCES THAT EXCEED THRESHOLD ==================']);

data = [pos;neg];
labelspeak = [ones(size(poslabel)); zeros(size(neglabel))];
testontrain = 1;

for iter = 1:15
    [instpeakout,bagpeakout,instpeakscr, bagpeakscr, peakacc,peakbagacc,valscr, valacc, valbagacc,trainpred,trainscr] = svmtest(data,labelspeak,alltest,truelabeltest,baglabelstest,baginstancecountstest,allval,truelabelval,baglabelsval,baginstancecountsval,testontrain);
    disp(['Peak tagging, Iteration ' num2str(iter) ': Instance acc ' num2str(peakacc) '; Bag acc ' num2str(peakbagacc)]);
    disp(['                        Val: Instance acc ' num2str(valacc) '; Bag acc ' num2str(valbagacc)]);

    % Get cutoff
    nb = sum(bagpossize);
    theta = max(trainscr(1:nb));
    theta = 0.9999*theta;

    % All positive-bag instances with score > cutoff
    labelspeak = zeros(size(labelspeak));  % Default
    trainscrpos = trainscr(1:nb);
    goodids = find(trainscrpos > theta); size(goodids)
    labelspeak(goodids) = 1; %trainpred(goodids);

    % Pick at least 1 pos from each bag
    ne = 0;
    for b = 1:numposbags
        nb = ne+1;
        ne = ne + bagpossize(b);

        bagscr = trainscr(nb:ne);
        [bagmaxscr,bagmaxidx] = max(bagscr);

        labelspeak(nb+bagmaxidx(1)-1) = 1;  % At least the top 1 is 1
    end
end


%==========================================================================
% Expt 6

% mi-SVM