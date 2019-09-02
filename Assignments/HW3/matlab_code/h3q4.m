data = readtable('wine.csv');
data = data.Variables;
y = data(1:end-50,1);
X = data(1:end-50,2:end);

y_val = data(end-49:end,1);
X_val = data(end-49:end,2:end);

%a One-vs-One Classification
fprintf("4a) One-vs-One Classification \n")
exclude_class = [1 ,2, 3];
y_pred = [];

for i=1:length(exclude_class)
    y_train = y(y~=exclude_class(i));
    X_train = X(y~=exclude_class(i),:);
    [class,err,POSTERIOR,logp,B_lda] = classify(X_val, X_train, y_train);
    y_pred = [y_pred class];
end

predictions = zeros(size(y_val));
ambiguous_classification = 0;correct_classification = 0;incorrect_classification = 0;
for i=1:length(X_val)
    if y_pred(i,1) ~= y_pred(i,2) && y_pred(i,1) ~= y_pred(i,3) && y_pred(i,2) ~= y_pred(i,3)
        ambiguous_classification = ambiguous_classification + 1;
    else
        predictions(i) = mode(y_pred(i,:));
    end
end

acc = sum(predictions==y_val)/length(y_val);
fprintf("Ambigous classification = %f \n", ambiguous_classification/length(X_val));
fprintf("Correct classification = %f \n", acc);
fprintf("Incorrect classification = %f \n", 1 - acc -ambiguous_classification/length(X_val) );
disp(" ")

%a One-vs-All Classification
fprintf("4b) One-vs-All Classification \n")
y = data(1:end-50,1);
X = data(1:end-50,2:end);

y_val = data(end-49:end,1);
X_val = data(end-49:end,2:end);

y_train(y==1)=1;
y_train(y~=1)=0;
[class1,~,~,~,coeff] = classify(X_val, X, y_train);
w1 = [coeff(1,2).const ; coeff(1,2).linear];

y_train(y==2)=1;
y_train(y~=2)=0;
[class2,~,~,~,~] = classify(X_val, X, y_train);

y_pred = [class1 class2];
predictions = zeros(size(y_val));
for i=1:length(X_val)
    if y_pred(i,1) == 1 &&  y_pred(i,2) == 1
        ambiguous_classification = ambiguous_classification + 1;
    elseif y_pred(i,1) == 1
        predictions(i) = 1;
    elseif y_pred(i,2) == 1
        predictions(i) = 2;
    else
        predictions(i) = 3;
    end
end
acc = sum(predictions==y_val)/length(predictions);

fprintf("Ambigous classification = %f \n", ambiguous_classification/length(y_val));
fprintf("Correct classification = %f \n", acc);
fprintf("Incorrect classification = %f \n", 1-acc-ambiguous_classification/length(y_val));
%}
disp(" ")
%c mnrfit
clear all;
fprintf("4c) Classification using mnrfit\n")
data = readtable('wine.csv');
data = data.Variables;
y = data(1:end-50,1);
X = data(1:end-50,2:end);

y_val = data(end-49:end,1);
X_val = data(end-49:end,2:end);

B = mnrfit(X,y);
y_pred = mnrval(B,X_val);
[~,I]=max(y_pred, [], 2);
acc = sum(y_val ==I);
acc = acc/length(y_pred);
fprintf("Correct classification = %f \n", acc);
fprintf("Incorrect classification = %f \n", 1-acc);
disp(" ")

%1d
fprintf("4d) Based on the validation accuracy, I would choose One-vs-All or One-vs-One Classification Method \n");