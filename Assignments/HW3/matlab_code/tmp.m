fprintf("Classification using mnrfit\n")
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