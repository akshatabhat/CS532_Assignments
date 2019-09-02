% Solution to Problem 6
data = readtable('bikeshare.csv');
Y = data.cnt;
X = [ones(size(Y)), data.holiday, data.temp, data.hum, data.windspeed];

%Ordinary Least Squares 
disp('6a)') %Ordinary Least Sqaures
B_ols = inv(X'*X)*X'*Y;
disp('B_olsf for [intercept, holiday, temperature, humidity, windspeed]')
disp(B_ols)
disp("Coefficients B_2, B_4 and B_5 for attributes holiday, humidity and windspeed are negative, which indicate that") 
disp("usage of bike is low when there is a holiday, when humidity and windspeed is high. We also see that maginitude ")
disp("of B_3 is high, indicating that temperature has a high positive correlation with bike usage - probably people ")
disp("like to bike during summers. Hence coefficients seem convincing.")
disp(" ")

disp("6b)")
%Ridge Regression
s = size(X);
I = eye(s(2));

disp('Ridge Regression with lambda = 0.1')
lambda = 0.1;
B_ridge = inv((X'*X) + (lambda * I))*X'*Y

disp('Ridge Regression with lambda = 1')
lambda = 1;
B_ridge = inv((X'*X) + (lambda * I))*X'*Y

disp('Ridge Regression with lambda = 10')
lambda = 10;
B_ridge = inv((X'*X) + (lambda * I))*X'*Y

disp('Ridge Regression with lambda = 100')
lambda = 100;
B_ridge = inv((X'*X) + (lambda * I))*X'*Y

disp('Ridge Regression with lambda = 1000')
lambda = 1000;
B_ridge = inv((X'*X) + (lambda * I))*X'*Y
disp(" ")
disp(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ")