% Solution to HW1 - Problem 5 
X = mvnrnd([0;0;0], [2,0,0;0,4,0;0,0,10],100); % Generating X for 100 samples 
E = normrnd(0, sqrt(12), [100,1]); % Generating error for 100 samples
B = [0;1;-1]; % Regression coefficients 
Y = X*B+E;

% (5a)
B_ols = inv(X'*X)*X'*Y; % Calculating regression parameters using ordinary least squares
disp("5a)")
disp("B_ols : ")
disp(B_ols)
z = 1.645; % For constructing 90% confidence interval, z = 1.645
sd = sqrt(12);
%sqrt(sum((Y - X*B_ols).^2)/100); % Calcuating standard deviation from the sample 
v = diag(inv(X'*X)) ;
disp("90% Confidence Interval for B (B_1, B_2 and B_3): ")
CI = [B_ols - (z * (sd * sqrt(v))),  B_ols + (z * (sd * sqrt(v)))];
disp(CI)
disp("Yes, the interval convers B_1");
disp(" ")
% (5b)
disp("5b)")
count = 0;
for i=1:500
    E = normrnd(0, sqrt(12), [100,1]);
    Y = X*B+E;
    B_ols = inv(X'*X)*X'*Y;
    sd = sqrt(12); % sqrt(sum((Y - X*B_ols).^2)/100); % Calcuating standard deviation from the sample 
    v = diag(inv(X'*X));
    CI = [B_ols - (z * (sd * sqrt(v))),  B_ols + (z * (sd * sqrt(v)))];
    if B(1) > CI(1,1) && B(1) < CI(1,2)
        count = count + 1;
    end   
end
disp("Percentage of time confidence interval covers B_1 is")
disp((count*100)/500)

% (5c)
disp("5c)")
E = -6 + (6+6)*rand(100,1);
B_ols = inv(X'*X)*X'*Y; % Calculating regression parameters using ordinary least squares
z = 1.645; % For constructing 90% confidence interval, z = 1.645
sd = sqrt(12); % sqrt(sum((Y - X*B_ols).^2)/100); % Calcuating standard deviation from the sample 
v = diag(inv(X'*X)); 
CI = [B_ols - (z * (sd * sqrt(v))),  B_ols + (z * (sd * sqrt(v)))];
count = 0;
for i=1:500
    E = -6 + (6+6)*rand(100,1);
    Y = X*B+E;
    B_ols = inv(X'*X)*X'*Y;
    sd = sqrt(12); % sqrt(sum((Y - X*B_ols).^2)/100); % Calcuating standard deviation from the sample 
    v = diag(inv(X'*X)) ;
    CI = [B_ols - (z * (sd * sqrt(v))),  B_ols + (z * (sd * sqrt(v)))];
    if B(1) > CI(1,1) && B(1) < CI(1,2)
        count = count + 1;
    end   
end
disp("Percentage of time confidence interval covers B_1 is")
disp((count*100)/500)