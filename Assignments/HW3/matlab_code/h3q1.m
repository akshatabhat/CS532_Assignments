% Homework 3 - Question 1
clear all;
disp("1a) glmfit for Wisconsin Breast Cancer Dataset")
data = readtable('bc_wisc.csv');
data = data.Variables;
y = data(:,2);
X = data(:,[10, 23, 24, 30, 31]);

B_glm = glmfit(X,y,'binomial');
disp("Coeffecients of B");
disp(B_glm);
X_test = [ones(size(y)), X];
y_pred = X_test*B_glm>=0;
num_correct_predictions = sum(y_pred==y);
fprintf("Proportion of correction predictions = %f \n",num_correct_predictions/length(y));
disp(" ")

%1b

fprintf("1b) Logistic Regression using Gradient Descent iterates\n");
X = [ones(size(X,1),1), X];
B_old = zeros(6,1);

step_size = 0.00012;
num_iterations = 10000000;
err = zeros(num_iterations,1);

for i=1:num_iterations
    u = 1./(exp(X*B_old)+1);
    delta = (1-y).*X - u.*X;
    grad = sum(delta,1)';
    B_new = B_old - step_size*grad;
    err(i) = norm(B_new-B_glm);
    B_old = B_new;
end
fprintf("The following is the plot for the error in iterates of beta as a function of iterations.\nThis method converges takes a very long time to converge. \n");

figure(1)
plot(err);
xlabel('Iteration')
ylabel('B error')
snapnow
disp("")
% 1c
fprintf("1c) Logistic Regression using Newton-Raphson iterates\n");
B_old = zeros(6,1);
num_iterations = 10;
err = zeros(num_iterations,1);
for i=1:num_iterations
    %calculate gradient
    u = 1./(exp(X*B_old)+1);
    delta = (1-y).*X - u.*X;
    grad = sum(delta,1)';
    %calcualte hessian matrix
    N = size(X,1);
    W = zeros(N);
    for j=1:N
        W(j,j) = exp(X(j,:)*B_old)/((exp(X(j,:)*B_old)+1).^2);
    end
    B_new = B_old - inv(X'*W*X) * grad;
    err(i) = norm(B_new-B_glm);
    B_old = B_new;
end
fprintf("The following is the plot for the error in iterates of beta as a function of iterations.\nThis method converges in just 8 iterations. \n");
figure(2)
plot(err);
xlabel('Iteration')
ylabel('B error')
snapnow;
disp("")
