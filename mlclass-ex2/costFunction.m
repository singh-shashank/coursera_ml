function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Cost calculation

temp1 =  X * theta;
h = sigmoid(temp1);

temp2 = -((y .* log(h)) + ((ones(m,1)-y) .* log(ones(m,1) - h)));
J = sum(temp2) / m;

% GRAD calculations

temp3 = h - y;

for i = 1:n
    grad(i,1) = sum(temp3 .* X(:,i)) / m;
end

% =============================================================

end
