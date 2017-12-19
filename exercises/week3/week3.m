
function predictions = predict(W1,W2,X)
  z1=logistic_neuron(W1,X);
  z2=linear_neuron(W2,z1);
  predictions=z2;
endfunction

function z = logistic_neuron(W, X)
  z = (W.') * X ;
  g = 1/(1 + exp(-z));
endfunction

function z = linear_neuron(W, X)
  z = (W.') * X ;
endfunction

W1=[0.2,-0.4,0.5;0.3,0.5,1]
W2=[2;-1;5]
X=[1;3]

z = predict(W1,W2,X)

# below passed
function predictions = predict(W1,W2, X)
  % Your code goes here.
  z1 = X * W1;
  g1 = 1. ./ (1. + exp(-z1));
  z2 = g1 * W2;
  predictions = z2;
endfunction
