function value = obj_cal(X,anchors,F,G,B,p)

m=size(anchors,2);
n=size(X,2);

T = - 1 / (n + m) * ones(n, m);
H_n = eye(n, n) - 1 / (n + m) * ones(n, n);
H_m = eye(m, m) - 1 / (n + m) * ones(m, m);

value = 2 * ( trace(F' * ( diag(sum(B, 2)) - p * H_n) * F) + trace(G' * ( diag(sum(B, 1)) - p * H_m) * G) - 2 * trace(F' * (B + p * T) * G));