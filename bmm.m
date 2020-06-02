% Bayesian Mixture of Multinomials applied to the KOS dataset

% ADVICE: consider doing clear, close all
clear all
close all
load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 10;                 % parameter of the Dirichlet over mixture components, large for topic uniformity
gamma = 0.1;                % parameter of the Dirichlet over words, small to ensure that all words come from the same topic

% Initialization: assign each document a mixture component at random
sd = ceil(K*rand(D,1));     % mixture component assignment of each document
swk = zeros(W,K);           % K multinomials over W unique words
sk_docs = zeros(K,1);            % number of documents assigned to each mixture component

% This populates count matrices swk, sk_docs and sk_words
for d = 1:D                % cycle through the documents
  w = A(A(:,1)==d,2);      % unique words in doc d
  c = A(A(:,1)==d,3);      % counts
  k = sd(d);               % doc d is in mixture k
  swk(w,k) = swk(w,k) + c; % num times word w is assigned to mixture component k
  sk_docs(k) = sk_docs(k) + 1;
end
sk_words = sum(swk,1)';    % num words assigned to mixture component k accross all docs


% perplexity_iter=zeros(50,1);
%  theta = zeros(20,1);
%  theta_1 = zeros(51,1);
%  theta_5 = zeros(51,1);
%  theta_10 = zeros(51,1);
%  theta_15 = zeros(51,1);
% theta_0 = zeros(20,1);
% theta_1 = gpml_randn(1, 20, 1);
% theta_2 = gpml_randn(2, 20, 1);
% theta_3 = gpml_randn(3, 20, 1);
% theta_4 = gpml_randn(4, 20, 1);
% theta_1 = theta_1/sum(theta_1);
% theta_2 = theta_2/sum(theta_2);
% theta_3 = theta_3/sum(theta_3);
% theta_4 = theta_4/sum(theta_4);
% theta1_0_arr=zeros(51,1);
% theta1_1_arr=zeros(51,1);
% theta1_1_arr(1)=theta_1(10);
% theta1_2_arr=zeros(51,1);
% theta1_2_arr(1)=theta_2(10);
% theta1_3_arr=zeros(51,1);
% theta1_3_arr(1)=theta_3(10);
% theta1_4_arr=zeros(51,1);
% theta1_4_arr(1)=theta_4(10);
% 
% theta5_0_arr=zeros(51,1);
% theta5_1_arr=zeros(51,1);
% theta5_1_arr(1)=theta_1(10);
% theta5_2_arr=zeros(51,1);
% theta5_2_arr(1)=theta_2(10);
% theta5_3_arr=zeros(51,1);
% theta5_3_arr(1)=theta_3(10);
% theta5_4_arr=zeros(51,1);
% theta5_4_arr(1)=theta_4(10);
% 
% theta10_0_arr=zeros(51,1);
% theta10_1_arr=zeros(51,1);
% theta10_1_arr(1)=theta_1(10);
% theta10_2_arr=zeros(51,1);
% theta10_2_arr(1)=theta_2(10);
% theta10_3_arr=zeros(51,1);
% theta10_3_arr(1)=theta_3(10);
% theta10_4_arr=zeros(51,1);
% theta10_4_arr(1)=theta_4(10);
% 
% theta15_0_arr=zeros(51,1);
% theta15_1_arr=zeros(51,1);
% theta15_1_arr(1)=theta_1(10);
% theta15_2_arr=zeros(51,1);
% theta15_2_arr(1)=theta_2(10);
% theta15_3_arr=zeros(51,1);
% theta15_3_arr(1)=theta_3(10);
% theta15_4_arr=zeros(51,1);
% theta15_4_arr(1)=theta_4(10);
% This makes a number of Gibbs sampling sweeps through all docs and words
for iter = 1:50     % number of Gibbs sweeps
  for d = 1:D       % for each document iterate through all its words
    w = A(A(:,1)==d,2);    % unique words in doc d
    c = A(A(:,1)==d,3);    % counts
    swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
    sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
    sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
    lb = zeros(1,K);    % log probability of doc d under mixture component k
    for k = 1:K
      ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
      lb(k) = log(sk_docs(k) + alpha) + ll;
    end
    b = exp(lb-max(lb));  % exponentiation of log probability plus constant
    
    kk = sampDiscrete(b); % sample from unnormalized discrete distribution
    swk(w,kk) = swk(w,kk) + c;        % add back document word counts
    sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
    sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
    sd(d) = kk;
  end

%   theta_0 = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   theta1_0_arr(iter+1)=theta_0(1);
%   theta5_0_arr(iter+1)=theta_0(5);
%   theta10_0_arr(iter+1)=theta_0(10);
%   theta15_0_arr(iter+1)=theta_0(15);
%   theta = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   theta_1(iter+1)=theta(1);
%   theta_5(iter+1)=theta(5);
%   theta_10(iter+1)=theta(10);
%   theta_15(iter+1)=theta(15);

end
% plot(theta_1)
% hold on
% plot(theta_5)
% plot(theta_10)
% plot(theta_15)
% xlabel('Gibbs Sweeps')
% ylabel('Posterior Probabilities')
% legend('$\theta_1$','$\theta_5$','$\theta_{10}$','$\theta_{15}$','Interpreter','latex')

% for iter = 1:50     % number of Gibbs sweeps
%   for d = 1:D       % for each document iterate through all its words
%     w = A(A(:,1)==d,2);    % unique words in doc d
%     c = A(A(:,1)==d,3);    % counts
%     swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
%     sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
%     sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
%     lb = zeros(1,K);    % log probability of doc d under mixture component k
%     for k = 1:K
%       ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
%       lb(k) = log(sk_docs(k) + alpha) + ll;
%     end
%     b = exp(lb-max(lb));  % exponentiation of log probability plus constant
%     
%     kk = sampDiscrete(b); % sample from unnormalized discrete distribution
%     swk(w,kk) = swk(w,kk) + c;        % add back document word counts
%     sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
%     sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
%     sd(d) = kk;
%   end
%   theta_1 = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   theta1_1_arr(iter+1)=theta_1(1);
%   theta5_1_arr(iter+1)=theta_1(5);
%   theta10_1_arr(iter+1)=theta_1(10);
%   theta15_1_arr(iter+1)=theta_1(15);
% end
% 
% for iter = 1:50     % number of Gibbs sweeps
%   for d = 1:D       % for each document iterate through all its words
%     w = A(A(:,1)==d,2);    % unique words in doc d
%     c = A(A(:,1)==d,3);    % counts
%     swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
%     sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
%     sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
%     lb = zeros(1,K);    % log probability of doc d under mixture component k
%     for k = 1:K
%       ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
%       lb(k) = log(sk_docs(k) + alpha) + ll;
%     end
%     b = exp(lb-max(lb));  % exponentiation of log probability plus constant
%     
%     kk = sampDiscrete(b); % sample from unnormalized discrete distribution
%     swk(w,kk) = swk(w,kk) + c;        % add back document word counts
%     sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
%     sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
%     sd(d) = kk;
%   end
%   theta_2 = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   
%   theta1_2_arr(iter+1)=theta_2(1);
%   theta5_2_arr(iter+1)=theta_2(5);
%   theta10_2_arr(iter+1)=theta_2(10);
%   theta15_2_arr(iter+1)=theta_2(15);
% end
% 
% for iter = 1:50     % number of Gibbs sweeps
%   for d = 1:D       % for each document iterate through all its words
%     w = A(A(:,1)==d,2);    % unique words in doc d
%     c = A(A(:,1)==d,3);    % counts
%     swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
%     sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
%     sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
%     lb = zeros(1,K);    % log probability of doc d under mixture component k
%     for k = 1:K
%       ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
%       lb(k) = log(sk_docs(k) + alpha) + ll;
%     end
%     b = exp(lb-max(lb));  % exponentiation of log probability plus constant
%     
%     kk = sampDiscrete(b); % sample from unnormalized discrete distribution
%     swk(w,kk) = swk(w,kk) + c;        % add back document word counts
%     sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
%     sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
%     sd(d) = kk;
%   end
%   theta_3 = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   theta1_3_arr(iter+1)=theta_3(1);
%   theta5_3_arr(iter+1)=theta_3(5);
%   theta10_3_arr(iter+1)=theta_3(10);
%   theta15_3_arr(iter+1)=theta_3(15);
% end
% 
% for iter = 1:50     % number of Gibbs sweeps
%   for d = 1:D       % for each document iterate through all its words
%     w = A(A(:,1)==d,2);    % unique words in doc d
%     c = A(A(:,1)==d,3);    % counts
%     swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
%     sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
%     sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
%     lb = zeros(1,K);    % log probability of doc d under mixture component k
%     for k = 1:K
%       ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
%       lb(k) = log(sk_docs(k) + alpha) + ll;
%     end
%     b = exp(lb-max(lb));  % exponentiation of log probability plus constant
%     
%     kk = sampDiscrete(b); % sample from unnormalized discrete distribution
%     swk(w,kk) = swk(w,kk) + c;        % add back document word counts
%     sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
%     sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
%     sd(d) = kk;
%   end
%   theta_4 = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);
% 
%   theta1_4_arr(iter+1)=theta_4(1);
%   theta5_4_arr(iter+1)=theta_4(5);
%   theta10_4_arr(iter+1)=theta_4(10);
%   theta15_4_arr(iter+1)=theta_4(15);
% end
% 
% tiledlayout(2,2)
% nexttile
% plot(theta1_0_arr)
% hold on
% plot(theta1_1_arr)
% hold on
% plot(theta1_2_arr)
% plot(theta1_3_arr)
% plot(theta1_4_arr)
% xlabel('Gibbs Sweeps')
% ylabel('Posterior Probability for $\theta_{1}$','Interpreter','latex')
% 
% nexttile
% plot(theta5_0_arr)
% hold on
% plot(theta5_1_arr)
% hold on
% plot(theta5_2_arr)
% plot(theta5_3_arr)
% plot(theta5_4_arr)
% xlabel('Gibbs Sweeps')
% ylabel('Posterior Probability for $\theta_{5}$','Interpreter','latex')
% 
% nexttile
% plot(theta10_0_arr)
% hold on
% plot(theta10_1_arr)
% hold on
% plot(theta10_2_arr)
% plot(theta10_3_arr)
% plot(theta10_4_arr)
% xlabel('Gibbs Sweeps')
% ylabel('Posterior Probability for $\theta_{10}$','Interpreter','latex')
% 
% nexttile
% plot(theta15_0_arr)
% hold on
% plot(theta15_1_arr)
% hold on
% plot(theta15_2_arr)
% plot(theta15_3_arr)
% plot(theta15_4_arr)
% xlabel('Gibbs Sweeps')
% ylabel('Posterior Probability for $\theta_{15}$','Interpreter','latex')


lp = 0; nd = 0;
for d = unique(B(:,1))'
  w = B(B(:,1)==d,2);    % unique words in doc d
  c = B(B(:,1)==d,3);    % counts
  z = log(sk_docs(:) + alpha) - log(sum(sk_docs(:)+alpha));
  for k = 1:K
    b = (swk(:,k)+gamma)/(sk_words(k) + gamma*W);
    z(k) = z(k) + c'*log(b(w));    % probability, doc d
  end
  lp = lp + log(sum(exp(z-max(z))))+max(z);
  nd = nd + sum(c);             % number of words, doc d
end
perplexity = exp(-lp/nd)   % perplexity
perplexity_iter(iter)=perplexity;
plot(perplexity_iter)
xlabel('Gibbs Sweeps')
ylabel('Perplexity')

% % this code allows looking at top I words for each mixture component
% I = 20;
% for k=1:K, [i ii] = sort(-swk(:,k)); ZZ(k,:)=ii(1:I); end
% for i=1:I, for k=1:K, fprintf('%-15s',V{ZZ(k,i)}); end; fprintf('\n'); end
