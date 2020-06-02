load kos_doc_data.mat

M = max([A(:,2); B(:,2)]);%number of unique words
D = max(A(:,1)); %nr of documents
N = sum(A(:,3)); % total number of words

words_2001 = unique(B(B(:,1)==2001,2));
doc_2001 = B(B(:,1)==2001,:,:);


beta = zeros(M,1);
c  = zeros(M,1);
alpha = 0.01;

for m=1:M
    c(m) = sum(A(A(:,2)==m,3));
    beta(m) = (alpha + c(m))/((M*alpha)+N);
end


log_prob = 0;
c= zeros(M,1);
N_2001 = sum(doc_2001(:,3));
log_beta = log(beta);

for j = 1:M
    c(j) = sum(doc_2001(doc_2001(:,2)==j,3));
    log_prob = log_prob + c(j) * log_beta(j);
end

log_prob
N_2001
perplexity = exp(-log_prob/N_2001)