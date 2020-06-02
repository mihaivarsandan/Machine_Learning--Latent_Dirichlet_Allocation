load kos_doc_data.mat

M = max([A(:,2); B(:,2)]);%number of unique words
D = max(A(:,1)); %nr of documents
N = sum(A(:,3)); % total number of words


B_doc = unique(B(:,1));

beta = zeros(M,1);
c  = zeros(M,1);
alpha = 2000;

for m=1:M
    c(m) = sum(A(A(:,2)==m,3));
    beta(m) = 1/M;
end

B_doc = unique(B(:,1));
log_beta = log(beta);
perplexity = zeros(size(B_doc));
log_prob_B =0;

for d=1:size(B_doc,1)
    
    d_doc =B(B(:,1)==B_doc(d),:,:);
    N_doc = sum(d_doc(:,3));
    
    log_prob = 0;
    c = zeros(M,1);
    
    for j = 1:M
        c(j) = sum(d_doc(d_doc(:,2)==j,3));
        log_prob = log_prob + c(j) * log_beta(j);
       
    end
    
    log_prob_B= log_prob_B + log_prob;
    
    perplexity(d) = exp(-log_prob/N_doc);
end

bar(B_doc,perplexity)
xlabel('Documents')
ylabel('Perplexity')
sum(perplexity)
log_prob_B
    