load kos_doc_data.mat

W = max([A(:,2)]);  % number of unique words
D = max(A(:,1)); %nr of documents
N = sum(A(:,3)); % total number of words

beta = zeros(W,1);
c_m  = zeros(W,1);
for word=1:W
    c_m(word) = sum(A(A(:,2)==word,3));
    beta(word) = c_m(word)/N;
end

[beta,Index] = sort(beta,'descend');
c_m = sort(c_m,'descend');

Beta_Top = flip(beta(1:20));
Index_Top =flip(Index(1:20));

words = V(Index_Top);

barh(Beta_Top)
set(gca, 'YTickLabel', words, 'YTick', 1:20,'FontSize',12)
xlabel('$\beta_m$ value','Interpreter','latex')