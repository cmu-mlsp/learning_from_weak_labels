function [bagout, bagscr, bagacc] = computebagacc(inst, instscr, baglabels,baginstcounts)

nbags = length(baglabels);
n = 1;
for i = 1:nbags
    bagout(i) = max(inst(n:n+baginstcounts(i)-1));
    bagscr(i) = max(instscr(n:n+baginstcounts(i)-1));
    n = n + baginstcounts(i);
end

bagacc = 100 * sum(bagout(:) == baglabels(:)) / nbags;
