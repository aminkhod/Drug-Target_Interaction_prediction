function x=simplexproj(x,C)
x = max(0,x);
while 1 % mathematically it repeats at most D iterations; D = size(x,2)
    xi = max(0,sum(x,2)-C);
    if any(xi)
        x=max(0,bsxfun(@rdivide,bsxfun(@minus,x,xi),(sum(x~=0,2))));
    else
        break
    end
end
