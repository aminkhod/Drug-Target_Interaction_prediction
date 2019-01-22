function [result]=AUC(args)

test_targets = args.test_targets
output = args.output

%计算AUC值,test_targets为原始样本标签,output为分类器得到的标签
%均为行或列向量
% k=1;
% output=zeros(length(test_targets1)-length(find((test_targets1)==0)),1);
% test_targets=zeros(length(test_targets1)-length(find((test_targets1)==0)),1);
% for i=1:length(test_targets1)
%     if(test_targets1(i)~=0)
%         test_targets(k)=test_targets1(i);
%         output(k)=output1(i);
%         k=k+1;
%     end
% end
[A,I]=sort(output);
    M=0;N=0;
    for i=1:length(output)
        if(test_targets(i)==1)
            M=M+1;
        end
        if(test_targets(i)~=1)
            N=N+1;
        end
    end
    sigma=0;
    for i=M+N:-1:1
        if(test_targets(I(i))==1)
            sigma=sigma+i;
        end
    end
    result=(sigma-(M+1)*M/2)/(M*N);
    if result<0
        result=0;
    end
    

