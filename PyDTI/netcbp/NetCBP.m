function [predictR]= NetCBP(args)
    Kx = args.Kx;
 	Kz = args.Kz;
 	Y = args.Y';
    alpha=args.alpha;
    beta=args.beta;
%   clear
%   clc
%    path='data\';
%      ds=2;
%     % the different datasets
%     datasets={'e','ic','gpcr','nr'};
%     fprintf('\nData Set: %s\n', datasets{ds});
% 
%     [Y,Kx,Kz,Did,Tid]=getdata(path,datasets{ds});
%     alpha=0.2;
%     beta=0.2;
  
%     a=association'
%     [r,c]=size(Y);
%    Y=Y{1:r,2:c};
%    [kxr,kxc]=size(Kx);
%    Kx=Kx{1:kxr,2:kxc};
%    [kzr,kzc]=size(Kz);
%    Kz=Kz{1:kzr,2:kzc};
    [d_r, d_c]=size(Kx); %d: drugs, r: number of rows, c: number of columns
%     normD=zeros(d_r);%construct a diagonal matrix for drug normalization
%     for i=1:d_r
%         temp=0;
%         for j=2:d_c
%             temp=temp+Kx(i,j);
%         end
%         normD(i,i)=temp;
%     end
%     norm_drug=normD^(-0.5)*Kx*normD^(-0.5);%normalize the drug matrix

    bb=sum(Kx);
    for i=1:d_r
        if(bb(1,i)~=0)
            w(:,i)=Kx(:,i)/bb(1,i);
        else
            w(:,i)=zeros(d_r,1);
        end
    end
    norm_drug=w;

    I=eye(d_r);
    d=linspace(0,0,d_r);

    for d_pos=1:d_r
        %drug position
        d(d_pos)=1;
        d_=(1-alpha)*(I-alpha*norm_drug)^(-1)*d';%calculate graph Laplacian scores of drugs 
        result_d(:,d_pos)=d_;
        d(d_pos)=0;
    end

    [p_r, p_c]=size(Kz); %p: targets, r: number of rows, c: number of columns
%     normP=zeros(p_r);%construct a diagonal matrix for target normalization
%     for i=1:p_r
%         temp=0;
%         for j=1:p_c
%             temp=temp+Kz(i,j);
%         end
%         normP(i,i)=temp;
%     end
%     norm_target=normP^(-0.5)*Kz*normP^(-0.5);%normalize the target matrix

    bb=sum(Kz);
    for i=1:p_r
        if(bb(1,i)~=0)
            T(:,i)=Kz(:,i)/bb(1,i);
        else
            T(:,i)=zeros(p_r,1);
        end
    end
    norm_target=T;

    result_p=zeros(p_r);
    p=linspace(0,0,p_c);
    E=eye(p_c);

    for p_pos=1:p_r
        p(p_pos)=1;
        p_=(1-beta)*(E-beta*norm_target)^(-1)*p';%calculate graph Laplacian scores of targets
        result_p(:,p_pos)=p_;
        p(p_pos)=0;
    end
            

    for d_pos=1:d_r

        %calculate PCC results
        for p_pos=1:p_c
             resu(d_pos,p_pos)=max(corr( result_d(:,d_pos),(Y'*result_p(:,p_pos))),corr( Y*result_d(:,d_pos),(result_p(:,p_pos))));
        end
    end
    
    predictR=resu;
end
