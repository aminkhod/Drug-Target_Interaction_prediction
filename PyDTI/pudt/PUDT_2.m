clear
cd('..\')
addpath('pudt\libsvm-3.23\windows');

% load W.txt
% w=W;
cd('..\data\datasets\')
% %1) ENZYMES
% % % %adjacency matrix:
%     load e_admat_dgc2.txt
%     y = e_admat_dgc2(:,2:(size(e_admat_dgc2,2)));
% % % % % %compound similarity matrix:
%     load e_simmat_dc2.txt
%     kCompound = e_simmat_dc2(:,2:(size(e_simmat_dc2,2)));
% % % % % %target similarity matrix:
%     load e_simmat_dg2.txt
%     kTarget = e_simmat_dg2(:,2:(size(e_simmat_dg2,2)));
% % % % % %target_struct similarity matrix
%     load e_matrix_TMalign.txt
%     kTarget_struct = e_matrix_TMalign;
% % % % % % % %%target_FC similarity matrix
%       load e_FC.txt
%       kTarget_FC=e_FC;
% % % % % % % % % %target_GO similarity matrix
%       load e_GO.txt
%       kTarget_GO=e_GO;

% %2) ION CHANNELS
% % % % % % % % % % %adjacency matrix:
%            load ic_admat_dgc2.txt
%            y = ic_admat_dgc2(:,2:(size(ic_admat_dgc2,2)));
% % % % % % % % % % % % %compound similarity matrix:
%            load ic_simmat_dc2.txt
%            kCompound = ic_simmat_dc2(:,2:(size(ic_simmat_dc2,2)));
% % % % % % % % % % % %target similarity matrix:
%           load ic_simmat_dg2.txt
%           kTarget = ic_simmat_dg2(:,2:(size(ic_simmat_dg2,2)));
% % % %  % % % % % % %target_struct similarity matrix
%           load ic_matrix_TMalign.txt
%           kTarget_struct=ic_matrix_TMalign;
% % % %  % % % % % % %%target_FC similarity matrix
%           load ic_FC.txt
%           kTarget_FC=ic_FC;
% % % %  % % % % % % % % %target_GO similarity matrix
%           load ic_GO.txt
%           kTarget_GO=ic_GO;

% 
% % %3)GPCRs
% % % % % % % % % % % % % %adjacency matrix:
%              load gpcr_admat_dgc2.txt
%              y = gpcr_admat_dgc2(:,2:(size(gpcr_admat_dgc2,2)));
% % % % % % % % % % % % % % % %compound similarity matrix:
%              load gpcr_simmat_dc2.txt
%              kCompound = gpcr_simmat_dc2(:,2:(size(gpcr_simmat_dc2,2)));
% % % %  % % % % % % % % % % %target similarity matrix:
%              load gpcr_simmat_dg2.txt
%              kTarget = gpcr_simmat_dg2(:,2:(size(gpcr_simmat_dg2,2)));
% % % % % % % % % % % % % % % %target_struct similarity matrix
%              load gpcr_matrix_TMalign.txt
%              kTarget_struct=gpcr_matrix_TMalign;
% % % % % % % % % % % % % % %%target_FC similarity matrix
%              load gpcr_FC.txt
%              kTarget_FC=gpcr_FC;
% % % % % % % % % % % % % % % % %target_GO similarity matrix
%              load gpcr_GO.txt
%              kTarget_GO=gpcr_GO;

%4)NUCLEAR RECEPTORS
% % % % % % % % % % %adjacency matrix:
            load nr_admat_dgc2.txt
            y = nr_admat_dgc2(:,2:(size(nr_admat_dgc2,2)));
% % % % % % % % % % % %compound similarity matrix:
            load nr_simmat_dc2.txt
            kCompound = nr_simmat_dc2(:,2:(size(nr_simmat_dc2,2)));
% % % % % % % % % % % %target similarity matrix:
            load nr_simmat_dg2.txt
            kTarget = nr_simmat_dg2(:,2:(size(nr_simmat_dg2,2)));
% % % % % % % % % % % %target_struct similarity matrix
            load nr_simmat_str2.txt
            kTarget_struct=nr_simmat_str2;
% % % % % % % % % % % %%target_FC similarity matrix
           load nr_FC.txt
           kTarget_FC=nr_FC;
% % % % % % % % % % % % % %target_GO similarity matrix
           load nr_GO.txt
           kTarget_GO=nr_GO;
cd('..\')
cd('..\PyDTI\pudt\')
kCompound = (kCompound + kCompound')/2;
y_orgine=y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epsilon = .1;
while sum(eig(kCompound) >= 0) < size(kCompound,1) || isreal(eig(kCompound))==0 
    kCompound = kCompound + epsilon*eye(size(kCompound,1));
end
while sum(eig(kTarget) >= 0) < size(kTarget,1) || isreal(eig(kTarget))==0 
    kTarget = kTarget + epsilon*eye(size(kTarget,1));
end
while sum(eig(kTarget_struct) >= 0) < size(kTarget_struct,1) || isreal(eig(kTarget_struct))==0 
    kTarget_struct = kTarget_struct + epsilon*eye(size(kTarget_struct,1));
end
while sum(eig(kTarget_FC) >= 0) < size(kTarget_FC,1) || isreal(eig(kTarget_FC))==0 
    kTarget_FC = kTarget_FC + epsilon*eye(size(kTarget_FC,1));
end
while sum(eig(kTarget_GO) >= 0) < size(kTarget_GO,1) || isreal(eig(kTarget_GO))==0 
    kTarget_GO = kTarget_GO + epsilon*eye(size(kTarget_GO,1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%intialise some variables:
lengthKCompound = size(kCompound,1); 
myPredictions = zeros(lengthKCompound,size(y,1));
myPredictions11 = zeros(lengthKCompound,size(y,1));
myPredictions12 = zeros(lengthKCompound,size(y,1));
lengthKTarget = size(kTarget,1); 
myPredictions2 = zeros(lengthKTarget,size(y,2));
myPredictions21 = zeros(lengthKTarget,size(y,2));
myPredictions22 = zeros(lengthKTarget,size(y,2));
numbEdges1 = zeros(lengthKCompound,size(y,1));
numbEdges2 = zeros(lengthKTarget,size(y,2));
lengthkTarget_struct= size(kTarget_struct,1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
network=kTarget_struct;
for i=1:lengthkTarget_struct
	if sum(network(:,i))>0
		network(:,i)=kTarget_struct(:,i)/sum(network(:,i));
	end
end

THRESHOLD = 1e-10;
residue = 1;
iter = 1;
c=0.9;
for i=1:lengthKCompound
    InitalVector=y(:,i); 
    [row,col] =find(InitalVector == 1);   
    total=sum(InitalVector);
    for k=1:lengthkTarget_struct
        InitalVector(k,:)=InitalVector(k,:)/total;
    end
    p0=InitalVector;
    p=(1-c)*network*p0+c*p0;
    while (residue > THRESHOLD ),
        p1=p;
        p=(1-c)*network*p1+c*p0;
        residue         = norm(p-p1);
        iter            = iter + 1; 
    end
    pp=p;  
    finalvector=zeros(lengthkTarget_struct,1);
   [lengthRow,lengthCol]=size(row);
   ave_dis=1;
    for x=1:1:lengthRow                     
        ave_dis=ave_dis-p(row(x,1),1);
        pp(row(x,1),1)=0;
        finalvector(row(x,1),1)=1;

    end    
    ave_dis=ave_dis/(lengthkTarget_struct-lengthRow);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
   
    for j=1:lengthkTarget_struct                    
           if p(j,1)<ave_dis
               finalvector(j,1)=-1;
           end
        
    end
    y2(:,i)=finalvector;
   
end
y1=y2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%K-NN clustering
[na,nb] = size(y);
y3=y;
y3(y3==0)=-1;
K=20;
 for i=1:nb  
     inital_lable=y(:,i);
     positive_position=find(inital_lable==1);
     [nc,nd]=size(positive_position);
     for j=1:nc   
           sim_feature=kTarget_FC(positive_position(j,1),:);  
           if size(find(sim_feature==0),2)<(na-1)
             [sim_value,neighbors]=sort(sim_feature,'descend');
             sim_value=sim_value(1:K);
             neighbors=neighbors(1:K);
             [ne,nf]=size(neighbors);
             for k=1:nf
                 if neighbors(1,k)~=positive_position(j,1);
                     y3(neighbors(1,k),i)=0;
                 end
             end
           end
     end
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Heat kernel
y4=y;
beta=0.8;
iter=10;
A=kTarget_GO;
D=diag(sum(A,2));
L=eye(lengthKTarget)-D^(-1)*A;
P=(eye(lengthKTarget)-beta*L/iter)^(iter)*y;
total_P=sum(P,1);
for i=1:nb
    raw_pos=find(y(:,i)==1);
    num_pos=length(raw_pos);
    for j=1:num_pos
      total_P(1,i)=total_P(1,i)-P(raw_pos(j,1),i);
    end
    total_P(1,i)=total_P(1,i)/(na-num_pos);
    for k=1:na
        if P(k,i)< total_P(1,i)
            y4(k,i)=-1;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% vote
y5=y;
for i=1:na
    for j=1:nb
        if y5(i,j)~=1
            if (y2(i,j)+y3(i,j)+y4(i,j))<-3
                y5(i,j)=-1;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rand('state',1234567890);
[na,nb] = size(y);
opts.num_folds=5;
opts.num_repetitions=5;
 opts.missing = -1; 
division = repmat(1:opts.num_folds,1,ceil(na*nb/opts.num_folds)); 
for rep = 1:opts.num_repetitions
		which_fold{rep} = division(randperm(na*nb));   
end
	% Run the tests
	predictions = cell(opts.num_repetitions, 1);
    AUC_final = cell(opts.num_repetitions, 1);
	logs = cell(opts.num_repetitions, opts.num_folds);
    param=['-t 4 -c 1 -w1 1 -w0 0.1'];
    param1=['-t 4 -c 1 -w1 1 -w-1 0.6'];
%
replace=1;
 for fold = 1:opts.num_folds
			which = (which_fold{rep}==fold);  
            lable=y5;
            test=find(which==1);  
            train=find(which==0);  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Uall = find(lable~=1); 
            Pall = find(lable==1); 
            Utrain = setdiff(train',Pall); 
            Ptrain = setdiff(train',Utrain);
%             Utest = setdiff(test',Pall); 
%             Ptest = setdiff(test',Utest);
            Record=struct('trainneg',[],'diffset',[]);
            count = zeros(length(test),1);
            n_i=1;
            NBscore=10;
            while any(count<NBscore)
                Record(n_i).trainneg=randsample(Utrain,length(Ptrain),replace);  %从U选出和训练集中正样本相同个数的U来作为负样本
                Record(n_i).diffset=find(ismember(test, Record(n_i).trainneg)==0);
                count(Record(n_i).diffset) = count(Record(n_i).diffset)+1;
                n_i=n_i+1;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            scoretemp = NaN(length(test),n_i-1);
            for ii=1:length(Record)
                alltrain=[Ptrain;Record(ii).trainneg];
                [indTrain1,indTrain2] = ind2sub([na nb], alltrain);
                KTrain_traget=kTarget(indTrain1,indTrain1);
                KTrain_drug=kCompound(indTrain2,indTrain2);
                K_Train=KTrain_traget.*KTrain_drug;
                [indTest1,indTest2] = ind2sub([na nb], test);
                KTest_traget=kTarget(indTrain1,indTest1);
                KTest_drug=kCompound(indTrain2,indTest2);
                K_Test=KTest_traget.*KTest_drug;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                lable(lable==-1)=0;
                
                model = svmtrain(lable(alltrain), [(1:length(alltrain))',K_Train],param);
          
                [p,a,v] = svmpredict(ones(1,length(indTest1))',[(1:length(indTest1))',K_Test'],model); 
                mypre = v.*sign(p - 1/2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RL
                 lable(lable==0)=-1;
                 
                model1 = svmtrain(lable(alltrain), [(1:length(alltrain))',K_Train],param1);
             
                [p1,a1,v1] = svmpredict(ones(1,length(indTest1))',[(1:length(indTest1))',K_Test'],model1); 
                mypre1 = v1.*sign(p1 - 1/2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                mypre2 = zeros(length(mypre),1);
                final_lable =y5(test)';
                for i=1:length(mypre2)
                    if final_lable(i) ==1
                        mypre2(i)=max(mypre(i),mypre1(i));
                    end
                    if final_lable(i)==0
                        mypre2(i)=mypre(i);
                    end
                    if final_lable(i)== -1
                         mypre2(i)=mypre1(i);
                    end
                end
              scoretemp(Record(ii).diffset,ii)=mypre2;
            end
			predictions{fold} = zeros(size(test,2)',1);
            score=mean(scoretemp,2);
            predictions{fold}=score;
            args.test_targets=final_lable;
            args.output=score;
            auc(fold)=AUC(args);
            [Xlog,Ylog,Tlog,AUClog]=perfcurve(final_lable,score,'1');
            [Xpr,Ypr,Tpr,AUCpr] = perfcurve(final_lable,score, 1, 'xCrit', 'reca', 'yCrit', 'prec');
 end
1+1
% 	end
	
