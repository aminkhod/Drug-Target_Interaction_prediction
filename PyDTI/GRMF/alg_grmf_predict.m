function [A,B]=alg_grmf_predict(Y,A,B,Ld,Lt,lambda_l,lambda_d,lambda_t,num_iter,W)
%alg_grmf_predict is a helper function of GRMF/WGRMF that executes the
%alternating least squares method to obtain the solution.
    
    K = size(A,2);
    lambda_d_Ld = lambda_d*Ld;
    lambda_t_Lt = lambda_t*Lt;
    lambda_l_eye_K = lambda_l*eye(K);

    if nargin < 10
        for z=1:num_iter
            A = (Y*B  - lambda_d_Ld*A) / (B'*B + lambda_l_eye_K);
            B = (Y'*A - lambda_t_Lt*B) / (A'*A + lambda_l_eye_K);
        end

    else
        H = W .* Y;
        for z=1:num_iter
            A_old = A;
            HB_minus_alpha_Ld_A_old = H*B - lambda_d_Ld*A_old;
            for a=1:size(A,1)
                A(a,:) = HB_minus_alpha_Ld_A_old(a,:) / (B'*diag(W(a,:))*B + lambda_l_eye_K);
            end
            
            B_old = B;
            HtA_minus_beta_Lt_B_old = H'*A - lambda_t_Lt*B_old;
            for b=1:size(B,1)
                B(b,:) = HtA_minus_beta_Lt_B_old(b,:) / (A'*diag(W(:,b))*A + lambda_l_eye_K);
            end

%             % for readability...
%             A_old = A;
%             for i=1:size(A,1)
%                 A(i,:) = (H(i,:)*B - alpha*Ld(i,:)*A_old) / (B'*diag(W(i,:))*B + lambda*eye(k));
%             end
%             B_old = B;
%             for j=1:size(B,1)
%                 B(j,:) = (H(:,j)'*A - beta*Lt(j,:)*B_old) / (A'*diag(W(:,j))*A + lambda*eye(k));
%             end
        end
    end
    
end