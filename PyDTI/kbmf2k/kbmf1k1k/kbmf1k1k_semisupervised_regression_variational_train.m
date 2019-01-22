% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmf1k1k_semisupervised_regression_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    R = parameters.R;
    sigma_g = parameters.sigma_g;
    sigma_y = parameters.sigma_y;

    Lambdax.alpha = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.beta = parameters.beta_lambda * ones(Dx, R);
    Ax.mu = randn(Dx, R);
    Ax.sigma = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mu = randn(R, Nx);
    Gx.sigma = repmat(eye(R, R), [1, 1, Nx]);

    Lambdaz.alpha = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.beta = parameters.beta_lambda * ones(Dz, R);
    Az.mu = randn(Dz, R);
    Az.sigma = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mu = randn(R, Nz);
    Gz.sigma = repmat(eye(R, R), [1, 1, Nz]);

    KxKx = Kx * Kx';
    KzKz = Kz * Kz';

    lambdax_indices = repmat(logical(eye(Dx, Dx)), [1, 1, R]);
    lambdaz_indices = repmat(logical(eye(Dz, Dz)), [1, 1, R]);

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambdax
        Lambdax.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Ax.mu.^2 + reshape(Ax.sigma(lambdax_indices), Dx, R)));
        %%%% update Ax
        for s = 1:R
            Ax.sigma(:, :, s) = (diag(Lambdax.alpha(:, s) .* Lambdax.beta(:, s)) + KxKx / sigma_g^2) \ eye(Dx, Dx);
            Ax.mu(:, s) = Ax.sigma(:, :, s) * (Kx * Gx.mu(s, :)' / sigma_g^2);
        end
        %%%% update Gx
        for i = 1:Nx
            indices = ~isnan(Y(i, :));
            Gx.sigma(:, :, i) = (eye(R, R) / sigma_g^2 + (Gz.mu(:, indices) * Gz.mu(:, indices)' + sum(Gz.sigma(:, :, indices), 3)) / sigma_y^2) \ eye(R, R);
            Gx.mu(:, i) = Gx.sigma(:, :, i) * (Ax.mu' * Kx(:, i) / sigma_g^2 + Gz.mu(:, indices) * Y(i, indices)' / sigma_y^2);
        end

        %%%% update Lambdaz
        Lambdaz.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mu.^2 + reshape(Az.sigma(lambdaz_indices), Dz, R)));
        %%%% update Az
        for s = 1:R
            Az.sigma(:, :, s) = (diag(Lambdaz.alpha(:, s) .* Lambdaz.beta(:, s)) + KzKz / sigma_g^2) \ eye(Dz, Dz);
            Az.mu(:, s) = Az.sigma(:, :, s) * (Kz * Gz.mu(s, :)' / sigma_g^2);
        end
        %%%% update Gz
        for j = 1:Nz
            indices = ~isnan(Y(:, j));
            Gz.sigma(:, :, j) = (eye(R, R) / sigma_g^2 + (Gx.mu(:, indices) * Gx.mu(:, indices)' + sum(Gx.sigma(:, :, indices), 3)) / sigma_y^2) \ eye(R, R);
            Gz.mu(:, j) = Gz.sigma(:, :, j) * (Az.mu' * Kz(:, j) / sigma_g^2 + Gx.mu(:, indices) * Y(indices, j) / sigma_y^2);
        end
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.parameters = parameters;
end