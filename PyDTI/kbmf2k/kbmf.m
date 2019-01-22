function predictR = kbmf(args)
	Kx = args.Kx;
	Kz = args.Kz;
	Y = args.Y;
	R = args.R;
	state = kbmf_regression_train(Kx, Kz, Y, R);
	prediction = kbmf_regression_test(Kx, Kz, state);
	predictR = prediction.Y.mu;
end