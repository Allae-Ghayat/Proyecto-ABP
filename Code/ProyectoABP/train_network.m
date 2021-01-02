function net = train_network(net,x,y,l_rate,epoch,n_outputs)
	for i=1:epoch
        for j=1:size(x,1)
            row = x(j,:);
            [outputs,net] = forward_propagation(net,row);
            expected = zeros(1,n_outputs);
            expected(y(j)) = 1;
            sum_error = sum((expected - outputs).^2);
            net = error_backpropagation(net,expected);
            net = update_weights(net,x(j,:),l_rate);
        end
        %sum_error
	end
end