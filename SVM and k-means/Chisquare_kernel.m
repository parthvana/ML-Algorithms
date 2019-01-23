
function [Kernel_train, Kernel_test] = cmpExpX2Kernel(train_data, test_data, g,e)

    Kernel_train=zeros(size(train_data));
    [n,d]=size(train_data);
    padding=1:n;
    for k=1:n
        for l=1:n
           Kernel_train(k,l)=exp_chi(train_data(k,:),train_data(l,:),d,g,e);
        end
    end
    Kernel_train=[padding.' , Kernel_train];
    [m,d]=size(test_data);
    Kernel_test=zeros(m,n);
    for k=1:m
        for l=1:n
            Kernel_test(k,l)=exp_chi(test_data(k,:),train_data(l,:),d,g,e);
        end
    end
    
    padding=1:m;
    Kernel_test=[padding.',Kernel_test];
    
end

function val=exp_chi(t1,t2,d,g,e)
    temp=0;
    for i=1:d
        temp=temp+((t1(i)-t2(i)).^2)/(t1(i)+t2(i)+e);
    end
    temp=-temp/g;
    val=exp(temp);
end