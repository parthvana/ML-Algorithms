load('new_data.mat')
g=[10];
c=[1000];
e=[0.000001];
%g=[0.001,0.01,0.1,1,10,25,50,100,250,500];
%c=[1,10,50,100,500,600,700,800,900,1000,1500,2000];
%e=[0.001,0.0001,0.00001];
temp_result=[];
n=size(trD,2);
for e_index=1:size(e,2)
    for c_index=1:size(c,2)
        for g_index=1:size(g,2)
            index_kfold=ml_kFoldCV_Idxs(n,5,0);
            fold_acc=0;
            for i=1:5
                % Setting train and test data for ith fold
                test_index=cell2mat(index_kfold(i));
                train_data=trD;
                train_data(:,test_index)=[];
                train_label=trLbs;
                train_label(test_index)=[];
                test_data=trD(:,test_index);
                test_label=trLbs(test_index);        
                % Calling kernel function to calculate exponential chisquare kernel    
                [trainK, testK] = cmpExpX2Kernel(train_data.', test_data.', g(g_index),e(e_index));
                % Calling svm function  
                %op='-c 500 -t 4 -q';
                op=sprintf('-c %f -t 4 -q',c(c_index));
                model=svmtrain(train_label, trainK , op);
                [predicted_label, accuracy, ~] = svmpredict(test_label,testK, model,'-q');
                %[predicted_label, accuracy, ~] = svmpredict(trLbstemp,trDtemp.',model);
                fold_acc=fold_acc+accuracy(1);
            end
            fprintf("C %f gamma %f epsilon %f accuracy %f\n",c(c_index),g(g_index),e(e_index),fold_acc/5);
            temp_result=[temp_result;[c(c_index),g(g_index),e(e_index),fold_acc/5]];
        end

    end
end