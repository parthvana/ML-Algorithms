%HW4_Utils.genRsltFile(w,b, 'test', '111870563.mat');
% 
% [w,b,obj,alpha]=SVM_regular(trD.',trLb,10);
% predLb=valD.'*w+b;
% predLb(predLb>=0)=1;
% predLb(predLb<0)=-1;
% cp=confusionmat(valLb,predLb);
% acc=(cp(1,1)+cp(2,2))/length(predLb);
% 
% pred_value=trD.'*w+b;
% count_sv=0
% % for i=1:length(pred_value)
% %     if(trLb(i)==1)
% %         if(round(pred_value(i),2)<=1)
% %             count_sv=count_sv+1
% %         end
% %     else
% %         if(round(pred_value(i),2)>=-1)
% %             count_sv=count_sv+1
% %         end
% %     end
% % end
% 
% count_sv=sum(alpha>0,'all')

m=10
[w_all,b_all]=multi_SVM(trD.',trLb,10,m)
pred_val=zeros(size(valLb))
pred_label=zeros(size(valLb))
for i=1:m
    pred_temp=valD.'*w_all(:,i)+b_all(i)
    for j=1:length(valLb)
        if pred_temp(j)<pred_val(j)
            pred_label(j)=i
            pred_val(j)=pred_temp(j)
        end 
    end
end

acc_temp=0;
for j=1:length(pred_label)
    if pred_label(j)==valLb(j)
        acc_temp=acc_temp+1
    end
end
acc=acc_temp/length(pred_label)


%cp=confusionmat(valLb,predLb);
%acc=(cp(1,1)+cp(2,2))/length(predLb);


       