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





function [w,b,obj,alpha] = SVM_(X,y,C)
    
    H = [y * y.'].* [X * X.'];
    %H=double(H)
    f = -1 * ones(size(y));
    alpha = quadprog(double(H),f,zeros(size(H)),zeros(size(y)),double(y.'),0,zeros(size(y)),C*ones(size(y)));
    temp = [alpha.*y].* X;
    %disp(size(temp))
    w = sum(temp).';
    %disp(size(w))
    i=1;
    while((round((alpha(i) - C),1)==0 | alpha(i)==0)&& (i<length(y)))
        i=i+1;
    end
    disp(i)
    b = y(i) - X(i,:) * w;
    %b=y-X*w;
    %disp(size(b))
    obj = sum(alpha,'all')- 0.5 * sum((alpha * alpha.').* H,'all')
    
end

