X = load("digit/digit.txt");
labels = load("digit/labels.txt");
err=zeros(1,10);
p1_new=zeros(1,10);
p2_new=zeros(1,10);
p3_new=zeros(1,10);
for p=1:10
for l=1:10
[cluster,error]=kmeans1(X,l);
err(l)=err(l)+error;
k=l;
p1_numerator=0;
p1_denominator=0;
p2_denominator=0;
p2_numerator=0;
for i=0:9
    n=sum(labels==i);
    p1_denominator=p1_denominator+n*(n-1)/2;
    temp_list=cluster(labels==i);
    idx=zeros(size(k));
    for j=1:k
        idx(j)=sum(temp_list==j);
    end
    idx=(idx.*(idx-1))/2;
    p1_numerator=p1_numerator+sum(idx,"all");
end
p1_denominator;
p1=p1_numerator/p1_denominator;
n=size(X,1);
p2_denominator=(n*(n-1)/2)-p1_denominator;


index_set=zeros(10,k);
for i=0:9
    temp_list=cluster(labels==i);
    for j=1:k
        index_set(i+1,j)=sum(temp_list==j);
    end
end

p2_numerator=0;
for i=1:9
    for j=i+1:10
        temp_i=sum(index_set(i,:),"all");
        temp_j=sum(index_set(j,:),"all");
        temp=temp_i*temp_j-sum(index_set(i,:).*index_set(j,:),"all");
        p2_numerator=p2_numerator+temp;
    end
end

p2=p2_numerator/p2_denominator;
p3=(p1+p2)/2;

p1_new(l)=p1_new(l)+p1;
p2_new(l)=p2_new(l)+p2;
p3_new(l)=p3_new(l)+p3;
end
end


err=err/10;
p1_new=p1_new/10;
p2_new=p2_new/10;
p3_new=p3_new/10;


figure
plot(1:10,err);
xlabel('k');
ylabel('SS');


%{
figure
plot(1:10,p1_new,'DisplayName','p1');
xlabel('k');

hold on

plot(1:10,p2_new,'DisplayName','p2');

hold on

plot(1:10,p3_new,'DisplayName','p3');
legend

%}
function [cluster,error]=kmeans1(X,k)

[n,d]=size(X);
max_iter=20;
r=randperm(n);
centroid = X(r(1:k),:);
labels=zeros(n,1);
stop_flag=0;
iter=1;


while iter<=max_iter && stop_flag==0
    disp(iter);
    stop_flag=1;
    for i=1:n
        min_dist=1e15;
        temp_label=labels(i);
        for j=1:k
            temp_dist=norm(X(i,:)- centroid(j,:));
            if temp_dist<min_dist
                min_dist=temp_dist;
                labels(i)=j;
            end
        end
        
        if labels(i)~=temp_label
            stop_flag=0;
        end
    end
    if stop_flag==1
        break
    end
    
    for c=1:k
        if sum(labels==c,"all")==1
            centroid(c,:)=X(labels==c,:);
        else
            centroid(c,:) = mean(X(labels==c,:));
        end
    end
    iter=iter+1;
    
        
end

cluster=labels;
error_list=zeros(k,1);
for i=1:k
    temp_list=X(labels==i,:);
    temp_error=0;
    for j=1:size(temp_list,1)
        temp_error=temp_error+norm(temp_list(j,:)- centroid(i,:)).^2;
    end
error_list(i)=temp_error;
end
disp(size(error_list));
error=sum(error_list,'all');
end