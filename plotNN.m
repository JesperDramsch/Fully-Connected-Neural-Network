function plotNN(in,neurons,out,layer)

levels = [in];
for ebenen=2:layer-1
    levels = [levels; neurons];
end
levels = [levels; out];

s=[];
t=[];
level = 1;
base=1;
for nodes = 1:sum(levels)-out+layer-1
   if nodes>sum(levels(1:level))+level
       base=base+levels(level);
       level=level+1;
   end
   for nextnode = 1:levels(level+1)
       s = [s ; nodes];
       t = [t ; nextnode+level+base+levels(level)-1];
   end
end
G = digraph(s,t);
plot(G)