% % Assuming you have a 148x148 PLV matrix stored in plv_matrix
% plv_matrix = plv_AD{1}; % Replace this with your actual PLV matrix
% 
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/DirectedSpanningTree')

% % Create the graph from the PLV matrix
% G = graph(plv_matrix);
% 
% % Find the Minimum Spanning Tree (MST) of the graph
% mst = minspantree(G);
% 
% % Visualize the MST graph
% plot(mst, 'EdgeLabel', mst.Edges.Weight);
% 
% 
% 
% 
% s = {'A' 'A' 'A' 'B' 'B' 'D' 'E' 'E' 'C'};
% t = {'D' 'B' 'E' 'D' 'E' 'E' 'F' 'C' 'F'};
% weights = [4 1 3 4 2 4 7 4 5];
% G = graph(s,t,weights);
% p = plot(G,'EdgeLabel',G.Edges.Weight);
% 
% % The output minimum spannin tree is the on produced using the
% % Prims's Algorithm (default behavior). Total Weight = 16
% mst1 = minspantree(G);
% highlight(p,mst1);
% 
% 
% %THIS IS MY CODE:
% filename = 'TABLA_AF1.xlsx';
% s = xlsread(filename,'A2:A25');
% t = xlsread(filename,'B2:B25');
% weights = xlsread(filename,'C2:C25');
% G = graph(s,t,weights);
% p = plot(G,'EdgeLabel',G.Edges.Weight);
% [T,pred] = minspantree(G);
% highlight(p,T)
% 
% 
% 
% % Assuming your PLV matrix is stored in the variable 'plvMatrix'
% % plvMatrix = yourPLVFunction(); % Replace with your PLV calculation function
% 
% % Create a graph object
% plvMatrix = plv_AD{1};
% G = graph(plvMatrix, 'OmitSelfLoops');
% 
% % Find the minimum spanning tree
% MST = minspantree(G);
% % Plot the MST
% figure;
% plot(MST);
% title('Minimum Spanning Tree of PLV Matrix');


addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/2019_03_03_BCT');
% Load the PLV connectivity matrix (replace 'your_matrix_file.mat' with the actual file name)
% Calculate the MST of the connectivity matrix using BCT
plvMatrix = triu(plv_AD{1});

%%
% % Convert the matrix to a graph object
% G = sparse(plvMatrix);
% % Obtain the maximum spanning tree of the graph
% [T, ~, W] = graphmaxspantree(G);
% % Extract the edges and weights of the maximum spanning tree
% [E1, E2] = findedge(T);
% Edges = [E1, E2];
% Weights = W(E1);
% % Display the edges and weights of the maximum spanning tree
% disp('Edges    Weight');
% disp([Edges, Weights]);

%%
% % Convert the matrix to a graph object
% G = graph(plvMatrix, 'upper');
% 
% % Obtain the maximum spanning tree of the graph using Prim's algorithm
% [T, pred] = minspantree(G, 'Type', 'tree');
% 
% % Extract the edges and weights of the maximum spanning tree
% Edges = table2array(T.Edges);
% Weights = T.EdgeTable.Weight;
% 
% % Display the edges and weights of the maximum spanning tree
% disp('Edges    Weight');
% disp([Edges, Weights]);

% G = graph(plvMatrix, 'upper');
% mst_matrix = minspantree(G);
% 
% % Analyze the properties of the MST
% % For example, calculate the degree of each node
% % node_degree = degree(mst_matrix);
% figure;
% plot(mst_matrix);
% title('Minimum Spanning Tree of PLV Matrix');


% maxST = MaximalDirectedMSF(G);
% 
% % Visualize the Maximum Spanning Tree
% plot(maxST, 'EdgeLabel', maxST.Edges.Weight);


% S = [1 1 1 2 2 3]; t=[2 3 4 3 4 4];weights=1./[0.762012728824472 0.775013750991889 0.589365182923029 0.544927843818407 0.855511678596702 0.379417644477407];
S = [];
t=[];
weights=[];
for i = 1:147
    start_node = i*ones(148-i, 1);  % Replace with your logic
    end_node = [i+1:148];    % Replace with your logic
    % Append the start and end nodes to the S and t vectors
    S = [S start_node'];
    t = [t end_node];
end
for j=1:size(S,2)
    weights(j)=1./(plvMatrix(S(j),t(j)));
end
% Convert the matrix to a graph object
G = graph(S,t,weights);
G.Edges
figure,
p = plot(G,'EdgeLabel',G.Edges.Weight);
tree = minspantree(G);
tree.Edges
highlight(p,tree)






