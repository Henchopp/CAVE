function [] = plot_loss(log_dir, func, view_rng)

if nargin == 2
    view_rng = 1.1;
end
if view_rng < 1
    view_rng = 1;
end

% Read data
x = readmatrix(['output_logs/', log_dir, '/x.csv']);
ab = readmatrix(['output_logs/', log_dir, '/ab.csv']);
n_gd = ab(1,1); 
m = ab(2,1);
v = ab(2,2);
ab = ab(3:end,:);

minab = min(ab);
maxab = max(ab);
rngab = maxab - minab;

% Mean and variance optimization
if all(rngab > 1e-8)
    rngab = rngab * (view_rng - 1);
    a = linspace(minab(1) - rngab(1), maxab(1) + rngab(1), 50);
    b = linspace(minab(2) - rngab(2), maxab(2) + rngab(2), 50);
    [a,b] = meshgrid(a, b);

    err_surf = zeros(size(a));
    for i = 1:size(a,1)
        parfor j = 1:size(a,2)
            temp = func(a(i,j) * x + b(i,j));
            err_surf(i,j) = (mean(temp) - m)^2 + (var(temp,1) - v)^2;
        end
    end
    
    err_data = zeros(size(ab,1), 1);
    parfor i = 1:numel(err_data)
        temp = func(ab(i,1) * x + ab(i,2));
        err_data(i) = (mean(temp) - m)^2 + (var(temp,1) - v)^2;
    end
    crng = (max(err_data) - min(err_data)) * 0.25 + min(err_data);
    
    figure;
    plot3(ab(1:n_gd, 1), ab(1:n_gd, 2), err_data(1:n_gd),...
        '-or', 'LineWidth', 2);
    hold on
    plot3(ab(n_gd:end,1), ab(n_gd:end,2), err_data(n_gd:end),...
        '-og', 'LineWidth', 2);
    surf(a,b,err_surf, 'FaceAlpha', 0.7);
    xlabel('a');
    ylabel('b');
    zlabel('Loss');
    title(['Joint Opt Final Loss: ', num2str(err_data(end))]);
    set(gca, 'ColorScale', 'log');
    caxis([0, crng]);
    set(gca,'LooseInset',get(gca,'TightInset'));
   
% Mean or variance optimization
else
    [~,k] = max(rngab);
    rngab = rngab * (view_rng - 1);
    ax1 = linspace(minab(k) - rngab(k), maxab(k) + rngab(k), 1000);
    ax = zeros(numel(ax1), 2);
    ax(:,k) = ax1;
    ax(:,3-k) = ab(1,3-k);
    err_plot = zeros(numel(ax1), 1);
    parfor i = 1:numel(ax1)
        temp = func(ax(i,1) * x + ax(i,2));
        if k == 1
            err_plot(i) = (var(temp,1) - v)^2;
        else
            err_plot(i) = (mean(temp) - m)^2;
        end
    end
    
    err_data = zeros(size(ab,1), 1);
    parfor i = 1:size(ab,1)
        temp = func(ab(i,1) * x + ab(i,2));
        if k == 1
            err_data(i) = (var(temp,1) - v)^2;
        else
            err_data(i) = (mean(temp) - m)^2;
        end
    end
    
    figure;
    plot(ax1, err_plot, 'LineWidth', 2);
    hold on
    plot(ab(1:n_gd, k), err_data(1:n_gd), '-og');
    plot(ab(n_gd:end, k), err_data(n_gd:end), '-or');
    
    if k == 1
        xlabel('a');
        title(['Var Opt Final Loss: ', num2str(err_data(end))]);
    else
        xlabel('b');
        title(['Mean Opt Final Loss: ', num2str(err_data(end))]);
    end
    ylabel('Loss');
    set(gca,'LooseInset',get(gca,'TightInset'));
end

end